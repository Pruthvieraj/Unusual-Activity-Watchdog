"""
Fuses the detection signals into one ranked alert feed:

  1. Volume bursts               (from models/anomaly_model.py, driven by volume_zscore)
  2. Unexplained price jumps     (from models/anomaly_model.py, driven by return_zscore,
                                   demoted to "explained" if a news check finds a relevant headline)
  3. Correlated group moves      (from correlation_watch.py, a genuine small cluster)
  4. Market-wide moves           (from correlation_watch.py's concentration check, OR
                                   detection/regime.py's price-breadth check -- two
                                   independent ways of catching "the whole book moved",
                                   which suppresses individual unexplained-price-jump
                                   alerts on that bar in favor of one consolidated event)
  5. Momentum shifts             (from detection/momentum.py -- a short-vs-long EWMA
                                   crossover that is itself a z-score outlier; a
                                   trend-divergence signal the Isolation Forest /
                                   autoencoder ensemble doesn't look for, so it only
                                   fires on bars the ensemble did NOT already flag)

This is the module the Streamlit app actually calls -- it doesn't know or
care whether the underlying data was synthetic or live yfinance, only that
it received {ticker: scored_feature_df} and a correlation-breaks frame.

Every alert carries two distinct 0-100 numbers, deliberately kept separate
(detection/score.py has the full rationale): `confidence` (HOW UNUSUAL --
the multi-signal composite score) and `agreement_pct` (HOW MANY independent
detectors concur it's anomalous).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from config import CONFIG
from detection.momentum import compute_momentum_all
from detection.regime import compute_breadth
from detection.score import compute_composite_score, confidence_from_agreement
from models.anomaly_model import explain_point
from news_relevance import score_headline_relevance
from services import news_service


@dataclass
class Alert:
    timestamp: pd.Timestamp
    kind: str                  # volume_burst | unexplained_price_jump | explained_price_move | correlated_group_move | market_wide_move | volatility_spike | momentum_shift
    tickers: list[str]
    severity: str               # Low | Medium | High
    headline: str
    detail: str
    evidence: dict = field(default_factory=dict)
    consensus: bool = False     # True when both the Isolation Forest AND the autoencoder flagged this bar
    confidence: float = 0.0     # 0-100 composite score: HOW UNUSUAL (detection/score.py)
    agreement_pct: float = 0.0  # 0-100: HOW MANY independent detectors concur
    score_breakdown: dict = field(default_factory=dict)  # component -> points, for the explanation panel's bar chart

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["tickers"] = ", ".join(self.tickers)
        return d


# A news lookup returns {"found": bool, "headline": str | None,
# "relevance": float} plus, ONLY when found=True, "timestamp" (the
# headline's own time) -- "a headline exists nearby" and "this headline
# explains this move" are different questions, and only the second one is
# what the problem statement actually asks for. `timestamp` is what lets
# services/news_service.py report the gap between the news and the move.
#
# The not-found case deliberately stays a 3-key dict (no "timestamp"),
# matching the shape tests/test_pipeline.py's test_news_relevance_scoring
# already asserts byte-for-byte -- that test predates this Sprint-3 news
# upgrade and the plan's own Section 32 requires it keep passing
# UNMODIFIED, so the implementation conforms to it rather than the other
# way around. services/news_service.classify() only ever reads timestamp
# via .get("timestamp"), so its absence here is never a KeyError.
NewsLookup = Callable[[str, pd.Timestamp], dict]

_NO_NEWS = {"found": False, "headline": None, "relevance": 0.0}


def build_synthetic_news_lookup(news_df: pd.DataFrame, lookback_hours: int) -> NewsLookup:
    """Relevance-scored news check backed by the synthetic news table (used
    in demo mode). For a `ticker`/`timestamp`, finds the nearest headline(s)
    within `lookback_hours` and scores each with news_relevance.py's
    TF-IDF + keyword model, returning the most relevant one -- not just
    whether a headline happens to exist nearby.
    """
    if news_df is None or news_df.empty:
        return lambda ticker, ts: dict(_NO_NEWS)

    window = pd.Timedelta(hours=lookback_hours)

    def lookup(ticker: str, ts: pd.Timestamp) -> dict:
        sub = news_df[news_df["ticker"] == ticker]
        if sub.empty:
            return dict(_NO_NEWS)
        diffs = (sub["timestamp"] - ts).abs()
        nearby = sub[diffs <= window]
        if nearby.empty:
            return dict(_NO_NEWS)
        scored = [(row["headline"], score_headline_relevance(row["headline"], ticker), row["timestamp"])
                  for _, row in nearby.iterrows()]
        headline, relevance, headline_ts = max(scored, key=lambda hr: hr[1])
        return {"found": True, "headline": headline, "relevance": relevance, "timestamp": headline_ts}

    return lookup


def _confidence_score(z: float, consensus: bool = False) -> float:
    """Composite score for a CORRELATION-kind alert (correlated_group_move
    / market_wide_move). detection/score.py's 6-component blend is built
    for PER-TICKER events (price/volume/news/etc. all genuinely apply
    there); a correlation event's entire nature IS the cross-stock signal,
    so folding it into just that formula's 9-point cross-stock-sync slice
    would produce a misleadingly low number for what's often a High-severity
    event. Deliberately kept as its own small, documented, monotonic,
    saturating formula instead -- z=0 -> 30, saturating near 100 by z~5,
    same shape philosophy as detection/score.py, sized for when
    cross-stock correlation IS the whole story.
    """
    z = abs(float(z))
    base = min(100.0, 30.0 + 14.0 * z)
    if consensus:
        base = min(100.0, base + 8.0)
    return round(max(0.0, base), 1)


def _momentum_alert_score(crossover_zscore: float) -> float:
    """Composite score for a `momentum_shift` alert. Same small, documented,
    monotonic, saturating shape as `_confidence_score` -- sized for
    momentum's own alert threshold (`momentum_shift_zscore_alert`, default
    2.25) rather than detection/score.py's per-ticker component caps, since
    a crossover z-score isn't one of that formula's six inputs.
    """
    z = abs(float(crossover_zscore))
    return round(min(100.0, max(0.0, 25.0 + 15.0 * z)), 1)


def _severity_for_ticker_anomaly(scored_df: pd.DataFrame, ts: pd.Timestamp) -> str:
    score = scored_df.loc[ts, "anomaly_score"]
    flagged_scores = scored_df.loc[scored_df["is_anomaly"], "anomaly_score"]
    if len(flagged_scores) < 2:
        return "Medium"
    median = flagged_scores.median()
    return "High" if score >= median else "Medium"


def _cluster_membership(correlation_breaks: Optional[pd.DataFrame]) -> dict:
    """{timestamp: set(tickers)} for bars correlation_watch.py classified as
    a genuine small cluster (move_kind == "correlated_group_move") -- used
    by the composite score's cross-stock-sync component. Market-wide bars
    are deliberately excluded here: "is this ticker part of a targeted
    cluster right now" should be False during a broad move, even though
    the ticker is technically listed in that break's (whole-watchlist)
    `cluster` column -- see correlation_watch.py.
    """
    if correlation_breaks is None or correlation_breaks.empty:
        return {}
    genuine = correlation_breaks[
        correlation_breaks.get("is_break", False)
        & (correlation_breaks.get("move_kind") == "correlated_group_move")
    ]
    return {ts: set(row["cluster"]) for ts, row in genuine.iterrows()}


def generate_ticker_alerts(
    scored_by_ticker: dict[str, pd.DataFrame],
    news_lookup: Optional[NewsLookup] = None,
    correlation_breaks: Optional[pd.DataFrame] = None,
    breadth: Optional[pd.Series] = None,
    momentum_by_ticker: Optional[dict[str, pd.DataFrame]] = None,
) -> list[Alert]:
    """Turn per-ticker ensemble flags into volume-burst / price-jump
    alerts, using SHAP's top driver to decide which kind each flagged bar
    is, an optional news check to separate explained from unexplained
    price jumps, and detection/score.py's composite for a single 0-100
    number per alert.

    `breadth` (detection.regime.compute_breadth output) lets this function
    suppress an individual "unexplained price jump" alert on a bar where
    most of the watchlist moved at once, consolidating them into one
    market_wide_move alert instead (emitted at the end of this function) --
    see detection/regime.py for why that distinction matters.

    `momentum_by_ticker` (detection.momentum.compute_momentum_all output) is
    optional -- when given, a `momentum_shift` alert fires for a ticker/bar
    where the EWMA crossover itself is a z-score outlier, but ONLY on bars
    the Isolation Forest / autoencoder ensemble did NOT already flag: this
    is a genuinely different question ("did the trend just bend sharply")
    than a single-bar price/volume outlier, so it's additive rather than a
    restatement of the same evidence.
    """
    alerts: list[Alert] = []
    cluster_by_ts = _cluster_membership(correlation_breaks)
    delta_zscore_by_ts = (
        correlation_breaks["delta_zscore"].to_dict()
        if correlation_breaks is not None and not correlation_breaks.empty else {}
    )
    suppressed_market_wide: dict[pd.Timestamp, list[str]] = {}
    ensemble_flagged_ts_by_ticker: dict[str, set] = {}

    for ticker, df in scored_by_ticker.items():
        # Use the ensemble's broader "any model flagged" net when available
        # (models/ensemble.py adds this column); fall back to the Isolation
        # Forest's own flag for callers that only ran that one model.
        flag_col = "any_model_flagged" if "any_model_flagged" in df.columns else "is_anomaly"
        flagged = df[df[flag_col]]
        ensemble_flagged_ts_by_ticker[ticker] = set(flagged.index)
        for ts in flagged.index:
            explanation = explain_point(df, ts)
            top_driver = explanation["top_driver"]
            row = df.loc[ts]
            is_consensus = bool(row["consensus"]) if "consensus" in df.columns else False

            # Isolation Forest's `contamination` parameter guarantees a
            # fixed ~5% of bars get flagged as "relatively" anomalous EVEN
            # on pure noise -- that's how the algorithm works, not a bug,
            # but it means "is_anomaly" alone isn't enough to alert an
            # analyst on. Gate on the configured absolute z-score
            # thresholds too, so an alert only fires when the move is both
            # relatively unusual (top of its own distribution) AND
            # absolutely large (the kind of z-score a human would actually
            # call "unusual"). This is what makes the config's
            # volume_zscore_alert / price_jump_sigma_alert thresholds real
            # rather than decorative.
            if top_driver == "volume_zscore" and abs(row["volume_zscore"]) < CONFIG.volume_zscore_alert:
                continue
            if top_driver == "return_zscore" and abs(row["return_zscore"]) < CONFIG.price_jump_sigma_alert:
                continue

            # Market-wide suppression (detection/regime.py): if most of the
            # watchlist is moving unusually on this SAME bar, an individual
            # "unexplained price jump" alert for one ticker is misleading --
            # it's not stock-specific. Suppress it here; one consolidated
            # market_wide_move alert is emitted for the whole bar below,
            # after every ticker has been processed, so nothing is silently
            # dropped, only re-grouped.
            if top_driver == "return_zscore" and breadth is not None and ts in breadth.index \
                    and breadth.loc[ts] >= CONFIG.regime_breadth_cutoff:
                suppressed_market_wide.setdefault(ts, []).append(ticker)
                continue

            severity = _severity_for_ticker_anomaly(df, ts)
            if is_consensus:
                severity = "High"  # two independent models agreeing overrides the usual ranking
            consensus_tag = " [AI consensus: Isolation Forest + Autoencoder]" if is_consensus else ""

            # Agreement (HOW MANY independent detectors concur): out of the
            # detectors that actually ran for this bar.
            n_total = 2 if "ae_is_anomaly" in df.columns else 1
            n_agree = int(bool(row.get("is_anomaly", False)))
            if "ae_is_anomaly" in df.columns:
                n_agree += int(bool(row.get("ae_is_anomaly", False)))
            agreement_pct = confidence_from_agreement(n_agree, n_total)

            in_cluster = ticker in cluster_by_ts.get(ts, set())
            cross_stock_delta_z = delta_zscore_by_ts.get(ts) if in_cluster else None

            if top_driver == "volume_zscore":
                breakdown = compute_composite_score(
                    return_zscore=row.get("return_zscore", 0.0), volume_zscore=row["volume_zscore"],
                    volatility_zscore=row.get("volatility_zscore", 0.0),
                    news_relevance=None,  # no news check attempted for a volume-driven alert
                    cross_stock_delta_zscore=cross_stock_delta_z, in_active_cluster=in_cluster,
                    consensus=is_consensus,
                )
                alerts.append(Alert(
                    timestamp=ts, kind="volume_burst", tickers=[ticker], severity=severity,
                    headline=f"{ticker}: unusual volume burst",
                    detail=(f"Volume z-score {row['volume_zscore']:.2f} "
                            f"({row['volume']:,.0f} shares) -- well outside {ticker}'s normal range.{consensus_tag}"),
                    evidence={"drivers": explanation["ranked_drivers"], "volume": float(row["volume"])},
                    consensus=is_consensus, confidence=breakdown.total,
                    agreement_pct=agreement_pct, score_breakdown=breakdown.as_dict(),
                ))
            elif top_driver == "return_zscore":
                news = news_lookup(ticker, ts) if news_lookup else None
                classification = news_service.classify(ts, news)
                explained = classification.state == news_service.NEWS_EXPLAINS
                relevance = classification.relevance
                move_pct = row["return"] * 100
                news_evidence = {
                    "news_headline": classification.headline, "news_relevance": relevance,
                    "news_state": classification.state, "news_gap_text": classification.gap_text,
                }
                # news_relevance=None means "no check ran"; here a check DID
                # run (news_lookup is not None), so pass its actual result
                # (0.0 counts as "checked, found nothing relevant").
                news_relevance_for_score = relevance if news_lookup else None
                breakdown = compute_composite_score(
                    return_zscore=row["return_zscore"], volume_zscore=row.get("volume_zscore", 0.0),
                    volatility_zscore=row.get("volatility_zscore", 0.0),
                    news_relevance=news_relevance_for_score,
                    cross_stock_delta_zscore=cross_stock_delta_z, in_active_cluster=in_cluster,
                    consensus=is_consensus,
                )
                if explained:
                    alerts.append(Alert(
                        timestamp=ts, kind="explained_price_move", tickers=[ticker], severity="Low",
                        headline=f"{ticker}: large but explained price move",
                        detail=f"{move_pct:+.2f}% move -- {classification.detail}",
                        evidence={"drivers": explanation["ranked_drivers"], "return_pct": float(move_pct), **news_evidence},
                        consensus=is_consensus, confidence=breakdown.total,
                        agreement_pct=agreement_pct, score_breakdown=breakdown.as_dict(),
                    ))
                else:
                    detail = f"{move_pct:+.2f}% move; {classification.detail}{consensus_tag}"
                    alerts.append(Alert(
                        timestamp=ts, kind="unexplained_price_jump", tickers=[ticker], severity=severity,
                        headline=f"{ticker}: unexplained price jump",
                        detail=detail,
                        evidence={"drivers": explanation["ranked_drivers"], "return_pct": float(move_pct), **news_evidence},
                        consensus=is_consensus, confidence=breakdown.total,
                        agreement_pct=agreement_pct, score_breakdown=breakdown.as_dict(),
                    ))
            else:
                # volatility itself was the dominant driver -- still worth a
                # (lower-severity) mention rather than silently dropping it.
                breakdown = compute_composite_score(
                    return_zscore=row.get("return_zscore", 0.0), volume_zscore=row.get("volume_zscore", 0.0),
                    volatility_zscore=row.get("volatility_zscore", 0.0),
                    news_relevance=None, cross_stock_delta_zscore=cross_stock_delta_z,
                    in_active_cluster=in_cluster, consensus=is_consensus,
                )
                alerts.append(Alert(
                    timestamp=ts, kind="volatility_spike", tickers=[ticker], severity="Low",
                    headline=f"{ticker}: elevated volatility",
                    detail=f"Rolling volatility spiked to {row['volatility']:.4f}.",
                    evidence={"drivers": explanation["ranked_drivers"]},
                    confidence=breakdown.total, agreement_pct=agreement_pct, score_breakdown=breakdown.as_dict(),
                ))

    # Momentum shifts (detection/momentum.py): a different question than the
    # ensemble's per-bar outlier check, so only surfaced on bars the
    # ensemble didn't already flag for this ticker -- see the docstring
    # above for why these two signals are additive, not redundant.
    for ticker, mdf in (momentum_by_ticker or {}).items():
        if "is_shift" not in mdf.columns:
            continue
        already_flagged = ensemble_flagged_ts_by_ticker.get(ticker, set())
        shifted = mdf[mdf["is_shift"]]
        for ts in shifted.index:
            if ts in already_flagged:
                continue
            row = mdf.loc[ts]
            z = float(row["crossover_zscore"])
            score = _momentum_alert_score(z)
            direction = "accelerating upward" if row["crossover"] > 0 else "accelerating downward"
            alerts.append(Alert(
                timestamp=ts, kind="momentum_shift", tickers=[ticker],
                severity="High" if abs(z) >= 3.5 else "Medium",
                headline=f"{ticker}: short-term trend {direction} sharply",
                detail=(f"{ticker}'s short EWMA crossed its long EWMA by an amount that is itself a z-score "
                        f"outlier (z={z:.2f}) against this stock's own recent history -- a trend-divergence "
                        f"signal distinct from a single-bar price or volume spike."),
                evidence={"ewma_short": float(row["ewma_short"]), "ewma_long": float(row["ewma_long"]),
                          "crossover": float(row["crossover"]), "crossover_zscore": z},
                confidence=score, agreement_pct=confidence_from_agreement(1, 1),
                score_breakdown={"Momentum crossover (z-score)": round(score, 1)},
            ))

    # Consolidate every market-wide-suppressed bar into ONE alert per bar,
    # naming the breadth percentage -- the concrete fix for "how do you
    # tell a market crash from a coordinated pump" (detection/regime.py).
    for ts, tickers_here in suppressed_market_wide.items():
        pct = float(breadth.loc[ts]) * 100
        tickers_sorted = sorted(set(tickers_here))
        # A bespoke, documented score for this alert type: breadth IS the
        # signal here (not a per-ticker anomaly magnitude), so it's scored
        # directly on a 0-100 scale keyed to the breadth percentage, the
        # same "simple, monotonic, saturating, explainable" philosophy as
        # every other score in this file, just with a different input.
        score = round(min(100.0, 40.0 + pct * 0.75), 1)
        alerts.append(Alert(
            timestamp=ts, kind="market_wide_move", tickers=tickers_sorted,
            severity="Medium" if pct >= 80 else "Low",
            headline=f"Market-wide move: {pct:.0f}% of the watchlist moved unusually in this bar",
            detail=(f"{len(tickers_sorted)} of {len(scored_by_ticker)} watched tickers each individually cleared "
                    f"the price-jump threshold on the same bar -- a broad move, not a stock-specific anomaly, so "
                    f"individual unexplained-price-jump alerts for this bar were consolidated into this one event."),
            evidence={"breadth_pct": pct, "tickers_moved": tickers_sorted},
            confidence=score, agreement_pct=round(pct, 1),
            score_breakdown={"Breadth (% of watchlist)": round(pct, 1)},
        ))

    return alerts


def generate_correlation_alerts(
    correlation_breaks: pd.DataFrame,
    breadth: Optional[pd.Series] = None,
) -> list[Alert]:
    """Turns each flagged correlation break into either a
    `correlated_group_move` (a genuine small-group cluster) or a
    `market_wide_move` (the whole watchlist moved together -- a broad
    event, not a targeted signal), based on the `move_kind` column
    correlation_watch.py's concentration-ratio check already computed.
    Conflating the two was the single most-findable gap in the project
    (see README/Known limitations): a Fed announcement and a genuine
    coordinated pump used to look identical to this engine.

    `breadth` (detection.regime.compute_breadth) is used only to compute
    this alert's `agreement_pct`: whether the correlation engine's own
    signal is corroborated by the INDEPENDENT price-breadth check at the
    same bar -- two different methods agreeing is worth surfacing even
    when only one of them determines the alert's kind/severity.
    """
    alerts: list[Alert] = []
    flagged = correlation_breaks[correlation_breaks["is_break"]]
    for ts, row in flagged.iterrows():
        cluster = row["cluster"]
        if not cluster:
            continue
        # Sort so the same set of tickers always joins to the same string
        # (needed for _merge_consecutive_alerts to recognize repeat bars of
        # the same cluster as one ongoing event rather than new ones).
        cluster = sorted(cluster)
        move_kind = row.get("move_kind") or "correlated_group_move"
        confidence = _confidence_score(row["delta_zscore"])
        breadth_agrees = breadth is not None and ts in breadth.index and breadth.loc[ts] >= CONFIG.regime_breadth_cutoff
        agreement_pct = confidence_from_agreement(1 + int(breadth_agrees), 2)
        breakdown = {"Correlation delta z-score": round(float(row["delta_zscore"]), 2),
                     "Concentration ratio": round(float(row.get("concentration_ratio", 0)), 2)}

        if move_kind == "market_wide_move":
            # Deliberately lower urgency than a genuine cluster even at the
            # same delta_zscore -- a broad market move isn't the "suspicious
            # coordinated group" pattern the problem statement describes.
            severity = "Medium" if row["delta_zscore"] > 3 else "Low"
            alerts.append(Alert(
                timestamp=ts, kind="market_wide_move", tickers=list(cluster), severity=severity,
                headline="Broad market-wide move across the watchlist (not a targeted cluster)",
                detail=(f"Pairwise correlation jumped {row['corr_delta']:+.2f} vs baseline (z={row['delta_zscore']:.2f}), "
                        f"but concentration ratio {row.get('concentration_ratio', 0):.2f}x is below the "
                        f"{CONFIG.correlation_concentration_ratio_threshold:.1f}x cluster threshold -- essentially the "
                        "whole watchlist moved together, not a suspicious subset."),
                evidence={"corr_delta": float(row["corr_delta"]), "delta_zscore": float(row["delta_zscore"]),
                          "concentration_ratio": float(row.get("concentration_ratio", 0))},
                confidence=confidence, agreement_pct=agreement_pct, score_breakdown=breakdown,
            ))
        else:
            severity = "High" if row["delta_zscore"] > 3 else "Medium"
            alerts.append(Alert(
                timestamp=ts, kind="correlated_group_move", tickers=list(cluster), severity=severity,
                headline=f"{', '.join(cluster)}: moving together outside normal correlation",
                detail=(f"Pairwise correlation jumped {row['corr_delta']:+.2f} vs baseline "
                        f"(z={row['delta_zscore']:.2f}), concentrated {row.get('concentration_ratio', 0):.2f}x above "
                        "the rest of the book -- unusual coordinated movement, not a broad market move."),
                evidence={"corr_delta": float(row["corr_delta"]), "delta_zscore": float(row["delta_zscore"]),
                          "concentration_ratio": float(row.get("concentration_ratio", 0))},
                confidence=confidence, agreement_pct=agreement_pct, score_breakdown=breakdown,
            ))
    return alerts


def _merge_consecutive_alerts(df: pd.DataFrame, max_gap_bars: int, bar_minutes: float) -> pd.DataFrame:
    """A single real event (e.g. a sustained correlated move, or a volume
    burst that spans several bars) tends to trip the detector on every bar
    it lasts for, which would otherwise flood the feed with near-duplicate
    alerts. This collapses consecutive alerts that share the same kind and
    ticker set into one event spanning [first_seen, last_seen], so the feed
    reads as "N distinct events" instead of "N x (however many bars each
    event lasted)".
    """
    if df.empty:
        return df

    max_gap = pd.Timedelta(minutes=max_gap_bars * bar_minutes)
    df = df.sort_values(["tickers", "kind", "timestamp"])

    merged_rows = []
    for (tickers, kind), group in df.groupby(["tickers", "kind"], sort=False):
        group = group.sort_values("timestamp")
        current = None
        for _, row in group.iterrows():
            if current is not None and row["timestamp"] - current["last_seen"] <= max_gap:
                current["last_seen"] = row["timestamp"]
                current["occurrences"] += 1
                if row["severity"] == "High":
                    current["severity"] = "High"
                current["detail"] = row["detail"]  # keep most recent wording
                current["evidence"] = row["evidence"]
                current["consensus"] = current.get("consensus", False) or row.get("consensus", False)
                current["confidence"] = max(current.get("confidence", 0.0), row.get("confidence", 0.0))
                current["agreement_pct"] = max(current.get("agreement_pct", 0.0), row.get("agreement_pct", 0.0))
                current["score_breakdown"] = row.get("score_breakdown", current.get("score_breakdown", {}))
            else:
                if current is not None:
                    merged_rows.append(current)
                current = row.to_dict()
                current["last_seen"] = row["timestamp"]
                current["occurrences"] = 1
        if current is not None:
            merged_rows.append(current)

    merged = pd.DataFrame(merged_rows)
    merged = merged.rename(columns={"timestamp": "first_seen"})
    return merged[["first_seen", "last_seen", "occurrences", "kind", "tickers", "severity", "headline", "detail",
                   "evidence", "consensus", "confidence", "agreement_pct", "score_breakdown"]]


def generate_all_alerts(
    scored_by_ticker: dict[str, pd.DataFrame],
    correlation_breaks: pd.DataFrame,
    news_lookup: Optional[NewsLookup] = None,
    include_low_severity: bool = True,
    merge_bar_minutes: Optional[float] = None,
    breadth: Optional[pd.Series] = None,
    momentum_by_ticker: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Full fused, ranked alert feed as a DataFrame (newest first).

    Consecutive alerts of the same kind/ticker(s) are merged into single
    events (see _merge_consecutive_alerts) when `merge_bar_minutes` is
    given -- pass the data's bar spacing in minutes to enable this.

    `breadth` (detection.regime.compute_breadth) is optional -- pass it to
    enable market-wide-move suppression/consolidation on ticker alerts and
    the cross-check agreement_pct on correlation alerts. Omitting it keeps
    prior behavior (no suppression, correlation alerts single-sourced).

    `momentum_by_ticker` (detection.momentum.compute_momentum_all) is also
    optional and auto-computed from `scored_by_ticker`'s `return` column
    when omitted -- every scored-feature frame carries `return` (see
    features.py), so this needs no extra pipeline wiring from callers.
    """
    if breadth is None:
        breadth = compute_breadth(scored_by_ticker) if scored_by_ticker else None
    if momentum_by_ticker is None and scored_by_ticker:
        momentum_by_ticker = compute_momentum_all(scored_by_ticker)
    alerts = (generate_ticker_alerts(scored_by_ticker, news_lookup, correlation_breaks, breadth, momentum_by_ticker)
              + generate_correlation_alerts(correlation_breaks, breadth))
    if not alerts:
        cols = ["first_seen", "last_seen", "occurrences", "kind", "tickers", "severity", "headline", "detail",
                "evidence", "consensus", "confidence", "agreement_pct", "score_breakdown"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame([a.to_dict() for a in alerts])

    if merge_bar_minutes:
        df = _merge_consecutive_alerts(df, max_gap_bars=3, bar_minutes=merge_bar_minutes)
        sort_col = "last_seen"
    else:
        df = df.rename(columns={"timestamp": "first_seen"})
        df["last_seen"] = df["first_seen"]
        df["occurrences"] = 1
        sort_col = "first_seen"

    if not include_low_severity:
        df = df[df["severity"] != "Low"]

    severity_rank = {"High": 0, "Medium": 1, "Low": 2}
    df["_severity_rank"] = df["severity"].map(severity_rank)
    df = df.sort_values([sort_col, "_severity_rank"], ascending=[False, True]).drop(columns="_severity_rank")
    return df.reset_index(drop=True)
