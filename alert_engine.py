"""
Fuses the three detection signals into one ranked alert feed:

  1. Volume bursts               (from models/anomaly_model.py, driven by volume_zscore)
  2. Unexplained price jumps     (from models/anomaly_model.py, driven by return_zscore,
                                   demoted to "explained" if a news check finds a headline)
  3. Correlated group moves      (from correlation_watch.py)

This is the module the Streamlit app actually calls -- it doesn't know or
care whether the underlying data was synthetic or live yfinance, only that
it received {ticker: scored_feature_df} and a correlation-breaks frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from config import CONFIG
from models.anomaly_model import explain_point
from news_relevance import score_headline_relevance


@dataclass
class Alert:
    timestamp: pd.Timestamp
    kind: str                  # volume_burst | unexplained_price_jump | explained_price_move | correlated_group_move | market_wide_move | volatility_spike
    tickers: list[str]
    severity: str               # Low | Medium | High
    headline: str
    detail: str
    evidence: dict = field(default_factory=dict)
    consensus: bool = False     # True when both the Isolation Forest AND the autoencoder flagged this bar
    confidence: float = 0.0     # 0-100 numeric confidence, alongside the Low/Medium/High severity label

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["tickers"] = ", ".join(self.tickers)
        return d


# A news lookup returns {"found": bool, "headline": str | None, "relevance": float}
# rather than a bare bool -- "a headline exists nearby" and "this headline
# explains this move" are different questions, and only the second one is
# what the problem statement actually asks for.
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
        scored = [(row["headline"], score_headline_relevance(row["headline"], ticker)) for _, row in nearby.iterrows()]
        headline, relevance = max(scored, key=lambda hr: hr[1])
        return {"found": True, "headline": headline, "relevance": relevance}

    return lookup


def _confidence_score(z: float, consensus: bool = False) -> float:
    """Collapse an anomaly-magnitude z-score into a single 0-100 confidence
    figure, so an alert's evidence reads as one glance-able number instead
    of only a Low/Medium/High label -- mirrors the brief's own example
    alert format ("Anomaly Score: 94/100"). Deliberately simple and
    documented rather than a fitted/calibrated probability: monotonically
    increasing in |z|, saturating at 100 once z is well past the alert
    thresholds already in config.py (~2.25-3.0), so the number stays
    meaningful without a hard cliff at any single value.

    Dual-model consensus (both the Isolation Forest AND the autoencoder
    independently flagging the same bar) adds a fixed boost on top, since
    that agreement is strictly stronger evidence than either model alone --
    so for the same underlying z, a consensus alert is always >= the
    single-model score.
    """
    z = abs(float(z))
    base = min(100.0, 30.0 + 14.0 * z)
    if consensus:
        base = min(100.0, base + 8.0)
    return round(max(0.0, base), 1)


def _severity_for_ticker_anomaly(scored_df: pd.DataFrame, ts: pd.Timestamp) -> str:
    score = scored_df.loc[ts, "anomaly_score"]
    flagged_scores = scored_df.loc[scored_df["is_anomaly"], "anomaly_score"]
    if len(flagged_scores) < 2:
        return "Medium"
    median = flagged_scores.median()
    return "High" if score >= median else "Medium"


def generate_ticker_alerts(
    scored_by_ticker: dict[str, pd.DataFrame],
    news_lookup: Optional[NewsLookup] = None,
) -> list[Alert]:
    """Turn per-ticker Isolation Forest flags into volume-burst /
    price-jump alerts, using SHAP's top driver to decide which kind each
    flagged bar is, and an optional news check to separate explained from
    unexplained price jumps.
    """
    alerts: list[Alert] = []

    for ticker, df in scored_by_ticker.items():
        # Use the ensemble's broader "any model flagged" net when available
        # (models/ensemble.py adds this column); fall back to the Isolation
        # Forest's own flag for callers that only ran that one model.
        flag_col = "any_model_flagged" if "any_model_flagged" in df.columns else "is_anomaly"
        flagged = df[df[flag_col]]
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

            severity = _severity_for_ticker_anomaly(df, ts)
            if is_consensus:
                severity = "High"  # two independent models agreeing overrides the usual ranking
            consensus_tag = " [AI consensus: Isolation Forest + Autoencoder]" if is_consensus else ""

            # Confidence is driven by the ensemble's own normalized
            # magnitude (models/ensemble.py's ensemble_score already blends
            # the Isolation Forest's and autoencoder's z-scaled outputs)
            # when available, falling back to the raw driver z-score for
            # callers that only ran the Isolation Forest on its own.
            driver_z = abs(float(row.get(top_driver, 0.0)))
            ensemble_z = abs(float(row["ensemble_score"])) if "ensemble_score" in df.columns and pd.notna(row.get("ensemble_score")) else driver_z
            confidence = _confidence_score(max(driver_z, ensemble_z), consensus=is_consensus)

            if top_driver == "volume_zscore":
                alerts.append(Alert(
                    timestamp=ts, kind="volume_burst", tickers=[ticker], severity=severity,
                    headline=f"{ticker}: unusual volume burst",
                    detail=(f"Volume z-score {row['volume_zscore']:.2f} "
                            f"({row['volume']:,.0f} shares) -- well outside {ticker}'s normal range.{consensus_tag}"),
                    evidence={"drivers": explanation["ranked_drivers"], "volume": float(row["volume"])},
                    consensus=is_consensus, confidence=confidence,
                ))
            elif top_driver == "return_zscore":
                news = news_lookup(ticker, ts) if news_lookup else dict(_NO_NEWS)
                has_news = news.get("found", False)
                relevance = news.get("relevance", 0.0)
                explained = has_news and relevance >= CONFIG.news_relevance_explained_threshold
                move_pct = row["return"] * 100
                if explained:
                    alerts.append(Alert(
                        timestamp=ts, kind="explained_price_move", tickers=[ticker], severity="Low",
                        headline=f"{ticker}: large but explained price move",
                        detail=(f"{move_pct:+.2f}% move -- \"{news.get('headline')}\" scored "
                                f"{relevance:.0%} relevant to this kind of move, likely news-driven."),
                        evidence={"drivers": explanation["ranked_drivers"], "return_pct": float(move_pct),
                                  "news_headline": news.get("headline"), "news_relevance": relevance},
                        consensus=is_consensus, confidence=confidence,
                    ))
                else:
                    if has_news:
                        # A headline exists nearby, but it didn't clear the
                        # relevance bar -- worth surfacing so an analyst
                        # doesn't wonder why "there's a headline" wasn't
                        # treated as an explanation.
                        detail = (f"{move_pct:+.2f}% move; a nearby headline (\"{news.get('headline')}\") scored only "
                                  f"{relevance:.0%} relevant -- not treated as an explanation.{consensus_tag}")
                    else:
                        detail = f"{move_pct:+.2f}% move with no matching headline in the surrounding window.{consensus_tag}"
                    alerts.append(Alert(
                        timestamp=ts, kind="unexplained_price_jump", tickers=[ticker], severity=severity,
                        headline=f"{ticker}: unexplained price jump",
                        detail=detail,
                        evidence={"drivers": explanation["ranked_drivers"], "return_pct": float(move_pct),
                                  "news_headline": news.get("headline"), "news_relevance": relevance},
                        consensus=is_consensus, confidence=confidence,
                    ))
            else:
                # volatility itself was the dominant driver -- still worth a
                # (lower-severity) mention rather than silently dropping it.
                alerts.append(Alert(
                    timestamp=ts, kind="volatility_spike", tickers=[ticker], severity="Low",
                    headline=f"{ticker}: elevated volatility",
                    detail=f"Rolling volatility spiked to {row['volatility']:.4f}.",
                    evidence={"drivers": explanation["ranked_drivers"]},
                    confidence=confidence,
                ))

    return alerts


def generate_correlation_alerts(correlation_breaks: pd.DataFrame) -> list[Alert]:
    """Turns each flagged correlation break into either a
    `correlated_group_move` (a genuine small-group cluster) or a
    `market_wide_move` (the whole watchlist moved together -- a broad
    event, not a targeted signal), based on the `move_kind` column
    correlation_watch.py's concentration-ratio check already computed.
    Conflating the two was the single most-findable gap in the project
    (see README/Known limitations): a Fed announcement and a genuine
    coordinated pump used to look identical to this engine.
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
                confidence=confidence,
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
                confidence=confidence,
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
    return merged[["first_seen", "last_seen", "occurrences", "kind", "tickers", "severity", "headline", "detail", "evidence", "consensus", "confidence"]]


def generate_all_alerts(
    scored_by_ticker: dict[str, pd.DataFrame],
    correlation_breaks: pd.DataFrame,
    news_lookup: Optional[NewsLookup] = None,
    include_low_severity: bool = True,
    merge_bar_minutes: Optional[float] = None,
) -> pd.DataFrame:
    """Full fused, ranked alert feed as a DataFrame (newest first).

    Consecutive alerts of the same kind/ticker(s) are merged into single
    events (see _merge_consecutive_alerts) when `merge_bar_minutes` is
    given -- pass the data's bar spacing in minutes to enable this.
    """
    alerts = generate_ticker_alerts(scored_by_ticker, news_lookup) + generate_correlation_alerts(correlation_breaks)
    if not alerts:
        cols = ["first_seen", "last_seen", "occurrences", "kind", "tickers", "severity", "headline", "detail", "evidence", "consensus", "confidence"]
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
