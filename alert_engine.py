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


@dataclass
class Alert:
    timestamp: pd.Timestamp
    kind: str                  # volume_burst | unexplained_price_jump | explained_price_move | correlated_group_move
    tickers: list[str]
    severity: str               # Low | Medium | High
    headline: str
    detail: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["tickers"] = ", ".join(self.tickers)
        return d


NewsLookup = Callable[[str, pd.Timestamp], bool]


def build_synthetic_news_lookup(news_df: pd.DataFrame, lookback_hours: int) -> NewsLookup:
    """News-presence check backed by the synthetic news table (used in demo
    mode). Returns True if there's a headline for `ticker` within
    `lookback_hours` of `timestamp`.
    """
    if news_df is None or news_df.empty:
        return lambda ticker, ts: False

    window = pd.Timedelta(hours=lookback_hours)

    def lookup(ticker: str, ts: pd.Timestamp) -> bool:
        sub = news_df[news_df["ticker"] == ticker]
        if sub.empty:
            return False
        diffs = (sub["timestamp"] - ts).abs()
        return bool((diffs <= window).any())

    return lookup


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
        flagged = df[df["is_anomaly"]]
        for ts in flagged.index:
            explanation = explain_point(df, ts)
            top_driver = explanation["top_driver"]
            row = df.loc[ts]

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

            if top_driver == "volume_zscore":
                alerts.append(Alert(
                    timestamp=ts, kind="volume_burst", tickers=[ticker], severity=severity,
                    headline=f"{ticker}: unusual volume burst",
                    detail=(f"Volume z-score {row['volume_zscore']:.2f} "
                            f"({row['volume']:,.0f} shares) -- well outside {ticker}'s normal range."),
                    evidence={"drivers": explanation["ranked_drivers"], "volume": float(row["volume"])},
                ))
            elif top_driver == "return_zscore":
                has_news = news_lookup(ticker, ts) if news_lookup else False
                move_pct = row["return"] * 100
                if has_news:
                    alerts.append(Alert(
                        timestamp=ts, kind="explained_price_move", tickers=[ticker], severity="Low",
                        headline=f"{ticker}: large but explained price move",
                        detail=f"{move_pct:+.2f}% move, but a matching headline was found nearby -- likely news-driven.",
                        evidence={"drivers": explanation["ranked_drivers"], "return_pct": float(move_pct)},
                    ))
                else:
                    alerts.append(Alert(
                        timestamp=ts, kind="unexplained_price_jump", tickers=[ticker], severity=severity,
                        headline=f"{ticker}: unexplained price jump",
                        detail=f"{move_pct:+.2f}% move with no matching headline in the surrounding window.",
                        evidence={"drivers": explanation["ranked_drivers"], "return_pct": float(move_pct)},
                    ))
            else:
                # volatility itself was the dominant driver -- still worth a
                # (lower-severity) mention rather than silently dropping it.
                alerts.append(Alert(
                    timestamp=ts, kind="volatility_spike", tickers=[ticker], severity="Low",
                    headline=f"{ticker}: elevated volatility",
                    detail=f"Rolling volatility spiked to {row['volatility']:.4f}.",
                    evidence={"drivers": explanation["ranked_drivers"]},
                ))

    return alerts


def generate_correlation_alerts(correlation_breaks: pd.DataFrame) -> list[Alert]:
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
        severity = "High" if row["delta_zscore"] > 3 else "Medium"
        alerts.append(Alert(
            timestamp=ts, kind="correlated_group_move", tickers=list(cluster), severity=severity,
            headline=f"{', '.join(cluster)}: moving together outside normal correlation",
            detail=(f"Pairwise correlation jumped {row['corr_delta']:+.2f} vs baseline "
                    f"(z={row['delta_zscore']:.2f}) -- unusual coordinated movement."),
            evidence={"corr_delta": float(row["corr_delta"]), "delta_zscore": float(row["delta_zscore"])},
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
    return merged[["first_seen", "last_seen", "occurrences", "kind", "tickers", "severity", "headline", "detail", "evidence"]]


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
        cols = ["first_seen", "last_seen", "occurrences", "kind", "tickers", "severity", "headline", "detail", "evidence"]
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
