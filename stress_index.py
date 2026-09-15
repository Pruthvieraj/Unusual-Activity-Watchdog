"""
Composite 0-100 "market stress index" -- a single headline number combining
all three detection signals so a viewer (or a judge glancing at the screen
for five seconds) gets the state of the whole watchlist at once, without
reading a table.

    stress = 40% cross-watchlist volatility regime
           + 35% correlation-break severity
           + 25% fraction of tickers currently flagged by the AI ensemble

Weights are a simple, documented, hand-set blend (not fit to data) -- the
point is a legible, explainable composite, not a tuned trading signal.
"""

from __future__ import annotations

import pandas as pd

VOL_WEIGHT = 0.40
CORR_WEIGHT = 0.35
FLAG_WEIGHT = 0.25


def compute_stress_series(scored_by_ticker: dict[str, pd.DataFrame], correlation_breaks: pd.DataFrame) -> pd.Series:
    idx = correlation_breaks.index
    if len(idx) == 0:
        return pd.Series(dtype=float)

    z_frame = pd.DataFrame({tkr: df["return_zscore"].reindex(idx) for tkr, df in scored_by_ticker.items()})
    vol_scaled = (z_frame.abs().mean(axis=1) / 4.0).clip(0, 1).fillna(0)

    corr_scaled = (correlation_breaks["delta_zscore"].clip(lower=0) / 4.0).clip(0, 1).fillna(0)

    flag_col = "any_model_flagged" if all("any_model_flagged" in df.columns for df in scored_by_ticker.values()) else "is_anomaly"
    flag_frame = pd.DataFrame({tkr: df[flag_col].reindex(idx).fillna(False) for tkr, df in scored_by_ticker.items()})
    flag_scaled = flag_frame.mean(axis=1).clip(0, 1)

    stress = 100 * (VOL_WEIGHT * vol_scaled + CORR_WEIGHT * corr_scaled + FLAG_WEIGHT * flag_scaled)
    return stress.clip(0, 100)


def stress_label(value: float) -> tuple[str, str]:
    """Returns (label, status_color_hex) for a stress value."""
    if value >= 70:
        return "Critical", "#e66767"
    if value >= 45:
        return "Elevated", "#c98500"
    if value >= 20:
        return "Watchful", "#3987e5"
    return "Calm", "#199e70"
