"""
Momentum-shift detection: a short EWMA crossing a long EWMA on returns,
flagged only when the crossover itself is a z-score outlier against its
own recent history -- the same self-calibrating pattern
correlation_watch.py already uses (an expanding-history z-score instead of
a fixed constant), applied to a new signal rather than a new philosophy.

Explicitly a low-cost, statistically-grounded addition, not a trading
signal or a forecast: it answers "did this stock's short-term trend just
diverge sharply from its longer-term trend, in a way that's unusual even
for this stock" -- a different question than return_zscore (a single bar's
size) or volume_zscore (a single bar's volume).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG


def compute_momentum_signal(returns: pd.Series) -> pd.DataFrame:
    """Returns a DataFrame indexed like `returns` with columns:
      ewma_short, ewma_long, crossover, crossover_zscore, is_shift
    """
    short = returns.ewm(span=CONFIG.momentum_short_span, adjust=False).mean()
    long_ = returns.ewm(span=CONFIG.momentum_long_span, adjust=False).mean()
    crossover = short - long_

    min_hist = CONFIG.momentum_history_window
    mean = crossover.expanding(min_periods=min_hist).mean().shift(1)
    std = crossover.expanding(min_periods=min_hist).std(ddof=0).shift(1)
    z = ((crossover - mean) / std.replace(0, np.nan)).fillna(0)

    out = pd.DataFrame({
        "ewma_short": short,
        "ewma_long": long_,
        "crossover": crossover,
        "crossover_zscore": z,
    })
    out["is_shift"] = out["crossover_zscore"].abs() >= CONFIG.momentum_shift_zscore_alert
    return out


def compute_momentum_all(features_by_ticker: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {tkr: compute_momentum_signal(df["return"]) for tkr, df in features_by_ticker.items()}
