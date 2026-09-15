"""
Feature engineering shared by the anomaly model and the dashboard charts.

All features are rolling / causal (computed only from data up to and
including the current bar) so nothing here "cheats" by looking into the
future -- important since the pitch is about catching things as they
happen, not with hindsight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    z = (series - mean) / std.replace(0, np.nan)
    return z.fillna(0)


def build_ticker_features(
    prices: pd.Series,
    volume: pd.Series,
    returns: pd.Series,
) -> pd.DataFrame:
    """Per-ticker feature frame: one row per timestamp.

    Columns:
      return            -- raw log return this bar
      return_zscore     -- return normalized by its own rolling volatility
                            (this is what "price jump" alerts key off)
      volume_zscore     -- volume normalized vs its own rolling baseline
                            (this is what "volume burst" alerts key off)
      volatility        -- rolling std of returns (context, not alerting on its own)
    """
    return_zscore = _rolling_zscore(returns, CONFIG.return_vol_window)
    volume_zscore = _rolling_zscore(volume.astype(float), CONFIG.volume_zscore_window)
    volatility = returns.rolling(CONFIG.return_vol_window, min_periods=5).std(ddof=0).fillna(0)

    return pd.DataFrame({
        "price": prices,
        "return": returns.fillna(0),
        "return_zscore": return_zscore,
        "volume": volume,
        "volume_zscore": volume_zscore,
        "volatility": volatility,
    })


def build_all_features(market: dict) -> dict[str, pd.DataFrame]:
    """Run build_ticker_features for every ticker in a market data dict
    (the {prices, volume, returns} shape produced by data/synthetic.py or
    data/live_feed.py). Returns {ticker: feature_df}.
    """
    prices, volume, returns = market["prices"], market["volume"], market["returns"]
    out = {}
    for tkr in prices.columns:
        out[tkr] = build_ticker_features(prices[tkr], volume[tkr], returns[tkr])
    return out
