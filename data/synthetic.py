"""
Synthetic multi-stock market data generator.

Why this exists: free real-time trade-tick data isn't available without a
paid feed, and yfinance is rate-limited / sometimes unreachable depending on
network conditions. For a hackathon demo you cannot depend on "something
weird happens to actually happen live" -- so this module generates a
realistic multi-stock price/volume tape with KNOWN, INJECTED anomalies:

  1. Volume bursts        -- one ticker's volume spikes for a few bars
  2. Unexplained price jumps -- a discontinuous return with no matching
     "news" headline
  3. Correlated group moves  -- a subset of tickers move together sharply,
     well outside their historical co-movement baseline

Because the ground truth is known, this also lets us report a precision/
recall sanity check for the detector (see tests/test_pipeline.py), which is
a much stronger demo story than "trust me, it works."
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG


def _generate_correlated_returns(n_bars: int, tickers: list[str], seed: int) -> pd.DataFrame:
    """Generate baseline (non-anomalous) log-returns with a realistic
    correlation structure: a shared market factor plus per-ticker noise,
    with tech names loaded more heavily on the market factor than others.
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)

    # Market-factor loadings: first few tickers behave like "tech", the
    # rest are more idiosyncratic -- gives the correlation engine a genuine
    # contrast between "normally co-moving" and "normally independent".
    loadings = np.array([0.85, 0.8, 0.8, 0.75, 0.75] + [0.3] * max(0, n - 5))[:n]

    market = rng.normal(0, 0.0015, n_bars)
    idio = rng.normal(0, 0.0022, size=(n_bars, n))

    returns = np.outer(market, loadings) + idio
    return pd.DataFrame(returns, columns=tickers)


def _base_volume(n_bars: int, tickers: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    base_levels = rng.integers(2_000_00, 8_000_00, size=len(tickers))
    vol = np.abs(rng.normal(loc=base_levels, scale=base_levels * 0.15, size=(n_bars, len(tickers))))
    return pd.DataFrame(vol.astype(int), columns=tickers)


def generate_market_tape(
    tickers: list[str] | None = None,
    n_bars: int | None = None,
    freq_minutes: int | None = None,
    n_anomalies: int | None = None,
    seed: int = 7,
) -> dict:
    """Build a synthetic OHLCV-ish tape plus a news table and ground-truth
    anomaly labels.

    Returns a dict with:
      prices:      DataFrame [timestamp x ticker] close price
      volume:      DataFrame [timestamp x ticker] volume
      returns:     DataFrame [timestamp x ticker] log return
      news:        DataFrame[ticker, timestamp, headline] -- one row per
                   headline, sparse (only near "explained" moves)
      ground_truth: list of dicts describing each injected anomaly
                     (ticker(s), timestamp, kind, description)
    """
    tickers = tickers or CONFIG.watchlist
    n_bars = n_bars or CONFIG.simulate_n_bars
    freq_minutes = freq_minutes or CONFIG.simulate_bar_freq_minutes
    n_anomalies = n_anomalies if n_anomalies is not None else CONFIG.simulate_anomaly_count

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("min"),
        periods=n_bars,
        freq=f"{freq_minutes}min",
    )

    returns = _generate_correlated_returns(n_bars, tickers, seed)
    volume = _base_volume(n_bars, tickers, seed)

    ground_truth = []
    news_rows = []

    # Sprinkle a handful of "explained" ordinary jumps first, each backed by
    # a matching headline, so the detector has to actually distinguish
    # "big move WITH news" from "big move WITHOUT news" rather than just
    # flagging every large return.
    explained_bars = rng.choice(range(20, n_bars - 20), size=n_anomalies, replace=False)
    for bar in explained_bars:
        tkr = rng.choice(tickers)
        jump = rng.choice([-1, 1]) * rng.uniform(0.02, 0.035)
        returns.loc[bar, tkr] += jump
        news_rows.append({
            "ticker": tkr,
            "timestamp": timestamps[bar],
            "headline": f"{tkr} moves on scheduled earnings update",
        })

    # Now inject the real anomalies judges/tests will look for.
    anomaly_bars = rng.choice(
        [b for b in range(20, n_bars - 20) if b not in explained_bars],
        size=n_anomalies,
        replace=False,
    )
    kinds = ["volume_burst", "unexplained_price_jump", "correlated_group_move"]

    for i, bar in enumerate(anomaly_bars):
        kind = kinds[i % len(kinds)]

        if kind == "volume_burst":
            tkr = rng.choice(tickers)
            multiplier = rng.uniform(6, 12)
            volume.loc[bar:bar + 2, tkr] = (volume.loc[bar:bar + 2, tkr] * multiplier).astype(int)
            ground_truth.append({
                "kind": kind, "tickers": [tkr], "timestamp": timestamps[bar],
                "description": f"{tkr} volume spikes {multiplier:.1f}x normal with no matching headline",
            })

        elif kind == "unexplained_price_jump":
            tkr = rng.choice(tickers)
            jump = rng.choice([-1, 1]) * rng.uniform(0.03, 0.06)
            returns.loc[bar, tkr] += jump
            # deliberately NO news row added
            ground_truth.append({
                "kind": kind, "tickers": [tkr], "timestamp": timestamps[bar],
                "description": f"{tkr} jumps {jump * 100:.1f}% with no news in the surrounding window",
            })

        else:  # correlated_group_move
            group = list(rng.choice(tickers, size=min(3, len(tickers)), replace=False))
            span = CONFIG.simulate_group_shock_bars
            # Sustain the shock over several consecutive bars (a real
            # coordinated move plays out over minutes, not one tick) so a
            # rolling correlation window actually picks up the co-movement
            # instead of it being washed out by a single noisy bar.
            shared_direction = rng.choice([-1, 1])
            for offset in range(span):
                shock = shared_direction * rng.uniform(0.016, 0.028)
                for tkr in group:
                    returns.loc[bar + offset, tkr] += shock + rng.normal(0, 0.0015)
            ground_truth.append({
                "kind": kind, "tickers": group, "timestamp": timestamps[bar],
                "description": f"{', '.join(group)} move together sharply for several bars, outside their normal correlation",
            })

    # Build price levels from returns (start every ticker at a plausible
    # price so charts look sane).
    start_prices = pd.Series(
        np.random.default_rng(seed + 2).uniform(80, 450, size=len(tickers)),
        index=tickers,
    )
    log_prices = np.log(start_prices).values + returns.cumsum().values
    prices = pd.DataFrame(np.exp(log_prices), columns=tickers, index=timestamps)
    returns.index = timestamps
    volume.index = timestamps

    news_df = pd.DataFrame(news_rows)

    return {
        "prices": prices,
        "volume": volume,
        "returns": returns,
        "news": news_df,
        "ground_truth": ground_truth,
    }
