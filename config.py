"""
Central configuration for the Unusual Activity Watchdog.

Everything a judge/teammate would want to tweak lives here so the rest of
the codebase never hardcodes a ticker, window size, or threshold.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class WatchdogConfig:
    # --- Universe ---------------------------------------------------------
    # Default watchlist: a mix of large-caps that tend to move together
    # (tech) plus a couple of unrelated names, so the correlation engine has
    # both a "genuine cluster" and "should stay independent" case to show.
    watchlist: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "JPM", "XOM"
    ])

    # --- Data windows -------------------------------------------------------
    lookback_days: int = 60          # history used to build the "normal" baseline
    intraday_interval: str = "5m"    # granularity for live/near-real-time mode
    intraday_period: str = "5d"      # how far back intraday data is pulled

    # --- Anomaly model (Isolation Forest) -----------------------------------
    isolation_forest_contamination: float = 0.05   # expected fraction of anomalies
    isolation_forest_estimators: int = 200
    random_state: int = 42

    # --- Rolling feature windows --------------------------------------------
    volume_zscore_window: int = 20     # bars, for volume-burst detection
    return_vol_window: int = 20        # bars, for volatility-adjusted price jumps
    correlation_window: int = 15       # bars, for rolling pairwise correlation
    correlation_baseline_window: int = 90  # bars, "normal" correlation baseline

    # --- Thresholds (tuned to be demo-friendly, not production-grade) ------
    volume_zscore_alert: float = 2.5        # |z| above this = volume burst
    price_jump_sigma_alert: float = 3.0     # return more than N * rolling std
    # Pairwise correlation is noisy over short windows, so the correlation
    # break isn't flagged on a fixed absolute jump -- it's flagged when the
    # jump is itself unusual relative to its own recent history (a z-score
    # of the delta series). This self-calibrates instead of needing a magic
    # constant re-tuned per watchlist.
    correlation_break_zscore: float = 2.25
    correlation_delta_history_window: int = 60  # bars used to judge "is this delta typical"
    news_lookback_hours: int = 6            # "no news" window around a price jump

    # --- Simulated demo data (used when live data is unavailable) ----------
    simulate_n_bars: int = 400
    simulate_bar_freq_minutes: int = 5
    simulate_anomaly_count: int = 4
    simulate_group_shock_bars: int = 10   # consecutive bars a correlated-move shock spans


CONFIG = WatchdogConfig()
