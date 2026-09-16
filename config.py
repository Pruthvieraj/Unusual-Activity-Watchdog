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
    # Hard wall-clock bound on any single outbound yfinance call (quotes or
    # news). Found necessary by Section 32's manual "fragment refresh, 3
    # cycles" test: yf.download's own timeout=10 default only bounds ONE of
    # the underlying HTTP calls it makes, and yf.Ticker(...).news exposes no
    # timeout kwarg at all -- on a network that silently drops packets
    # instead of actively rejecting them (rather than the fast, explicit
    # 403 this sandbox's own egress proxy returns), either call can hang for
    # a full OS-level TCP timeout (60s+), during which the st.fragment(
    # run_every=...) header/feed appear frozen even though nothing crashed.
    # data/live_feed.py wraps every yfinance call in this bound so a bad
    # network fails FAST and falls back to simulated data, keeping
    # "continuous monitoring" true under real-world flaky connectivity, not
    # just under a clean connection or an immediate, explicit rejection.
    live_fetch_timeout_s: float = 8.0

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
    # A break is only classified as a genuine small-group "correlated_group_move"
    # when the top-k cluster signal is at least this many times larger than the
    # FULL matrix's average |delta| at that same moment. Below this ratio, the
    # whole book moved together (a market-wide event), not a suspicious subset,
    # so it's labeled "market_wide_move" instead -- see correlation_watch.py.
    correlation_concentration_ratio_threshold: float = 1.8
    news_lookback_hours: int = 6            # "no news" window around a price jump
    # A headline within the lookback window only counts as "explaining" a
    # price jump if its relevance score (news_relevance.py: TF-IDF similarity
    # to market-moving event archetypes + keyword/entity match) clears this
    # bar -- presence alone ("a headline exists") is not enough.
    news_relevance_explained_threshold: float = 0.45

    # --- Neural autoencoder (second, independent anomaly model) -----------
    autoencoder_window: int = 3            # bars of feature history flattened into one input
    autoencoder_hidden_layers: tuple = (12, 4, 12)   # bottleneck at 4 -- forces compression
    autoencoder_zscore_alert: float = 2.25  # reconstruction-error z-score to flag

    # --- Order-flow surveillance (spoofing / layering / stuffing / wash) ---
    orderbook_minutes: int = 120
    orderbook_events_per_minute: int = 8
    orderbook_n_traders: int = 40

    # --- Simulated demo data (used when live data is unavailable) ----------
    simulate_n_bars: int = 400
    simulate_bar_freq_minutes: int = 5
    simulate_anomaly_count: int = 4
    simulate_group_shock_bars: int = 10   # consecutive bars a correlated-move shock spans

    # --- "Continuous" monitoring (Live mode autorefresh, st.fragment) -------
    # How often Live mode silently re-pulls yfinance data and re-scores,
    # without any user click -- this is what makes "continuously watches" a
    # true statement instead of aspirational language, and gives a fixed,
    # quotable worst-case detection latency (bar interval + this number).
    live_autorefresh_seconds: int = 20
    # Options offered on the sidebar's refresh-interval control.
    autorefresh_interval_choices: tuple = (10, 20, 60)
    # If the last successful refresh is older than this multiple of the
    # configured interval, the UI must show DELAYED rather than silently
    # keep displaying stale data as if it were current.
    stale_data_multiplier: float = 2.0

    # --- Market-regime awareness (detection/regime.py) ----------------------
    # A bar counts toward "breadth" when a ticker's |return_zscore| clears
    # this bar -- same shape as price_jump_sigma_alert but a separate,
    # lower constant since breadth cares about "unusual for this stock",
    # not "alert-worthy on its own".
    regime_breadth_zscore_threshold: float = 2.0
    # If this fraction (or more) of the watchlist clears that threshold on
    # the SAME bar, it's tagged a market-wide event -- individual
    # unexplained-price-jump alerts on that bar are suppressed in favor of
    # one consolidated market_wide_move alert (see alert_engine.py).
    regime_breadth_cutoff: float = 0.6

    # --- Momentum-shift detection (detection/momentum.py) -------------------
    momentum_short_span: int = 5     # bars, short EWMA
    momentum_long_span: int = 20     # bars, long EWMA
    momentum_history_window: int = 30  # bars used to judge "is this crossover typical"
    momentum_shift_zscore_alert: float = 2.25

    # --- Persistence (data/store.py) ----------------------------------------
    # SQLite file for alert history/investigation-workflow persistence.
    # Lives on local disk -- see README's persistence caveat for what that
    # means on Render's free tier (survives the running instance, wiped on
    # redeploy/cold restart).
    sqlite_path: str = "watchdog_alerts.db"


CONFIG = WatchdogConfig()
