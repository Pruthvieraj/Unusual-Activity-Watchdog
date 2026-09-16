"""
Shared data-pipeline orchestration (Sprint 4's services/pipeline_service.py,
pulled forward into Sprint 3 because the new multipage app needs it too).

Streamlit's multipage convention (a `pages/` directory) runs each page as
its OWN top-level script -- a page can't just `from app import ...`
without re-executing app.py's entire top-level body (sidebar widgets,
st.set_page_config, and everything else). So the actual pipeline logic
(resolve market data -> features -> ensemble -> correlation -> alerts ->
stress) lives HERE, as plain cached functions with no page-specific
rendering in them, and both app.py (the Command Center) and every page
under pages/ import from this one place instead of each re-implementing
(or re-importing) the pipeline.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from alert_engine import build_synthetic_news_lookup, generate_all_alerts
from config import CONFIG
from correlation_watch import detect_correlation_breaks
from data.synthetic import generate_market_tape
from data_source import resolve_market_data
from features import build_all_features
from models.ensemble import build_ensemble
from order_flow import run_surveillance
from stress_index import compute_stress_series


@st.cache_data(show_spinner=False)
def load_simulated(seed: int, tickers: tuple[str, ...]):
    return generate_market_tape(tickers=list(tickers), seed=seed)


@st.cache_data(show_spinner=False, ttl=CONFIG.live_autorefresh_seconds)
def load_live(tickers: tuple[str, ...]):
    # `ttl` matches the autorefresh interval: once the cache entry expires,
    # the next refresh genuinely re-pulls fresh bars instead of silently
    # re-serving the same cached response -- this is what makes a periodic
    # rerun an actual data refresh, not just a cosmetic re-render on a timer.
    from data.live_feed import fetch_intraday
    return fetch_intraday(tickers=list(tickers))


def _fetch_live_with_retry(tickers, max_retries: int = 2, base_delay_s: float = 0.6):
    """Retry + exponential backoff in front of load_live(), so a single
    transient yfinance hiccup doesn't immediately drop Live mode into
    Simulated -- only a fetch that fails on every attempt does. Kept
    separate from data_source.py so resolve_market_data's own tested
    fallback contract -- and its unit test, which expects an immediate
    fallback on a failing fetch_live -- is untouched.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return load_live(tuple(tickers))
        except Exception as exc:  # noqa: BLE001 -- deliberately broad; resolve_market_data classifies it
            last_exc = exc
            if attempt < max_retries:
                time.sleep(base_delay_s * (2 ** attempt))
    raise last_exc


@st.cache_data(show_spinner=False)
def run_pipeline(market: dict, news_df_key: str, bar_minutes: float, live_news: bool = False):
    features = build_all_features(market)
    scored = build_ensemble(features)
    breaks = detect_correlation_breaks(market["returns"])
    news_df = market.get("news")
    if news_df is not None:
        news_lookup = build_synthetic_news_lookup(news_df, CONFIG.news_lookback_hours)
    elif live_news:
        # Live mode has no synthetic news table -- pull real yfinance
        # headlines and run them through the same relevance-scoring model
        # (news_relevance.py) the demo path uses, instead of leaving live
        # mode with no news check at all.
        from data.live_feed import build_live_news_lookup
        news_lookup = build_live_news_lookup(CONFIG.news_lookback_hours)
    else:
        news_lookup = None
    alerts = generate_all_alerts(scored, breaks, news_lookup, include_low_severity=True, merge_bar_minutes=bar_minutes)
    stress = compute_stress_series(scored, breaks)
    return scored, breaks, alerts, stress


def _apply_demo_shock(market: dict, watchlist_: list[str]) -> dict:
    """Demo-mode banner's "trigger synthetic anomaly now" button (Section
    27): rather than making a presenter wait for a new random scenario or
    re-seed, this stamps a large, unmistakable move onto the LATEST bar of
    one ticker in the already-loaded simulated data, so the very next
    pipeline run flags something live, on demand. Simulated-mode only --
    never applied to real Live data. Reads the request from
    st.session_state (set by app.py's button) so every call site
    (resolve_and_score's three callers -- header fragment, alert-feed
    fragment, and the top-level tabs) picks it up consistently without each
    needing to pass it through explicitly.
    """
    shock = st.session_state.get("_demo_shock")
    if not shock or "returns" not in market:
        return market
    ticker = shock.get("ticker")
    returns = market["returns"]
    if ticker not in watchlist_ or ticker not in returns.columns or returns.empty:
        return market

    # Shallow-copy the dict and deep-copy only the frames being edited --
    # load_simulated's cached result must never be mutated in place, or
    # every subsequent cache hit would silently carry the shock forward.
    market = dict(market)
    market["returns"] = returns.copy()
    if market["returns"][ticker].dtype.kind != "f":
        market["returns"][ticker] = market["returns"][ticker].astype(float)
    last_ts = market["returns"].index[-1]
    magnitude = float(shock.get("magnitude", 0.06))
    market["returns"].loc[last_ts, ticker] = magnitude

    if "prices" in market and market["prices"] is not None and ticker in market["prices"].columns:
        prices = market["prices"].copy()
        if prices[ticker].dtype.kind != "f":
            prices[ticker] = prices[ticker].astype(float)
        prev_price = prices[ticker].iloc[-2] if len(prices) > 1 else prices[ticker].iloc[-1]
        prices.loc[last_ts, ticker] = float(prev_price) * (1 + magnitude)
        market["prices"] = prices

    if "volume" in market and market["volume"] is not None and ticker in market["volume"].columns:
        volume = market["volume"].copy()
        baseline = volume[ticker].iloc[:-1].mean() if len(volume) > 1 else volume[ticker].iloc[-1]
        # Match the column's existing dtype (usually int64 share counts) --
        # assigning a bare float into an int64 column raises under recent
        # pandas instead of silently upcasting.
        spiked = baseline * 4.0
        volume.loc[last_ts, ticker] = volume[ticker].dtype.type(spiked)
        market["volume"] = volume

    return market


def resolve_and_score(mode_: str, watchlist_: list[str], seed_):
    """The full data-pipeline call (resolve market data -> build features ->
    ensemble score -> correlation breaks -> alerts -> stress). Pure w.r.t.
    Streamlit rendering (no UI calls) so every page can call it and decide
    for itself how to render the result; cheap to call repeatedly thanks to
    st.cache_data memoizing load_live/load_simulated/run_pipeline.
    """
    load_result = resolve_market_data(
        mode_, watchlist_, seed_,
        fetch_live=_fetch_live_with_retry,
        fetch_simulated=lambda tks, sd: load_simulated(sd, tuple(tks)),
    )
    market_ = load_result["market"]
    bar_minutes_ = load_result["bar_minutes"]
    resolved_mode = load_result["mode"]
    resolved_seed = load_result["seed"]
    note = load_result["note"]
    error = load_result["error"]

    if resolved_mode.startswith("Simulated"):
        market_ = _apply_demo_shock(market_, watchlist_)

    news_key = "synthetic" if "news" in market_ else "none"
    scored_, breaks_, alerts_, stress_series_ = run_pipeline(
        market_, news_key, bar_minutes_, live_news=resolved_mode.startswith("Live"),
    )

    if resolved_mode.startswith("Live") and error is None:
        st.session_state["_last_live_success_ts"] = pd.Timestamp.now()

    return {
        "market": market_, "bar_minutes": bar_minutes_, "mode": resolved_mode, "seed": resolved_seed,
        "note": note, "error": error, "scored": scored_, "breaks": breaks_, "alerts": alerts_,
        "stress_series": stress_series_,
    }


@st.cache_data(show_spinner=False)
def run_order_flow(ticker: str, seed: int):
    from data.orderbook_synthetic import generate_order_events
    result = generate_order_events(
        ticker, n_minutes=CONFIG.orderbook_minutes, events_per_minute=CONFIG.orderbook_events_per_minute,
        n_traders=CONFIG.orderbook_n_traders, seed=seed,
    )
    flags = run_surveillance(result["events"])
    return result, flags
