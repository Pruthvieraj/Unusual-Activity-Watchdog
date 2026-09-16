"""
Live/historical data access via yfinance, plus a lightweight news check.

This module talks to the real internet (Yahoo Finance) and will only work
where that's reachable -- e.g. on your own laptop, not necessarily inside a
locked-down sandbox. `app.py` falls back to `data/synthetic.py` automatically
if these calls fail, so the dashboard always has something to show.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd
import yfinance as yf

from config import CONFIG
from news_relevance import score_headline_relevance

# One shared, tiny pool for bounding every outbound yfinance call below --
# see CONFIG.live_fetch_timeout_s for why this exists (yf.download's own
# timeout= only covers part of what it does, and .news exposes no timeout
# at all). A single worker is enough: these calls are already serialized by
# Streamlit's script-run thread, this pool just gives them a hard ceiling.
_NETWORK_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="live-feed-net")


def _bounded(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) but never wait longer than
    CONFIG.live_fetch_timeout_s for it -- raises TimeoutError instead of
    hanging past that, so a network that silently drops packets fails just
    as fast as one that actively rejects the connection. The underlying
    call, if it eventually does complete, keeps running in the pool thread
    (Python has no way to forcibly kill a blocked thread) but its result is
    simply discarded -- callers have already fallen back by then.
    """
    future = _NETWORK_POOL.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=CONFIG.live_fetch_timeout_s)
    except FutureTimeoutError as exc:
        raise TimeoutError(
            f"{getattr(fn, '__name__', fn)} did not respond within {CONFIG.live_fetch_timeout_s}s"
        ) from exc


def fetch_intraday(tickers: list[str] | None = None) -> dict:
    """Pull recent intraday bars for the watchlist and reshape into the same
    {prices, volume, returns} shape the synthetic generator produces, so the
    rest of the pipeline doesn't care which source it came from.
    """
    tickers = tickers or CONFIG.watchlist

    raw = _bounded(
        yf.download,
        tickers=tickers,
        period=CONFIG.intraday_period,
        interval=CONFIG.intraday_interval,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=CONFIG.live_fetch_timeout_s,
    )

    prices, volume = {}, {}
    for tkr in tickers:
        try:
            sub = raw[tkr] if len(tickers) > 1 else raw
            prices[tkr] = sub["Close"]
            volume[tkr] = sub["Volume"]
        except (KeyError, TypeError):
            continue

    prices_df = pd.DataFrame(prices).dropna(how="all")
    volume_df = pd.DataFrame(volume).reindex(prices_df.index)
    returns_df = np.log(prices_df / prices_df.shift(1))

    return {"prices": prices_df, "volume": volume_df, "returns": returns_df}


def fetch_recent_news(ticker: str, limit: int = 10) -> list[dict]:
    """Recent headlines for a ticker via yfinance's built-in news feed (no
    separate news API / key required). Returns a list of
    {title, publisher, published_at} dicts, newest first.
    """
    try:
        items = _bounded(lambda: yf.Ticker(ticker).news) or []
    except Exception:
        # Catches TimeoutError (from _bounded) alongside every other
        # yfinance failure mode -- a slow/hanging news lookup degrades to
        # "no headlines for this ticker" exactly like any other failure,
        # rather than blocking the whole pipeline run.
        return []

    out = []
    for item in items[:limit]:
        content = item.get("content", item)  # yfinance schema has varied across versions
        title = content.get("title") or item.get("title")
        publisher = (content.get("provider") or {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher")
        published = content.get("pubDate") or item.get("providerPublishTime")
        out.append({"title": title, "publisher": publisher, "published_at": published})
    return out


def has_recent_news(ticker: str, within_hours: int | None = None) -> bool:
    """Legacy presence-only check: is there ANY headline for this ticker
    within the last `within_hours`, regardless of what it's about? Kept for
    backward compatibility, but `build_live_news_lookup` below is what the
    pipeline actually uses now -- presence alone doesn't tell you whether a
    headline explains a specific move, only that one exists somewhere
    nearby. Fails open (treats as "no news found") on error so a flaky news
    call never silently suppresses a real anomaly alert.
    """
    within_hours = within_hours or CONFIG.news_lookback_hours
    news = fetch_recent_news(ticker)
    if not news:
        return False

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=within_hours)
    for item in news:
        ts = item.get("published_at")
        if ts is None:
            continue
        try:
            published = pd.to_datetime(ts, unit="s", utc=True) if isinstance(ts, (int, float)) else pd.to_datetime(ts, utc=True)
        except Exception:
            continue
        if published >= cutoff:
            return True
    return False


def _parse_published(raw_ts) -> pd.Timestamp | None:
    if raw_ts is None:
        return None
    try:
        if isinstance(raw_ts, (int, float)):
            return pd.to_datetime(raw_ts, unit="s", utc=True)
        return pd.to_datetime(raw_ts, utc=True)
    except Exception:
        return None


def build_live_news_lookup(within_hours: int | None = None):
    """Live-mode equivalent of alert_engine.build_synthetic_news_lookup:
    returns a `lookup(ticker, ts) -> {"found", "headline", "relevance"}`
    callable, scored with the same news_relevance.py TF-IDF/keyword model
    the synthetic demo path uses, instead of the old bare
    presence/absence check. Each ticker's headlines are fetched at most
    once per lookup object (small, bounded network cost -- one call per
    watchlist ticker, not per alert).
    """
    within_hours = within_hours or CONFIG.news_lookback_hours
    window = pd.Timedelta(hours=within_hours)
    cache: dict[str, list[dict]] = {}

    def lookup(ticker: str, ts: pd.Timestamp) -> dict:
        if ticker not in cache:
            try:
                cache[ticker] = fetch_recent_news(ticker)
            except Exception:
                cache[ticker] = []

        ts_utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        # No "timestamp" key here when nothing is found -- matches
        # alert_engine._NO_NEWS's contract exactly (see that module for why:
        # the pre-existing regression suite pins the not-found shape).
        # Downstream (services/news_service.classify) only ever reads
        # timestamp via .get(), so its absence here is never a KeyError.
        best = {"found": False, "headline": None, "relevance": 0.0}
        for item in cache[ticker]:
            published = _parse_published(item.get("published_at"))
            if published is None or abs(published - ts_utc) > window:
                continue
            relevance = score_headline_relevance(item.get("title"), ticker)
            if relevance > best["relevance"] or not best["found"]:
                best = {"found": True, "headline": item.get("title"), "relevance": relevance, "timestamp": published}
        return best
        return best

    return lookup
