"""
Live/historical data access via yfinance, plus a lightweight news check.

This module talks to the real internet (Yahoo Finance) and will only work
where that's reachable -- e.g. on your own laptop, not necessarily inside a
locked-down sandbox. `app.py` falls back to `data/synthetic.py` automatically
if these calls fail, so the dashboard always has something to show.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from config import CONFIG


def fetch_intraday(tickers: list[str] | None = None) -> dict:
    """Pull recent intraday bars for the watchlist and reshape into the same
    {prices, volume, returns} shape the synthetic generator produces, so the
    rest of the pipeline doesn't care which source it came from.
    """
    tickers = tickers or CONFIG.watchlist

    raw = yf.download(
        tickers=tickers,
        period=CONFIG.intraday_period,
        interval=CONFIG.intraday_interval,
        group_by="ticker",
        progress=False,
        threads=True,
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
        items = yf.Ticker(ticker).news or []
    except Exception:
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
    """Best-effort check: is there a headline for this ticker within the
    last `within_hours`? Used to separate 'explained' price jumps from
    'unexplained' ones. Fails open (treats as "no news found") on error so a
    flaky news call never silently suppresses a real anomaly alert.
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
