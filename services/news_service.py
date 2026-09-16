"""
Three-state news-relevance classification (Sprint 3).

alert_engine.py already does the hard part (news_relevance.py's TF-IDF +
keyword scoring, folded into detection/score.py's composite). This module
is a thin, UI-facing wrapper that turns a news-lookup result (see
alert_engine.NewsLookup) plus the alert's own timestamp into exactly one
of three states, each with a fixed, quotable label and a human-readable
timestamp-gap sentence -- so the Alert Feed and Stock Terminal never have
to re-derive "what does found=True, relevance=0.2 actually mean" by hand.

    NEWS EXPLAINS MOVEMENT         -- a headline was found nearby AND it
                                       clears the relevance bar.
    POSSIBLE NEWS-MOVEMENT MISMATCH -- a headline was found nearby but does
                                       NOT clear the relevance bar (routine
                                       or unrelated wire copy, not a
                                       plausible explanation).
    NO RELEVANT NEWS DETECTED      -- no headline at all within the
                                       lookback window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import CONFIG

NEWS_EXPLAINS = "NEWS EXPLAINS MOVEMENT"
NEWS_MISMATCH = "POSSIBLE NEWS-MOVEMENT MISMATCH"
NEWS_NONE = "NO RELEVANT NEWS DETECTED"

_ICON = {NEWS_EXPLAINS: "\U0001F4F0", NEWS_MISMATCH: "❓", NEWS_NONE: "\U0001F507"}


@dataclass
class NewsClassification:
    state: str
    icon: str
    headline: Optional[str]
    relevance: float
    gap_text: Optional[str]  # e.g. "headline was 12 min before the move"
    detail: str              # one ready-to-display sentence

    @property
    def badge(self) -> str:
        return f"{self.icon} {self.state}"


def _format_gap(alert_ts: pd.Timestamp, headline_ts: Optional[pd.Timestamp]) -> Optional[str]:
    if headline_ts is None:
        return None
    try:
        alert_ts = pd.Timestamp(alert_ts)
        headline_ts = pd.Timestamp(headline_ts)
        if alert_ts.tzinfo is not None and headline_ts.tzinfo is None:
            headline_ts = headline_ts.tz_localize(alert_ts.tzinfo)
        elif headline_ts.tzinfo is not None and alert_ts.tzinfo is None:
            alert_ts = alert_ts.tz_localize(headline_ts.tzinfo)
        delta = alert_ts - headline_ts
    except (TypeError, ValueError):
        return None

    minutes = delta.total_seconds() / 60.0
    direction = "before" if minutes >= 0 else "after"
    minutes = abs(minutes)
    if minutes < 60:
        span = f"{minutes:.0f} min"
    else:
        span = f"{minutes / 60:.1f} hr"
    return f"headline was {span} {direction} the move"


def classify(alert_ts: pd.Timestamp, news_result: Optional[dict]) -> NewsClassification:
    """`news_result` is exactly what an alert_engine.NewsLookup callable
    returns -- {"found", "headline", "relevance", "timestamp"} -- or None
    when no news check was even attempted for this alert (a volume-driven
    or volatility-driven alert kind, say). A missing check is reported as
    NO RELEVANT NEWS DETECTED with a distinct detail sentence, never
    conflated with "a check ran and found nothing".
    """
    if not news_result or not news_result.get("found"):
        checked = news_result is not None
        detail = ("No headline fell within the news-lookback window for this move."
                   if checked else "No news check was run for this alert kind.")
        return NewsClassification(NEWS_NONE, _ICON[NEWS_NONE], None, 0.0, None, detail)

    headline = news_result.get("headline")
    relevance = float(news_result.get("relevance", 0.0))
    gap_text = _format_gap(alert_ts, news_result.get("timestamp"))

    if relevance >= CONFIG.news_relevance_explained_threshold:
        detail = f"\"{headline}\" scored {relevance:.0%} relevant -- plausibly explains this move."
        if gap_text:
            detail += f" ({gap_text}.)"
        return NewsClassification(NEWS_EXPLAINS, _ICON[NEWS_EXPLAINS], headline, relevance, gap_text, detail)

    detail = f"A nearby headline (\"{headline}\") scored only {relevance:.0%} relevant -- not treated as an explanation."
    if gap_text:
        detail += f" ({gap_text}.)"
    return NewsClassification(NEWS_MISMATCH, _ICON[NEWS_MISMATCH], headline, relevance, gap_text, detail)
