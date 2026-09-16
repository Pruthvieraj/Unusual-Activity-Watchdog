"""
Shared display-formatting helpers (Sprint 4) -- empty/error/loading state
copy and the small recurring formatting snippets (window strings, badges,
percentages) that were previously written inline, slightly differently, in
each place app.py needed them. New pages should use these instead of
re-deriving their own wording, so the same situation always reads the same
way everywhere in the app.
"""

from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------------
# Empty / error / loading state copy -- centralized so the same situation
# (no data yet, a filter that matched nothing, a failed fetch) always reads
# the same way on every page instead of each one improvising its own
# wording.
# --------------------------------------------------------------------------
EMPTY_NO_ALERTS = "No events matching the current filters."
EMPTY_NO_HISTORY = "No historical alerts stored yet -- they'll appear here once the Command Center pipeline has run at least once."
EMPTY_NO_GROUND_TRUTH = "Detection accuracy is only computable against the simulated scenario's known ground truth. Switch to 'Simulated demo' mode to see it."
LOADING_PIPELINE = "Running pipeline + health checks..."
ERROR_LIVE_FALLBACK_PREFIX = "⚠️ Live data unavailable this run -- showing simulated data instead. Reason: "
STALE_DATA_TEMPLATE = "\U0001F7E0 Market data delayed — last update {seconds:.0f}s ago."


def format_window(first_seen: pd.Timestamp, last_seen: pd.Timestamp) -> str:
    """"Jan 05, 10:30" for a single-bar event, "Jan 05, 10:30 → 10:45" for
    one spanning several -- the same rule app.py's Alert Feed table uses,
    now shared with the History page so a merged event reads identically
    in both places."""
    first_seen, last_seen = pd.Timestamp(first_seen), pd.Timestamp(last_seen)
    if first_seen == last_seen:
        return first_seen.strftime("%b %d, %H:%M")
    return f"{first_seen.strftime('%b %d, %H:%M')} → {last_seen.strftime('%H:%M')}"


def format_pct(value: float, decimals: int = 0) -> str:
    return f"{value:.{decimals}f}%"


def format_confidence(value: float) -> str:
    return f"{value:.0f}/100"


def format_gap_seconds(seconds: float) -> str:
    """"12 min" / "1.5 hr" -- the same span-formatting rule
    services/news_service.py uses for headline-to-move gaps, shared here
    so any other page reporting a time gap (e.g. alert age on the History
    page) reads consistently."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} hr"


def kind_label(kind: str, kind_labels: dict) -> str:
    return kind_labels.get(kind, kind)
