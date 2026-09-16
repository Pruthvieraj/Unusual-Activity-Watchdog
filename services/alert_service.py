"""
Investigation-workflow service: the boundary between app.py / pages/ and
data/store.py's SQLite persistence. app.py and the multipage app never
import sqlite3 or hold a raw connection directly -- only this module's
functions, which own the one shared, Streamlit-cached connection.

`st.cache_resource` (not `st.cache_data`) is deliberate: a DB connection is
a stateful resource to be reused across reruns, not data to be hashed and
copied -- using cache_data here would either fail to hash a sqlite3
Connection or silently create a new one every rerun, defeating the point.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from data import store

ALLOWED_STATUSES = store.ALLOWED_STATUSES
STATUS_NEW = store.STATUS_NEW
STATUS_INVESTIGATING = store.STATUS_INVESTIGATING
STATUS_RESOLVED = store.STATUS_RESOLVED
STATUS_FALSE_POSITIVE = store.STATUS_FALSE_POSITIVE


@st.cache_resource(show_spinner=False)
def _shared_connection(db_path: Optional[str] = None):
    conn = store.get_connection(db_path)
    store.init_db(conn)
    return conn


def get_connection():
    return _shared_connection()


def sync_pipeline_alerts(alerts_df: pd.DataFrame) -> int:
    """Persist whatever the current pipeline run's generate_all_alerts()
    produced. Safe to call every rerun (including every st.fragment tick):
    it's an upsert keyed on (kind, tickers, first_seen), so an alert that
    already exists just gets its freshness columns refreshed, never a
    second row, and an analyst's status/notes on it are untouched. Returns
    how many were genuinely new, for a "N new since last check" badge.
    """
    return store.upsert_alerts(get_connection(), alerts_df)


def fetch_alerts(
    status: Optional[str] = None,
    kind: Optional[str] = None,
    severity: Optional[str] = None,
    ticker: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    return store.list_alerts(get_connection(), status=status, kind=kind, severity=severity, ticker=ticker, limit=limit)


def get_alert(alert_id: int) -> Optional[dict]:
    return store.get_alert(get_connection(), alert_id)


def alert_key(kind: str, tickers: str, first_seen) -> str:
    """Same identity store.py computes internally -- exposed so the UI can
    join a live in-memory alerts DataFrame against persisted status/notes
    with one bulk query instead of one round trip per row."""
    return store.alert_key(kind, tickers, first_seen)


def lookup_id(kind: str, tickers: str, first_seen) -> Optional[int]:
    """The in-memory alert DataFrame's row index isn't stable across
    reruns; this resolves a displayed (kind, tickers, first_seen) alert to
    its persisted id so the UI knows which stored row to update."""
    return store.get_id_by_key(get_connection(), kind, tickers, first_seen)


def transition_status(alert_id: int, new_status: str, note: Optional[str] = None) -> None:
    store.set_status(get_connection(), alert_id, new_status, note=note)


def add_note(alert_id: int, note: str) -> None:
    store.add_note(get_connection(), alert_id, note)


def status_counts() -> dict:
    return store.status_counts(get_connection())


def reset_all() -> None:
    """Wipes every persisted alert -- the demo-mode "reset" control's
    backing call (Section 27)."""
    store.clear_all(get_connection())
