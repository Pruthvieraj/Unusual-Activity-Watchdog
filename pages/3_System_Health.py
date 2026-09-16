"""
System Health -- Sprint 3 page.

Runs the same pipeline the Command Center uses and reports whether each
piece of the detection stack is actually working right now (data source,
autorefresh, ensemble, correlation engine, news scorer, persistence, alert
volume), independent of whether anything happens to be flagged this run.
See services/health_service.py for what each check actually verifies.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import CONFIG
from services import health_service
from services.pipeline_service import resolve_and_score

st.set_page_config(page_title="System Health -- Watchdog", layout="wide", page_icon="\U0001FA7A")

st.title("\U0001FA7A System Health")
st.caption(
    "Is the detection stack actually working right now? An empty Alert Feed alone can't answer that -- it "
    "could mean a quiet market, or a silently broken detector. This page runs a fixed battery of checks "
    "against a live pipeline run instead of inferring health from alert counts."
)

st.sidebar.title("\U0001F6A8 Watchdog Controls")
mode_choice = st.sidebar.radio(
    "Data source", ["Simulated demo (recommended)", "Live / historical (yfinance)"],
    help="Matches the Command Center's own data-source control -- pick the same mode to check the stack "
         "you're actually about to demo.",
)
watchlist = st.sidebar.multiselect(
    "Watchlist", options=sorted(set(CONFIG.watchlist) | {"TSLA", "META", "NFLX", "BAC", "WMT"}),
    default=CONFIG.watchlist,
)
if not watchlist:
    watchlist = CONFIG.watchlist

if mode_choice.startswith("Simulated"):
    seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=9999, value=12, step=1)
    live_autorefresh, refresh_interval = False, CONFIG.live_autorefresh_seconds
else:
    seed = None
    live_autorefresh = st.sidebar.checkbox("Auto-refresh is on in Command Center", value=True,
                                            help="Informational here -- toggle it on the Command Center page.")
    refresh_interval = st.sidebar.select_slider(
        "Refresh interval", options=list(CONFIG.autorefresh_interval_choices),
        value=CONFIG.live_autorefresh_seconds if CONFIG.live_autorefresh_seconds in CONFIG.autorefresh_interval_choices
        else CONFIG.autorefresh_interval_choices[1],
    )

if st.sidebar.button("\U0001F504 Re-run checks now"):
    st.cache_data.clear()

with st.spinner("Running pipeline + health checks..."):
    result = resolve_and_score(mode_choice, watchlist, seed)
    last_live_success_ts = st.session_state.get("_last_live_success_ts")
    checks = health_service.run_all_checks(result, mode_choice, live_autorefresh, refresh_interval, last_live_success_ts)

status_rank = {health_service.FAIL: 0, health_service.WARN: 1, health_service.OK: 2}
overall = min((c.status for c in checks), key=lambda s: status_rank[s])
overall_banner = {
    health_service.OK: ("✅", "All checks passing."),
    health_service.WARN: ("⚠️", "One or more checks need attention."),
    health_service.FAIL: ("\U0001F534", "One or more checks are failing."),
}[overall]
st.markdown(f"## {overall_banner[0]} {overall_banner[1]}")

m1, m2, m3 = st.columns(3)
m1.metric("OK", sum(1 for c in checks if c.status == health_service.OK))
m2.metric("Warnings", sum(1 for c in checks if c.status == health_service.WARN))
m3.metric("Failures", sum(1 for c in checks if c.status == health_service.FAIL))

st.markdown("---")
for check in sorted(checks, key=lambda c: status_rank[c.status]):
    with st.container(border=True):
        st.markdown(f"#### {check.badge} — {check.name}")
        st.write(check.detail)

st.markdown("---")
st.caption(
    "Checked: live-vs-simulated data source (and staleness in Live mode), the st.fragment autorefresh "
    "configuration, the Isolation Forest + autoencoder ensemble's output columns, the correlation-break "
    "engine's output shape, a news-relevance self-test, the SQLite alert store's reachability, and whether "
    "this run's alert volume is within a sane range."
)
