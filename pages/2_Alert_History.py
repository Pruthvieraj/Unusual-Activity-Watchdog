"""
Alert History & Analytics -- Sprint 3 page.

Read-only-ish analytics over data/store.py's persisted alert log: trends
over time, breakdowns by kind/severity/ticker, and investigation-outcome
stats (how many ended RESOLVED vs FALSE POSITIVE vs still open). Also lets
an analyst update an alert's investigation status from here, the same
workflow the Command Center's Alert Feed offers, for whenever they're
reviewing history rather than watching the live feed.

Reads whatever services/alert_service.py's SQLite store currently holds --
if the Command Center hasn't been run yet this session, that's empty; see
utils/formatting.EMPTY_NO_HISTORY for the copy shown in that case.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services import alert_service
from utils import design_tokens as tokens
from utils import formatting

st.set_page_config(page_title="Alert History -- Watchdog", layout="wide", page_icon="\U0001F4CA")
st.markdown(tokens.PAGE_CSS, unsafe_allow_html=True)

KIND_LABELS = {
    "volume_burst": "Volume burst",
    "unexplained_price_jump": "Unexplained price jump",
    "explained_price_move": "Explained price move",
    "correlated_group_move": "Correlated group move",
    "market_wide_move": "Market-wide move (not a cluster)",
    "volatility_spike": "Volatility spike",
    "momentum_shift": "Momentum shift (EWMA crossover)",
}

st.title("\U0001F4CA Alert History & Analytics")
st.caption(
    "Every alert the Command Center pipeline has produced this session, with its investigation outcome -- "
    "for spotting patterns a single live-feed snapshot can't show (which kinds recur, which tickers are "
    "noisiest, how much gets resolved vs. marked a false positive)."
)

all_alerts = alert_service.fetch_alerts()

if all_alerts.empty:
    st.info(formatting.EMPTY_NO_HISTORY)
    st.stop()

# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Filters")
    status_filter = st.multiselect("Investigation status", alert_service.ALLOWED_STATUSES, default=[])
    kind_filter = st.multiselect(
        "Alert type", sorted(all_alerts["kind"].unique()),
        format_func=lambda k: KIND_LABELS.get(k, k), default=[],
    )
    severity_filter = st.multiselect("Severity", ["High", "Medium", "Low"], default=[])
    ticker_filter = st.text_input("Ticker contains", value="")
    date_range = st.date_input(
        "First-seen date range",
        value=(all_alerts["first_seen"].min().date(), all_alerts["first_seen"].max().date()),
    )

filtered = all_alerts.copy()
if status_filter:
    filtered = filtered[filtered["status"].isin(status_filter)]
if kind_filter:
    filtered = filtered[filtered["kind"].isin(kind_filter)]
if severity_filter:
    filtered = filtered[filtered["severity"].isin(severity_filter)]
if ticker_filter:
    filtered = filtered[filtered["tickers"].str.contains(ticker_filter, case=False, na=False)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["first_seen"].dt.date >= start) & (filtered["first_seen"].dt.date <= end)
    ]

if filtered.empty:
    st.warning("No stored alerts match the current filters.")
    st.stop()

# --------------------------------------------------------------------------
# KPIs
# --------------------------------------------------------------------------
counts = filtered["status"].value_counts().to_dict()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total (filtered)", len(filtered))
k2.metric(f"{tokens.INVESTIGATION_ICON['NEW']} New", counts.get(alert_service.STATUS_NEW, 0))
k3.metric(f"{tokens.INVESTIGATION_ICON['INVESTIGATING']} Investigating", counts.get(alert_service.STATUS_INVESTIGATING, 0))
k4.metric(f"{tokens.INVESTIGATION_ICON['RESOLVED']} Resolved", counts.get(alert_service.STATUS_RESOLVED, 0))
k5.metric(f"{tokens.INVESTIGATION_ICON['FALSE POSITIVE']} False positive", counts.get(alert_service.STATUS_FALSE_POSITIVE, 0))

st.markdown("---")

# --------------------------------------------------------------------------
# Trends + breakdowns
# --------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Alerts over time")
    by_day = filtered.set_index("first_seen").resample("1D").size().rename("count").reset_index()
    fig = go.Figure(go.Bar(x=by_day["first_seen"], y=by_day["count"], marker_color=tokens.CATEGORICAL[0]))
    fig = tokens.plotly_base_layout(fig, height=tokens.CHART_HEIGHT_MD)
    fig.update_layout(title="Events by day (first-seen)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Breakdown by alert type")
    by_kind = filtered["kind"].map(lambda k: KIND_LABELS.get(k, k)).value_counts()
    fig = go.Figure(go.Bar(x=by_kind.values, y=by_kind.index, orientation="h", marker_color=tokens.CATEGORICAL[2]))
    fig = tokens.plotly_base_layout(fig, height=tokens.CHART_HEIGHT_MD)
    fig.update_layout(title="Event count by type", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.markdown("#### Breakdown by severity")
    by_sev = filtered["severity"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
    fig = go.Figure(go.Bar(x=by_sev.index, y=by_sev.values,
                            marker_color=[tokens.STATUS.get(s, tokens.CATEGORICAL[0]) for s in by_sev.index]))
    fig = tokens.plotly_base_layout(fig, height=tokens.CHART_HEIGHT_SM)
    fig.update_layout(title="Event count by severity")
    st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown("#### Noisiest tickers")
    ticker_counts = (
        filtered["tickers"].str.split(", ").explode().value_counts().head(10)
    )
    fig = go.Figure(go.Bar(x=ticker_counts.values, y=ticker_counts.index, orientation="h",
                            marker_color=tokens.CATEGORICAL[4]))
    fig = tokens.plotly_base_layout(fig, height=tokens.CHART_HEIGHT_SM)
    fig.update_layout(title="Top 10 tickers by alert count", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Investigation outcome stats
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### Investigation outcomes")
closed = filtered[filtered["status"].isin([alert_service.STATUS_RESOLVED, alert_service.STATUS_FALSE_POSITIVE])]
if closed.empty:
    st.caption("No alerts have been marked Resolved or False Positive yet in the current filter.")
else:
    resolved_pct = (closed["status"] == alert_service.STATUS_RESOLVED).mean() * 100
    st.metric("Of closed investigations, resolved as genuine", f"{resolved_pct:.0f}%",
              help="RESOLVED / (RESOLVED + FALSE POSITIVE) among alerts in the current filter that have reached "
                   "a closed state -- NEW and INVESTIGATING items are excluded from this rate.")

# --------------------------------------------------------------------------
# Row-level table + per-row status update
# --------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### Alert log")
display_df = filtered.copy()
display_df["window"] = display_df.apply(lambda r: formatting.format_window(r["first_seen"], r["last_seen"]), axis=1)
display_df["type"] = display_df["kind"].map(lambda k: KIND_LABELS.get(k, k))
display_df["status_badge"] = display_df["status"].map(lambda s: f"{tokens.INVESTIGATION_ICON.get(s, '')} {s}")
display_df["severity_badge"] = display_df["severity"].map(lambda s: f"{tokens.STATUS_ICON.get(s, '')} {s}")
display_df = display_df.sort_values("last_seen", ascending=False)

st.dataframe(
    display_df[["window", "severity_badge", "type", "tickers", "headline", "confidence", "status_badge"]].rename(columns={
        "window": "When", "severity_badge": "Severity", "type": "Type", "tickers": "Ticker(s)",
        "headline": "Alert", "confidence": "Confidence", "status_badge": "Investigation",
    }),
    use_container_width=True, hide_index=True, height=360,
)

st.markdown("#### Update an alert's investigation status")
options = list(display_df["id"])
labels = {row["id"]: f"#{row['id']} — {row['window']} — {row['headline']}" for _, row in display_df.iterrows()}
picked_id = st.selectbox("Choose an alert", options, format_func=lambda i: labels[i])
picked = alert_service.get_alert(int(picked_id))
if picked:
    st.write(picked["detail"])
    new_status = st.selectbox(
        "New status", alert_service.ALLOWED_STATUSES,
        index=alert_service.ALLOWED_STATUSES.index(picked["status"]),
        key=f"history_status_{picked_id}",
    )
    note = st.text_input("Add a note (optional)", key=f"history_note_{picked_id}")
    if st.button("Save", key=f"history_save_{picked_id}"):
        alert_service.transition_status(int(picked_id), new_status, note=note or None)
        st.success("Saved -- refresh the page to see it reflected in the charts above.")
    if picked.get("notes"):
        with st.expander("Notes history"):
            st.text(picked["notes"])
