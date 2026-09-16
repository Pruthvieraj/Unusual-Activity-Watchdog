"""
Unusual Activity Watchdog -- Streamlit dashboard.

Synapse 1.0 (AI in FinTech track) submission. Continuously "watches" live
buying/selling activity across a stock watchlist and raises alerts when
something breaks the normal pattern:
  - a sudden burst of trades
  - a price jump with no news behind it
  - several stocks moving together in a suspicious way

v2: adds a neural-autoencoder ensemble (two independent AI models voting),
market-manipulation surveillance (spoofing/layering/quote-stuffing/wash
trading) over a simulated order-event stream, a live animated replay mode
with a composite market-stress gauge, a 3D anomaly-feature-space view, a
correlation network diagram, browser voice alerts, and one-click PDF
incident reports.

v2.1 (post-review hardening): a background autorefresh in Live mode (so
"continuously watches" is a true statement, with a fixed, quotable
worst-case detection latency instead of "whenever someone last clicked
refresh"), a relevance-scored news check (TF-IDF + keyword match against
market-moving-event archetypes, not just "does a headline exist nearby"),
a numeric 0-100 confidence score per alert alongside the severity label,
and a correlation-engine concentration check that tells a genuine small
coordinated cluster apart from an ordinary market-wide move.

v2.2 (architecture upgrade, sprint 1): the Command Center header and Alert
Feed now refresh on their own via native `st.fragment(run_every=...)`
instead of the third-party `streamlit-autorefresh` package forcing a
full-page rerun -- cheaper (heavier tabs like the 3D landscape and
order-flow simulation no longer recompute every tick) and more targeted.
Live-mode fetches retry with backoff before falling back to simulated, and
a stale-data check degrades the "continuous monitoring" banner to a
"data delayed" warning once the last successful refresh is too old.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import random
import tempfile
import time
import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from config import CONFIG
from report_generator import generate_incident_report
from services import alert_service
from services.pipeline_service import load_live, resolve_and_score, run_order_flow
from stress_index import stress_label
from utils import design_tokens as tokens

from viz_extra import build_3d_landscape, build_correlation_network, speak_snippet

# --------------------------------------------------------------------------
# Palette -- the single source of truth now lives in utils/design_tokens.py
# (Sprint 4's design-token consolidation) so every page under pages/ shares
# it instead of each copy-pasting its own dark-mode constants. Re-exported
# under these same short names here purely so the rest of this
# already-large file doesn't need `tokens.` prefixed on every reference.
# --------------------------------------------------------------------------
SURFACE = tokens.SURFACE
PAGE = tokens.PAGE
INK_PRIMARY = tokens.INK_PRIMARY
INK_SECONDARY = tokens.INK_SECONDARY
INK_MUTED = tokens.INK_MUTED
GRIDLINE = tokens.GRIDLINE
BASELINE = tokens.BASELINE

CATEGORICAL = tokens.CATEGORICAL
STATUS = tokens.STATUS
STATUS_ICON = tokens.STATUS_ICON
INVESTIGATION_ICON = tokens.INVESTIGATION_ICON

KIND_LABELS = {
    "volume_burst": "Volume burst",
    "unexplained_price_jump": "Unexplained price jump",
    "explained_price_move": "Explained price move",
    "correlated_group_move": "Correlated group move",
    "market_wide_move": "Market-wide move (not a cluster)",
    "volatility_spike": "Volatility spike",
    "momentum_shift": "Momentum shift (EWMA crossover)",
}

SURVEILLANCE_LABELS = {
    "spoofing": "Spoofing", "layering": "Layering",
    "quote_stuffing": "Quote stuffing", "wash_trading": "Wash trading",
}

st.set_page_config(page_title="Unusual Activity Watchdog", layout="wide", page_icon="\U0001F6A8")

st.markdown(tokens.PAGE_CSS, unsafe_allow_html=True)


# The one dark Plotly theme every chart on this page uses -- shared with
# every page under pages/ via utils/design_tokens.py instead of each
# reimplementing the same template/gridcolor/margin settings.
plotly_base_layout = tokens.plotly_base_layout


def stress_gauge_figure(value: float, height: int = 220) -> go.Figure:
    label, color = stress_label(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "", "font": {"color": INK_PRIMARY, "size": 32}},
        title={"text": f"Market Stress -- {label}", "font": {"color": INK_SECONDARY, "size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": INK_MUTED, "tickfont": {"color": INK_MUTED, "size": 9}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": SURFACE,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 20], "color": "#1f3d2c"},
                {"range": [20, 45], "color": "#264a75"},
                {"range": [45, 70], "color": "#4a3a1a"},
                {"range": [70, 100], "color": "#4a2222"},
            ],
        },
    ))
    fig.update_layout(paper_bgcolor=SURFACE, font=dict(color=INK_SECONDARY), height=height,
                       margin=dict(l=20, r=20, t=40, b=10))
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("\U0001F6A8 Watchdog Controls")

mode = st.sidebar.radio(
    "Data source",
    ["Simulated demo (recommended)", "Live / historical (yfinance)"],
    help="Simulated mode guarantees a full, reproducible set of anomalies for "
         "a demo regardless of market hours or network access. Live mode pulls "
         "real intraday data via yfinance and reflects whatever is actually happening.",
)

default_watchlist = CONFIG.watchlist
watchlist = st.sidebar.multiselect("Watchlist", options=sorted(set(default_watchlist) | {
    "TSLA", "META", "NFLX", "BAC", "WMT"
}), default=default_watchlist)
if not watchlist:
    watchlist = default_watchlist

mode_choice = mode  # raw sidebar selection, captured before any live->simulated fallback
live_autorefresh = False
refresh_interval = CONFIG.live_autorefresh_seconds
if mode.startswith("Simulated"):
    seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=9999, value=12, step=1,
                                    help="Change this to generate a different (but equally reproducible) anomaly scenario.")
    if st.sidebar.button("\U0001F3B2 New random scenario"):
        seed = int(pd.Timestamp.now().timestamp()) % 9999 + 1
        st.session_state["seed_override"] = seed
    seed = st.session_state.get("seed_override", seed)
else:
    seed = None
    if st.sidebar.button("\U0001F504 Refresh live data now"):
        load_live.clear()
    live_autorefresh = st.sidebar.checkbox(
        "⏱️ Auto-refresh (continuous monitoring)", value=True,
        help=("Silently re-pulls live data and re-scores the Command Center header and Alert Feed on the "
              "interval below, with no click needed and no full-page rerun -- this is what makes "
              "\"continuously watches\" a true statement instead of only updating on page load or a manual "
              "refresh."),
    )
    refresh_interval = st.sidebar.select_slider(
        "Refresh interval",
        options=list(CONFIG.autorefresh_interval_choices),
        value=(CONFIG.live_autorefresh_seconds if CONFIG.live_autorefresh_seconds in CONFIG.autorefresh_interval_choices
               else CONFIG.autorefresh_interval_choices[len(CONFIG.autorefresh_interval_choices) // 2]),
        format_func=lambda s: f"{s}s",
        disabled=not live_autorefresh,
        help="How often the header and Alert Feed silently re-pull and re-score live data.",
    )

show_low_severity = st.sidebar.checkbox("Show low-severity items", value=False)
min_occurrences = st.sidebar.slider("Min. bars an event must persist", 1, 5, 1)
voice_alerts_enabled = st.sidebar.checkbox("\U0001F50A Voice alerts in Live Replay", value=False,
                                             help="Uses your browser's built-in text-to-speech to announce new "
                                                  "High-severity alerts as the replay plays through them.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Detection stack: an Isolation Forest + neural autoencoder ensemble (SHAP-explained, periodic "
    "retrospective sweeps of the recent window) for volume/price anomalies, a correlation-break engine "
    "with a concentration check (genuine small cluster vs. market-wide move) for cross-stock coordinated "
    "moves, a relevance-scored news check (not just \"a headline exists\"), and a rule-based order-flow "
    "surveillance layer for spoofing/layering/quote-stuffing/wash-trading. Every alert also carries a "
    "0-100 confidence score alongside its severity label."
)

# --------------------------------------------------------------------------
# Load data once for this full script run -- drives every tab except the
# two below that own their own st.fragment refresh -- with graceful
# fallback to simulated if live fails.
# --------------------------------------------------------------------------
_top_level_result = resolve_and_score(mode, watchlist, seed)
market = _top_level_result["market"]
bar_minutes = _top_level_result["bar_minutes"]
mode = _top_level_result["mode"]
seed = _top_level_result["seed"]
scored = _top_level_result["scored"]
breaks = _top_level_result["breaks"]
alerts = _top_level_result["alerts"]
stress_series = _top_level_result["stress_series"]
if _top_level_result["error"]:
    st.warning(_top_level_result["note"])

# --------------------------------------------------------------------------
# Header + KPIs + stress gauge, and (further below) the Alert Feed tab, each
# live in their own st.fragment(run_every=...) -- so in Live mode they
# silently re-pull and re-score on the configured interval WITHOUT rerunning
# the whole page. Heavier tabs (3D landscape, order-flow simulation, replay)
# only recompute on an actual full rerun: sidebar interaction, the manual
# refresh button, or initial page load. This is the native replacement for
# the prior streamlit-autorefresh-package approach, which forced a full
# script rerun on every tick.
# --------------------------------------------------------------------------
_live_refresh_active = mode_choice.startswith("Live") and live_autorefresh


def _render_autorefresh_status(resolved_mode: str, bar_minutes_: float):
    if not resolved_mode.startswith("Live"):
        return
    if not live_autorefresh:
        st.caption("⚪ Auto-refresh is off -- this pipeline recomputes on page load / manual refresh only right now.")
        return
    last_success = st.session_state.get("_last_live_success_ts")
    seconds_since = (pd.Timestamp.now() - last_success).total_seconds() if last_success is not None else None
    stale_cutoff = CONFIG.stale_data_multiplier * refresh_interval
    if seconds_since is not None and seconds_since > stale_cutoff:
        st.caption(f"\U0001F7E0 Market data delayed — last update {seconds_since:.0f}s ago.")
        return
    worst_case_latency_s = bar_minutes_ * 60 + refresh_interval
    st.caption(
        f"\U0001F7E2 Continuous monitoring active -- re-pulls and re-scores every "
        f"{refresh_interval}s. Worst-case detection latency ≈ {worst_case_latency_s:.0f}s "
        f"(bar interval + refresh interval), a fixed, quotable number instead of \"whenever someone last clicked refresh.\""
    )


def _render_demo_banner(resolved_mode: str):
    """Demo-mode banner (Section 27): only shown in Simulated mode, since
    "trigger a synthetic anomaly" would be dishonest applied to real Live
    data. Lets a presenter make something happen on demand instead of
    waiting for the scenario's own timeline or re-seeding to a fresh
    scenario -- the button stamps one large, unmistakable move onto the
    CURRENT scenario's latest bar (see pipeline_service._apply_demo_shock)
    so the very next refresh visibly flags it.
    """
    if not resolved_mode.startswith("Simulated"):
        return
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1:
            st.caption(
                "\U0001F3AC **Demo mode** -- data is simulated for a guaranteed, reproducible anomaly scenario. "
                "Use the button to make something happen right now instead of waiting for the scenario's own timeline."
            )
        with c2:
            if st.button("⚡ Trigger synthetic anomaly", key="trigger_demo_shock", use_container_width=True):
                st.session_state["_demo_shock"] = {
                    "ticker": random.choice(watchlist),
                    "magnitude": random.choice([-1, 1]) * random.uniform(0.05, 0.09),
                    "nonce": str(uuid.uuid4()),
                }
                st.rerun()
        if st.session_state.get("_demo_shock"):
            shock = st.session_state["_demo_shock"]
            if st.button("Clear injected anomaly", key="clear_demo_shock"):
                del st.session_state["_demo_shock"]
                st.rerun()
            else:
                st.caption(f"Last triggered: {shock['ticker']} ({shock['magnitude']:+.1%} on the latest bar).")


def _market_heatmap_figure(scored_by_ticker: dict, tickers: list[str]):
    """Real-time market heatmap (Section 27): every watched ticker's
    latest-bar return and volume z-scores in one glance -- the same
    signals driving alerts, but visible even when nothing has crossed the
    alert threshold yet. Refreshes with the rest of this fragment.
    """
    cols = [t for t in tickers if t in scored_by_ticker and not scored_by_ticker[t].empty]
    if not cols:
        return go.Figure()
    return_z = [float(scored_by_ticker[t]["return_zscore"].iloc[-1]) if "return_zscore" in scored_by_ticker[t] else 0.0 for t in cols]
    volume_z = [float(scored_by_ticker[t]["volume_zscore"].iloc[-1]) if "volume_zscore" in scored_by_ticker[t] else 0.0 for t in cols]
    z = [return_z, volume_z]
    fig = go.Figure(go.Heatmap(
        z=z, x=cols, y=["Return z-score", "Volume z-score"],
        zmid=0, zmin=-4, zmax=4, colorscale=[[0, "#1c5cab"], [0.5, SURFACE], [1, "#e66767"]],
        colorbar=dict(title="z"), text=[[f"{v:+.2f}" for v in row] for row in z], texttemplate="%{text}",
        hovertemplate="%{x}: %{y} = %{z:+.2f}<extra></extra>",
    ))
    fig = plotly_base_layout(fig, height=190)
    fig.update_layout(title="Real-time market heatmap (latest bar)", margin=dict(l=10, r=10, t=40, b=10))
    return fig


@st.fragment(run_every=refresh_interval if _live_refresh_active else None)
def render_command_center_header():
    result = resolve_and_score(mode_choice, watchlist, seed)
    if result["error"]:
        st.warning(result["note"])
    alerts_ = result["alerts"]
    stress_series_ = result["stress_series"]
    current_stress = float(stress_series_.iloc[-1]) if len(stress_series_) else 0.0

    alerts_view = alerts_[alerts_["severity"] != "Low"] if not show_low_severity else alerts_
    alerts_view = alerts_view[alerts_view["occurrences"] >= min_occurrences]

    st.title("\U0001F6A8 Unusual Activity Watchdog")
    # A visible, literal clock this fragment's own st.fragment(run_every=...)
    # advances on every tick with no click involved -- Section 32's "Fragment
    # refresh" check asks a manual tester to watch exactly this happen for at
    # least 3 cycles, which needs something on screen that actually changes
    # value each refresh (the surrounding prose captions below are static
    # text and don't serve that purpose on their own).
    st.caption(f"\U0001F552 Header last refreshed: {pd.Timestamp.now().strftime('%H:%M:%S')}")
    st.caption(
        "AI in FinTech · Synapse 1.0 -- watches buying/selling activity across a watchlist with periodic "
        "retrospective sweeps of the recent window (not literal tick-by-tick streaming inference) and raises "
        "alerts when something breaks the normal pattern."
    )
    if not result["error"] and result["note"]:
        st.caption(result["note"])
    _render_autorefresh_status(result["mode"], result["bar_minutes"])
    _render_demo_banner(result["mode"])

    kpi_col, gauge_col = st.columns([3, 1])
    with kpi_col:
        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Events flagged", len(alerts_view))
        r1c2.metric("High severity", int((alerts_view["severity"] == "High").sum()))
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Dual-model consensus", int(alerts_view["consensus"].sum()) if "consensus" in alerts_view.columns else 0)
        r2c2.metric("Tickers watched", len(watchlist))
    with gauge_col:
        st.plotly_chart(stress_gauge_figure(current_stress), use_container_width=True)

    st.plotly_chart(_market_heatmap_figure(result["scored"], watchlist), use_container_width=True)


render_command_center_header()

tab_feed, tab_explorer, tab_corr, tab_3d, tab_replay, tab_surveil, tab_accuracy = st.tabs([
    "\U0001F4E2 Alert Feed", "\U0001F4C8 Price & Volume", "\U0001F517 Correlation Monitor",
    "\U0001F9EC 3D & Network", "\U0001F3AC Live Replay", "\U0001F575️ Order-Flow Surveillance",
    "\U0001F3AF Detection Accuracy",
])

# --------------------------------------------------------------------------
# Tab 1: Alert feed -- its own st.fragment, refreshing on the same interval
# as the header above, independent of the heavier tabs below.
# --------------------------------------------------------------------------
@st.fragment(run_every=refresh_interval if _live_refresh_active else None)
def render_alert_feed():
    result = resolve_and_score(mode_choice, watchlist, seed)
    alerts_ = result["alerts"]
    scored = result["scored"]

    # Persist every alert this run produced (upsert -- an analyst's
    # investigation status/notes on an ongoing event are never touched by
    # this, only its freshness columns). Safe to call on every fragment
    # tick: see services/alert_service.sync_pipeline_alerts.
    new_count = alert_service.sync_pipeline_alerts(alerts_)

    alerts_view = alerts_[alerts_["severity"] != "Low"] if not show_low_severity else alerts_
    alerts_view = alerts_view[alerts_view["occurrences"] >= min_occurrences]

    if new_count:
        st.caption(f"\U0001F4BE {new_count} new event(s) saved to the investigation log this refresh.")

    if alerts_view.empty:
        st.info("No events matching the current filters.")
    else:
        display_df = alerts_view.copy()
        display_df["kind"] = display_df["kind"].map(KIND_LABELS).fillna(display_df["kind"])
        display_df["window"] = display_df.apply(
            lambda r: r["first_seen"].strftime("%b %d, %H:%M") if r["first_seen"] == r["last_seen"]
            else f"{r['first_seen'].strftime('%b %d, %H:%M')} → {r['last_seen'].strftime('%H:%M')}",
            axis=1,
        )
        display_df["severity_badge"] = display_df["severity"].map(lambda s: f"{STATUS_ICON[s]} {s}")
        display_df["ai"] = display_df["consensus"].map(lambda c: "\U0001F916\U0001F916 Consensus" if c else "\U0001F916 Single model")
        display_df["confidence_badge"] = display_df.get("confidence", 0.0).map(lambda c: f"{c:.0f}/100")

        # Join in each row's persisted investigation status via the same
        # (kind, tickers, first_seen) key data/store.py upserts on -- one
        # bulk fetch, not a round trip per row. `display_df["kind"]` was
        # already relabeled to its display string above, so the key lookup
        # uses alerts_view's RAW kind column (same row order/index).
        persisted = alert_service.fetch_alerts()
        status_by_key = dict(zip(persisted["alert_key"], persisted["status"])) if not persisted.empty else {}
        display_df["investigation_status"] = [
            status_by_key.get(alert_service.alert_key(k, t, fs), alert_service.STATUS_NEW)
            for k, t, fs in zip(alerts_view["kind"], alerts_view["tickers"], alerts_view["first_seen"])
        ]
        display_df["status_badge"] = display_df["investigation_status"].map(
            lambda s: f"{INVESTIGATION_ICON.get(s, '')} {s}"
        )

        st.dataframe(
            display_df[["window", "severity_badge", "confidence_badge", "ai", "kind", "tickers", "headline",
                        "occurrences", "status_badge"]].rename(columns={
                "window": "When", "severity_badge": "Severity", "confidence_badge": "Confidence", "ai": "AI agreement",
                "kind": "Type", "tickers": "Ticker(s)", "headline": "Alert", "occurrences": "Bars",
                "status_badge": "Investigation",
            }),
            use_container_width=True, hide_index=True, height=380,
        )

        st.markdown("#### Inspect an event")
        options = list(display_df.index)
        labels = [f"{display_df.loc[i, 'window']} — {display_df.loc[i, 'headline']}" for i in options]
        if options:
            picked = st.selectbox("Choose an alert to explain:", options, format_func=lambda i: labels[options.index(i)])
            row = alerts_view.loc[picked]

            colL, colR = st.columns([3, 2])
            with colL:
                st.markdown(f"**{row['headline']}**")
                st.write(row["detail"])
                badge_color = STATUS[row["severity"]]
                consensus_badge = (
                    f"<span class='watchdog-badge' style='background:#e6676722;color:#e66767;'>"
                    f"\U0001F916\U0001F916 Dual-model consensus</span>" if row.get("consensus") else ""
                )
                st.markdown(
                    f"<span class='watchdog-badge' style='background:{badge_color}22;color:{badge_color};'>"
                    f"{row['severity']} severity</span>"
                    f"<span class='watchdog-badge' style='background:{CATEGORICAL[0]}22;color:{CATEGORICAL[0]};'>"
                    f"Confidence {row.get('confidence', 0):.0f}/100</span>"
                    f"<span class='watchdog-badge' style='background:{INK_MUTED}22;color:{INK_SECONDARY};'>"
                    f"{row['occurrences']} bar(s)</span>{consensus_badge}",
                    unsafe_allow_html=True,
                )

                if st.button("\U0001F4C4 Generate PDF incident report", key=f"report_{picked}"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        generate_incident_report(row.to_dict(), scored, tmp.name)
                        with open(tmp.name, "rb") as f:
                            pdf_bytes = f.read()
                    st.download_button("⬇️ Download report", data=pdf_bytes,
                                        file_name=f"watchdog_incident_{picked}.pdf", mime="application/pdf",
                                        key=f"dl_{picked}")

                st.markdown("##### Investigation")
                db_id = alert_service.lookup_id(row["kind"], row["tickers"], row["first_seen"])
                if db_id is None:
                    st.caption("Not yet saved to the investigation log (will appear on the next refresh).")
                else:
                    stored = alert_service.get_alert(db_id)
                    current_status = stored["status"] if stored else alert_service.STATUS_NEW
                    st.caption(f"{INVESTIGATION_ICON.get(current_status, '')} Current status: **{current_status}**")
                    new_status = st.selectbox(
                        "Update status", alert_service.ALLOWED_STATUSES,
                        index=alert_service.ALLOWED_STATUSES.index(current_status),
                        key=f"status_select_{db_id}",
                    )
                    note = st.text_input("Add a note (optional)", key=f"status_note_{db_id}")
                    if st.button("Save investigation update", key=f"status_save_{db_id}"):
                        if new_status != current_status or note:
                            alert_service.transition_status(db_id, new_status, note=note or None)
                            st.success("Saved.")
                            # Without this, "Current status" above still shows
                            # the value read at the TOP of this same run (the
                            # status as of BEFORE this click), which reads as
                            # a stale/broken save even though it wrote fine --
                            # found while walking Section 32's persistence
                            # test-plan row end to end. st.rerun() inside a
                            # fragment reruns just this fragment, not the
                            # whole page, so it's cheap.
                            st.rerun()
                        else:
                            st.info("No change to save.")
                    if stored and stored.get("notes"):
                        with st.expander("Investigation notes history"):
                            st.text(stored["notes"])

            with colR:
                if row["kind"] in ("volume_burst", "unexplained_price_jump", "explained_price_move", "volatility_spike"):
                    drivers = row["evidence"].get("drivers", [])
                    if drivers:
                        fig = go.Figure(go.Bar(
                            x=[v for _, v in drivers], y=[k for k, _ in drivers], orientation="h",
                            marker_color=[CATEGORICAL[0] if v >= 0 else CATEGORICAL[1] for _, v in drivers],
                        ))
                        fig.update_layout(title="SHAP drivers of this alert", height=220,
                                           margin=dict(l=10, r=10, t=40, b=10),
                                           template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                                           font=dict(color=INK_SECONDARY, size=12))
                        st.plotly_chart(fig, use_container_width=True)
                    news_state = row["evidence"].get("news_state")
                    if news_state:
                        icon = {"NEWS EXPLAINS MOVEMENT": "\U0001F4F0", "POSSIBLE NEWS-MOVEMENT MISMATCH": "❓",
                                "NO RELEVANT NEWS DETECTED": "\U0001F507"}.get(news_state, "")
                        st.caption(f"{icon} **{news_state}**")
                        gap_text = row["evidence"].get("news_gap_text")
                        if gap_text:
                            st.caption(gap_text.capitalize() + ".")
                elif row["kind"] in ("correlated_group_move", "market_wide_move"):
                    st.metric("Correlation delta vs baseline", f"{row['evidence'].get('corr_delta', 0):+.2f}")
                    st.metric("Unusualness (z-score)", f"{row['evidence'].get('delta_zscore', 0):.2f}")
                    ratio = row["evidence"].get("concentration_ratio")
                    if ratio is not None:
                        st.metric(
                            "Concentration ratio",
                            f"{ratio:.2f}x",
                            help=f"Cluster threshold is {CONFIG.correlation_concentration_ratio_threshold:.1f}x -- "
                                 "above it, a small subset moved much more than the rest of the book (a genuine "
                                 "cluster); at/below it, roughly the whole watchlist moved together (market-wide).",
                        )
                elif row["kind"] == "momentum_shift":
                    st.metric("Crossover z-score", f"{row['evidence'].get('crossover_zscore', 0):.2f}")
                    st.metric("Short EWMA", f"{row['evidence'].get('ewma_short', 0):+.4f}")
                    st.metric("Long EWMA", f"{row['evidence'].get('ewma_long', 0):+.4f}",
                              help="A short-window return EWMA crossing a long-window one by an unusual amount -- "
                                   "trend divergence, not a single-bar price or volume spike.")

            # Ticker-level chart for the primary ticker(s) in this alert
            for tkr in row["tickers"].split(", "):
                if tkr in scored:
                    df = scored[tkr]
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.05)
                    fig.add_trace(go.Scatter(x=df.index, y=df["price"], mode="lines", name=f"{tkr} price",
                                              line=dict(color=CATEGORICAL[0], width=2)), row=1, col=1)
                    anomalies = df[df["any_model_flagged"]] if "any_model_flagged" in df.columns else df[df["is_anomaly"]]
                    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["price"], mode="markers",
                                              name="Flagged", marker=dict(color=CATEGORICAL[7], size=9, symbol="circle-open", line=dict(width=2))),
                                  row=1, col=1)
                    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name=f"{tkr} volume",
                                          marker_color=CATEGORICAL[2]), row=2, col=1)
                    fig = plotly_base_layout(fig, height=380)
                    fig.update_layout(title=f"{tkr}: price & volume with flagged bars")
                    st.plotly_chart(fig, use_container_width=True)


with tab_feed:
    render_alert_feed()

# --------------------------------------------------------------------------
# Tab 2: Price & Volume explorer
# --------------------------------------------------------------------------
with tab_explorer:
    tkr = st.selectbox("Ticker", list(scored.keys()))
    df = scored[tkr]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                         subplot_titles=("Price", "Volume", "Return z-score"))
    fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", line=dict(color=CATEGORICAL[0], width=2)), row=1, col=1)
    anomalies = df[df["any_model_flagged"]] if "any_model_flagged" in df.columns else df[df["is_anomaly"]]
    fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["price"], mode="markers", name="Flagged",
                              marker=dict(color=CATEGORICAL[7], size=9, symbol="circle-open", line=dict(width=2))), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=CATEGORICAL[2]), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["return_zscore"], name="Return z", line=dict(color=CATEGORICAL[3], width=2)), row=3, col=1)
    fig.add_hline(y=CONFIG.price_jump_sigma_alert, line_dash="dot", line_color=STATUS["High"], row=3, col=1)
    fig.add_hline(y=-CONFIG.price_jump_sigma_alert, line_dash="dot", line_color=STATUS["High"], row=3, col=1)
    fig = plotly_base_layout(fig, height=680)
    fig.update_layout(showlegend=False, title=f"{tkr}: full detail view")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3: Correlation monitor
# --------------------------------------------------------------------------
with tab_corr:
    st.markdown("#### Rolling average pairwise correlation vs baseline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=breaks.index, y=breaks["avg_corr_current"], name="Current (rolling)",
                              line=dict(color=CATEGORICAL[0], width=2)))
    fig.add_trace(go.Scatter(x=breaks.index, y=breaks["avg_corr_baseline"], name="Baseline",
                              line=dict(color=INK_MUTED, width=2, dash="dot")))
    flagged_pts = breaks[breaks["is_break"]]
    fig.add_trace(go.Scatter(x=flagged_pts.index, y=flagged_pts["avg_corr_current"], mode="markers",
                              name="Break flagged", marker=dict(color=STATUS["Medium"], size=9)))
    fig = plotly_base_layout(fig, height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Current vs. baseline correlation matrix (most recent window)")
    win, base_win = CONFIG.correlation_window, CONFIG.correlation_baseline_window
    returns = market["returns"]
    current_corr = returns.iloc[-win:].corr()
    baseline_corr = returns.iloc[-(base_win + win):-win].corr()

    colA, colB = st.columns(2)
    for col, corr, label in [(colA, baseline_corr, "Baseline"), (colB, current_corr, "Current")]:
        with col:
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                zmin=-1, zmax=1, colorscale=[[0, "#1c5cab"], [0.5, SURFACE], [1, "#e66767"]],
                colorbar=dict(title="corr"),
            ))
            fig = plotly_base_layout(fig, height=340)
            fig.update_layout(title=label)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 4: 3D anomaly landscape + correlation network
# --------------------------------------------------------------------------
with tab_3d:
    st.markdown("#### Correlation network")
    st.caption("Nodes = tickers. Edge color = sign of correlation, width = strength. "
               "Weak pairs (|corr| < 0.15) are hidden for legibility. **Click a node** for a quick "
               "look at that ticker without leaving this tab.")
    latest_break = breaks.iloc[-1] if len(breaks) else None
    highlight = latest_break["cluster"] if latest_break is not None and latest_break["is_break"] else []
    net_fig = build_correlation_network(market["returns"].iloc[-CONFIG.correlation_window:], highlight_cluster=highlight)
    net_event = st.plotly_chart(net_fig, use_container_width=True, on_select="rerun", key="corr_network_select")

    clicked_points = (net_event or {}).get("selection", {}).get("points", [])
    # `customdata` was set on the node trace as a flat, one-value-per-point
    # list (tickers), so Plotly's selection event gives each point's
    # customdata back as the bare ticker string -- NOT a per-point list, so
    # indexing it with [0] (as a first pass at this did) silently returns
    # just the first character instead of the ticker.
    clicked_ticker = next((p.get("customdata") for p in clicked_points if p.get("customdata")), None)
    if clicked_ticker:
        st.session_state["_selected_ticker"] = clicked_ticker

    focus_ticker = st.session_state.get("_selected_ticker")
    if focus_ticker and focus_ticker in scored:
        with st.container(border=True):
            fdf = scored[focus_ticker]
            latest_row = fdf.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{focus_ticker} price", f"${latest_row['price']:.2f}")
            c2.metric("Return z-score", f"{latest_row.get('return_zscore', 0):.2f}")
            c3.metric("Volume z-score", f"{latest_row.get('volume_zscore', 0):.2f}")
            in_cluster = bool(highlight) and focus_ticker in highlight
            c4.metric("In flagged cluster?", "Yes" if in_cluster else "No")
            st.page_link("pages/1_Stock_Terminal.py", label=f"Open {focus_ticker} in the Stock Terminal →", icon="\U0001F50D")

    st.markdown("---")
    st.markdown("#### 3D anomaly feature-space landscape")
    st.caption("Every bar plotted by (return z-score, volume z-score, volatility). Color = ensemble anomaly "
               "score; diamonds are bars where BOTH the Isolation Forest and the autoencoder agree.")
    tkr3d = st.selectbox("Ticker", list(scored.keys()), key="tkr_3d")
    st.plotly_chart(build_3d_landscape(scored[tkr3d], tkr3d), use_container_width=True)

# --------------------------------------------------------------------------
# Tab 5: Live Replay
# --------------------------------------------------------------------------
with tab_replay:
    st.markdown("#### Watch the AI detect anomalies as they happen")
    st.caption("Scrubs through the scenario bar-by-bar, revealing price action and alerts only up to the "
               "current point in time -- exactly what an analyst watching this system live would see.")

    idx = market["prices"].index
    n = len(idx)
    start_at = min(CONFIG.correlation_baseline_window + CONFIG.correlation_window, n - 1)

    state_key = f"playhead_{seed}_{mode}_{len(watchlist)}"
    if state_key not in st.session_state:
        st.session_state[state_key] = start_at
    if "spoken_alerts" not in st.session_state:
        st.session_state["spoken_alerts"] = set()

    # A widget's session_state key can't be written to after that widget has
    # been instantiated in the same script run (Streamlit raises
    # StreamlitWidgetAlreadyInstantiatedError) -- so the auto-play advance
    # requested at the bottom of the PREVIOUS run is applied here, before
    # the slider below (which owns `state_key`) is created this run.
    if st.session_state.get("_replay_pending_advance"):
        st.session_state[state_key] = min(st.session_state[state_key] + 1, n - 1)
        st.session_state["_replay_pending_advance"] = False

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 3])
    auto = ctrl1.checkbox("▶ Auto-play", value=False, key="replay_auto")
    speed = ctrl2.slider("Speed", 1, 12, 5, key="replay_speed")
    if ctrl3.button("⏮ Reset"):
        st.session_state[state_key] = start_at
        st.session_state["spoken_alerts"] = set()
    # IMPORTANT: the slider's `key` IS `state_key` -- once a widget has a
    # key, Streamlit's displayed value is driven entirely by
    # st.session_state[key] on every subsequent rerun, and a `value=`
    # argument passed alongside an existing key is silently ignored. So the
    # auto-play loop below advances the playhead by writing directly to
    # st.session_state[state_key] before the widget re-renders -- that's
    # the only way a slider with a key can be driven programmatically.
    playhead = ctrl4.slider("Scrub timeline", start_at, n - 1, key=state_key)

    ph = st.session_state[state_key]
    current_ts = idx[ph]
    st.markdown(f"**Current time:** {current_ts.strftime('%b %d, %Y %H:%M')}  ({ph + 1}/{n} bars)")

    replay_stress = float(stress_series.reindex([current_ts], method="ffill").iloc[0]) if len(stress_series) else 0.0
    gcol, acol = st.columns([1, 2])
    with gcol:
        st.plotly_chart(stress_gauge_figure(replay_stress, height=200), use_container_width=True)
    with acol:
        alerts_so_far = alerts[alerts["first_seen"] <= current_ts].sort_values("first_seen", ascending=False)
        st.markdown(f"**Alerts detected so far: {len(alerts_so_far)}**")
        if not alerts_so_far.empty:
            recent = alerts_so_far.head(6).copy()
            recent["badge"] = recent["severity"].map(lambda s: f"{STATUS_ICON[s]} {s}")
            st.dataframe(recent[["badge", "kind", "tickers", "headline"]].rename(
                columns={"badge": "Severity", "kind": "Type", "tickers": "Ticker(s)", "headline": "Alert"}),
                use_container_width=True, hide_index=True, height=200)

            newest = alerts_so_far.iloc[0]
            alert_key = f"{newest['kind']}|{newest['tickers']}|{newest['first_seen']}"
            if voice_alerts_enabled and newest["severity"] == "High" and alert_key not in st.session_state["spoken_alerts"]:
                st.session_state["spoken_alerts"].add(alert_key)
                components.html(speak_snippet(f"Watchdog alert. {newest['headline']}."), height=0)

    price_fig = go.Figure()
    for i, wtkr in enumerate(watchlist[:5]):  # cap traces for legibility
        pdf = scored[wtkr].iloc[:ph + 1]
        norm = pdf["price"] / pdf["price"].iloc[0] * 100
        price_fig.add_trace(go.Scatter(x=pdf.index, y=norm, name=wtkr, line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2)))
    price_fig = plotly_base_layout(price_fig, height=360)
    price_fig.update_layout(title="Watchlist performance so far (indexed to 100 at scenario start)")
    st.plotly_chart(price_fig, use_container_width=True)

    if auto and ph < n - 1:
        time.sleep(1.0 / speed)
        st.session_state["_replay_pending_advance"] = True
        st.rerun()

# --------------------------------------------------------------------------
# Tab 6: Order-flow surveillance
# --------------------------------------------------------------------------
with tab_surveil:
    st.markdown("#### Exchange-grade order-flow surveillance")
    st.caption(
        "Free market data gives price/volume bars, not the underlying order-by-order event stream real "
        "exchange surveillance desks watch. This tab simulates that finer-grained stream (with known, "
        "injected manipulation patterns) and runs the same rule shapes real surveillance systems use as "
        "their first-line detectors: **spoofing, layering, quote stuffing, and wash trading.**"
    )
    surveil_tkr = st.selectbox("Ticker to inspect", watchlist, key="surveil_tkr")
    order_seed = st.number_input("Order-flow scenario seed", min_value=1, max_value=9999, value=3, step=1)
    result, flags = run_order_flow(surveil_tkr, order_seed)
    events = result["events"]

    m1, m2, m3 = st.columns(3)
    m1.metric("Order events simulated", f"{len(events):,}")
    m2.metric("Distinct traders", events["trader_id"].nunique())
    m3.metric("Patterns flagged", len(flags))

    if flags.empty:
        st.info("No manipulation patterns flagged for this scenario.")
    else:
        disp = flags.copy()
        disp["kind"] = disp["kind"].map(SURVEILLANCE_LABELS).fillna(disp["kind"])
        disp["badge"] = disp["severity"].map(lambda s: f"{STATUS_ICON[s]} {s}")
        st.dataframe(
            disp[["timestamp", "badge", "kind", "trader_id", "detail"]].rename(columns={
                "timestamp": "Time", "badge": "Severity", "kind": "Pattern", "trader_id": "Trader", "detail": "Detail",
            }),
            use_container_width=True, hide_index=True, height=260,
        )

    st.markdown("#### Order-event timeline")
    plot_events = events.copy()
    plot_events["y"] = plot_events["side"].map({"buy": 1, "sell": -1}) * (plot_events["size"] / plot_events["size"].max() * 0.8 + 0.2)
    color_map = {"add": CATEGORICAL[0], "cancel": CATEGORICAL[7], "execute": CATEGORICAL[2]}
    fig = go.Figure()
    for ev_type, color in color_map.items():
        sub = plot_events[plot_events["event"] == ev_type]
        fig.add_trace(go.Scatter(x=sub["timestamp"], y=sub["y"], mode="markers", name=ev_type,
                                  marker=dict(color=color, size=5, opacity=0.55)))
    for _, f in flags.iterrows():
        fig.add_vline(x=f["timestamp"], line_dash="dot", line_color=STATUS[f["severity"]], opacity=0.6)
    fig = plotly_base_layout(fig, height=340)
    fig.update_layout(title=f"{surveil_tkr}: order events (buy = positive, sell = negative; dotted lines = flagged patterns)",
                       yaxis_title="side / relative size")
    st.plotly_chart(fig, use_container_width=True)

    if result["ground_truth"]:
        with st.expander("Injected ground truth (for validating the detectors)"):
            for g in result["ground_truth"]:
                st.markdown(f"- **{SURVEILLANCE_LABELS.get(g['kind'], g['kind'])}** @ {g['timestamp']}: {g['description']}")

# --------------------------------------------------------------------------
# Tab 7: Detection accuracy (only meaningful with known ground truth, i.e. simulated mode)
# --------------------------------------------------------------------------
with tab_accuracy:
    if "ground_truth" not in market:
        st.info("Detection accuracy is only computable against the simulated scenario's known ground truth. "
                "Switch to 'Simulated demo' mode to see it.")
    else:
        st.markdown(
            "The simulated scenario injects a known set of anomalies, so detection can be checked "
            "against ground truth -- something that isn't possible with unlabeled live data. "
            "This is for demoing/validating the detector, not a claim about real-world accuracy."
        )
        gt = market["ground_truth"]
        rows = []
        for g in gt:
            ts = g["timestamp"]
            window = pd.Timedelta(minutes=90)
            nearby = alerts[(alerts["first_seen"] <= ts + window) & (alerts["last_seen"] >= ts - window)]
            if g["kind"] == "correlated_group_move":
                match = nearby[nearby["kind"] == "correlated_group_move"]
                caught = any(len(set(g["tickers"]) & set(t.split(", "))) >= 2 for t in match["tickers"])
            else:
                match = nearby[nearby["tickers"].isin(g["tickers"])]
                caught = not match.empty
            rows.append({
                "Injected anomaly": g["description"], "Kind": KIND_LABELS.get(g["kind"], g["kind"]),
                "Timestamp": ts.strftime("%b %d, %H:%M"), "Caught?": "✅ Yes" if caught else "❌ Missed",
            })
        acc_df = pd.DataFrame(rows)
        hit_rate = (acc_df["Caught?"] == "✅ Yes").mean() * 100
        st.metric("Recall on injected anomalies", f"{hit_rate:.0f}%")
        st.dataframe(acc_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Detection stack: Isolation Forest + neural-autoencoder ensemble (SHAP-explained) for volume/price "
    "anomalies, a correlation-break engine for cross-stock coordinated moves, and rule-based order-flow "
    "surveillance for spoofing/layering/quote-stuffing/wash-trading. See README.md for full architecture, "
    "limitations, and how to point this at real live data."
)
