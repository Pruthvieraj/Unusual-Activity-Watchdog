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

Run with:  streamlit run app.py
"""

from __future__ import annotations

import tempfile
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from alert_engine import build_synthetic_news_lookup, generate_all_alerts
from config import CONFIG
from correlation_watch import detect_correlation_breaks
from data.synthetic import generate_market_tape
from features import build_all_features
from models.ensemble import build_ensemble
from order_flow import run_surveillance
from report_generator import generate_incident_report
from stress_index import compute_stress_series, stress_label
from viz_extra import build_3d_landscape, build_correlation_network, speak_snippet

# --------------------------------------------------------------------------
# Palette (validated dark-mode set -- see the dataviz design pass this app
# was built against). Categorical slots keep a fixed order across the app;
# status colors are reserved for severity and never reused as series colors.
# --------------------------------------------------------------------------
SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
BASELINE = "#383835"

CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
STATUS = {"High": "#e66767", "Medium": "#c98500", "Low": "#898781"}
STATUS_ICON = {"High": "\U0001F534", "Medium": "\U0001F7E1", "Low": "⚪"}

KIND_LABELS = {
    "volume_burst": "Volume burst",
    "unexplained_price_jump": "Unexplained price jump",
    "explained_price_move": "Explained price move",
    "correlated_group_move": "Correlated group move",
    "volatility_spike": "Volatility spike",
}

SURVEILLANCE_LABELS = {
    "spoofing": "Spoofing", "layering": "Layering",
    "quote_stuffing": "Quote stuffing", "wash_trading": "Wash trading",
}

st.set_page_config(page_title="Unusual Activity Watchdog", layout="wide", page_icon="\U0001F6A8")

st.markdown(f"""
<style>
  .stApp {{ background-color: {PAGE}; }}
  section[data-testid="stSidebar"] {{ background-color: {SURFACE}; }}
  div[data-testid="stMetric"] {{
      background-color: {SURFACE}; border: 1px solid {GRIDLINE};
      border-radius: 10px; padding: 12px 16px;
  }}
  .watchdog-badge {{
      display: inline-block; padding: 2px 10px; border-radius: 999px;
      font-size: 0.78rem; font-weight: 600; margin-right: 6px;
  }}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data pipeline (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_simulated(seed: int, tickers: tuple[str, ...]):
    return generate_market_tape(tickers=list(tickers), seed=seed)


@st.cache_data(show_spinner=False)
def load_live(tickers: tuple[str, ...]):
    from data.live_feed import fetch_intraday
    return fetch_intraday(tickers=list(tickers))


@st.cache_data(show_spinner=False)
def run_pipeline(market: dict, news_df_key: str, bar_minutes: float):
    features = build_all_features(market)
    scored = build_ensemble(features)
    breaks = detect_correlation_breaks(market["returns"])
    news_df = market.get("news")
    news_lookup = build_synthetic_news_lookup(news_df, CONFIG.news_lookback_hours) if news_df is not None else None
    alerts = generate_all_alerts(scored, breaks, news_lookup, include_low_severity=True, merge_bar_minutes=bar_minutes)
    stress = compute_stress_series(scored, breaks)
    return scored, breaks, alerts, stress


@st.cache_data(show_spinner=False)
def run_order_flow(ticker: str, seed: int):
    from data.orderbook_synthetic import generate_order_events
    result = generate_order_events(
        ticker, n_minutes=CONFIG.orderbook_minutes, events_per_minute=CONFIG.orderbook_events_per_minute,
        n_traders=CONFIG.orderbook_n_traders, seed=seed,
    )
    flags = run_surveillance(result["events"])
    return result, flags


def plotly_base_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY, size=13),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=BASELINE)
    fig.update_yaxes(gridcolor=GRIDLINE, zerolinecolor=BASELINE, linecolor=BASELINE)
    return fig


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

if mode.startswith("Simulated"):
    seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=9999, value=12, step=1,
                                    help="Change this to generate a different (but equally reproducible) anomaly scenario.")
    if st.sidebar.button("\U0001F3B2 New random scenario"):
        seed = int(pd.Timestamp.now().timestamp()) % 9999 + 1
        st.session_state["seed_override"] = seed
    seed = st.session_state.get("seed_override", seed)
else:
    seed = None
    if st.sidebar.button("\U0001F504 Refresh live data"):
        load_live.clear()

show_low_severity = st.sidebar.checkbox("Show low-severity items", value=False)
min_occurrences = st.sidebar.slider("Min. bars an event must persist", 1, 5, 1)
voice_alerts_enabled = st.sidebar.checkbox("\U0001F50A Voice alerts in Live Replay", value=False,
                                             help="Uses your browser's built-in text-to-speech to announce new "
                                                  "High-severity alerts as the replay plays through them.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Detection stack: an Isolation Forest + neural autoencoder ensemble (SHAP-explained) for "
    "volume/price anomalies, a correlation-break engine for cross-stock coordinated moves, and a "
    "rule-based order-flow surveillance layer for spoofing/layering/quote-stuffing/wash-trading."
)

# --------------------------------------------------------------------------
# Load data (with graceful fallback to simulated if live fails)
# --------------------------------------------------------------------------
data_source_note = ""
if mode.startswith("Simulated"):
    market = load_simulated(seed, tuple(watchlist))
    bar_minutes = CONFIG.simulate_bar_freq_minutes
else:
    try:
        market = load_live(tuple(watchlist))
        if market["prices"].dropna(how="all").empty:
            raise RuntimeError("empty response")
        bar_minutes = 5 if CONFIG.intraday_interval.endswith("m") else 1440
        data_source_note = "Live data via yfinance."
    except Exception as e:
        st.warning(
            f"Couldn't reach live market data from this environment ({e}). "
            "Falling back to the simulated demo scenario so the dashboard still works -- "
            "on a machine with normal internet access, live mode pulls real yfinance data."
        )
        market = load_simulated(12, tuple(watchlist))
        bar_minutes = CONFIG.simulate_bar_freq_minutes
        mode = "Simulated demo (recommended)"
        seed = 12

news_key = "synthetic" if "news" in market else "none"
scored, breaks, alerts, stress_series = run_pipeline(market, news_key, bar_minutes)
current_stress = float(stress_series.iloc[-1]) if len(stress_series) else 0.0

if not show_low_severity:
    alerts_view = alerts[alerts["severity"] != "Low"]
else:
    alerts_view = alerts
alerts_view = alerts_view[alerts_view["occurrences"] >= min_occurrences]

# --------------------------------------------------------------------------
# Header + KPIs + stress gauge
# --------------------------------------------------------------------------
st.title("\U0001F6A8 Unusual Activity Watchdog")
st.caption(
    "AI in FinTech · Synapse 1.0 -- continuously watches live buying/selling activity "
    "across a watchlist and raises alerts when something breaks the normal pattern."
)
if data_source_note:
    st.caption(data_source_note)

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

tab_feed, tab_explorer, tab_corr, tab_3d, tab_replay, tab_surveil, tab_accuracy = st.tabs([
    "\U0001F4E2 Alert Feed", "\U0001F4C8 Price & Volume", "\U0001F517 Correlation Monitor",
    "\U0001F9EC 3D & Network", "\U0001F3AC Live Replay", "\U0001F575️ Order-Flow Surveillance",
    "\U0001F3AF Detection Accuracy",
])

# --------------------------------------------------------------------------
# Tab 1: Alert feed
# --------------------------------------------------------------------------
with tab_feed:
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

        st.dataframe(
            display_df[["window", "severity_badge", "ai", "kind", "tickers", "headline", "occurrences"]].rename(columns={
                "window": "When", "severity_badge": "Severity", "ai": "AI agreement", "kind": "Type",
                "tickers": "Ticker(s)", "headline": "Alert", "occurrences": "Bars",
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
                elif row["kind"] == "correlated_group_move":
                    st.metric("Correlation delta vs baseline", f"{row['evidence'].get('corr_delta', 0):+.2f}")
                    st.metric("Unusualness (z-score)", f"{row['evidence'].get('delta_zscore', 0):.2f}")

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
               "Weak pairs (|corr| < 0.15) are hidden for legibility.")
    latest_break = breaks.iloc[-1] if len(breaks) else None
    highlight = latest_break["cluster"] if latest_break is not None and latest_break["is_break"] else []
    net_fig = build_correlation_network(market["returns"].iloc[-CONFIG.correlation_window:], highlight_cluster=highlight)
    st.plotly_chart(net_fig, use_container_width=True)

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
