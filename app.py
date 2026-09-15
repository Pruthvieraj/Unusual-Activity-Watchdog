"""
Unusual Activity Watchdog -- Streamlit dashboard.

Synapse 1.0 (AI in FinTech track) submission. Continuously "watches" live
buying/selling activity across a stock watchlist and raises alerts when
something breaks the normal pattern:
  - a sudden burst of trades
  - a price jump with no news behind it
  - several stocks moving together in a suspicious way

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from alert_engine import build_synthetic_news_lookup, generate_all_alerts
from config import CONFIG
from correlation_watch import detect_correlation_breaks
from data.synthetic import generate_market_tape
from features import build_all_features
from models.anomaly_model import FEATURE_COLUMNS, score_all

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
    scored = score_all(features)
    breaks = detect_correlation_breaks(market["returns"])
    news_df = market.get("news")
    news_lookup = build_synthetic_news_lookup(news_df, CONFIG.news_lookback_hours) if news_df is not None else None
    alerts = generate_all_alerts(scored, breaks, news_lookup, include_low_severity=True, merge_bar_minutes=bar_minutes)
    return scored, breaks, alerts


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

st.sidebar.markdown("---")
st.sidebar.caption(
    "Detects: sudden volume bursts, price jumps with no matching news, and "
    "stocks moving together outside their normal correlation -- flagged by a "
    "per-stock Isolation Forest (explained with SHAP) plus a cross-stock "
    "correlation-break engine."
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

news_key = "synthetic" if "news" in market else "none"
scored, breaks, alerts = run_pipeline(market, news_key, bar_minutes)

if not show_low_severity:
    alerts_view = alerts[alerts["severity"] != "Low"]
else:
    alerts_view = alerts
alerts_view = alerts_view[alerts_view["occurrences"] >= min_occurrences]

# --------------------------------------------------------------------------
# Header + KPIs
# --------------------------------------------------------------------------
st.title("\U0001F6A8 Unusual Activity Watchdog")
st.caption(
    "AI in FinTech · Synapse 1.0 -- continuously watches live buying/selling activity "
    "across a watchlist and raises alerts when something breaks the normal pattern."
)
if data_source_note:
    st.caption(data_source_note)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Events flagged", len(alerts_view))
c2.metric("High severity", int((alerts_view["severity"] == "High").sum()))
c3.metric("Tickers watched", len(watchlist))
c4.metric("Bars analyzed", len(market["prices"]))

tab_feed, tab_explorer, tab_corr, tab_accuracy = st.tabs(
    ["\U0001F4E2 Alert Feed", "\U0001F4C8 Price & Volume", "\U0001F517 Correlation Monitor", "\U0001F3AF Detection Accuracy"]
)

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

        st.dataframe(
            display_df[["window", "severity_badge", "kind", "tickers", "headline", "occurrences"]].rename(columns={
                "window": "When", "severity_badge": "Severity", "kind": "Type",
                "tickers": "Ticker(s)", "headline": "Alert", "occurrences": "Bars",
            }),
            use_container_width=True, hide_index=True, height=420,
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
                st.markdown(
                    f"<span class='watchdog-badge' style='background:{badge_color}22;color:{badge_color};'>"
                    f"{row['severity']} severity</span>"
                    f"<span class='watchdog-badge' style='background:{INK_MUTED}22;color:{INK_SECONDARY};'>"
                    f"{row['occurrences']} bar(s)</span>",
                    unsafe_allow_html=True,
                )

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
                    anomalies = df[df["is_anomaly"]]
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
    anomalies = df[df["is_anomaly"]]
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
# Tab 4: Detection accuracy (only meaningful with known ground truth, i.e. simulated mode)
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
    "Built on an Isolation Forest (per-ticker, SHAP-explained) for volume/price anomalies and a "
    "correlation-break engine for cross-stock coordinated moves. See README.md for architecture, "
    "limitations, and how to point this at real live data."
)
