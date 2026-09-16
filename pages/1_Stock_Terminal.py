"""
Stock Investigation Terminal -- Sprint 4 page.

A per-ticker deep dive: price/volume/return-zscore detail, the momentum
(EWMA crossover) signal, a news-relevance timeline, this ticker's own
correlation-cluster context, and its full alert/investigation history --
everything an analyst would want after the Command Center's feed points
them at one specific name, in one place instead of stitched together from
several tabs.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import CONFIG
from detection.momentum import compute_momentum_signal
from news_relevance import score_headline_relevance
from services import alert_service, news_service
from services.pipeline_service import resolve_and_score
from utils import design_tokens as tokens
from utils import formatting

st.set_page_config(page_title="Stock Terminal -- Watchdog", layout="wide", page_icon="\U0001F50D")
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

st.title("\U0001F50D Stock Investigation Terminal")
st.caption("A single ticker's full picture: price/volume detail, momentum, news relevance, correlation context, and its alert history.")

# --------------------------------------------------------------------------
# Sidebar -- same data-source controls as the Command Center, so this page
# is self-sufficient (Streamlit pages are independent scripts).
# --------------------------------------------------------------------------
st.sidebar.title("\U0001F6A8 Watchdog Controls")
mode_choice = st.sidebar.radio("Data source", ["Simulated demo (recommended)", "Live / historical (yfinance)"])
watchlist = st.sidebar.multiselect(
    "Watchlist", options=sorted(set(CONFIG.watchlist) | {"TSLA", "META", "NFLX", "BAC", "WMT"}),
    default=CONFIG.watchlist,
)
if not watchlist:
    watchlist = CONFIG.watchlist
seed = st.sidebar.number_input("Scenario seed", min_value=1, max_value=9999, value=12, step=1) \
    if mode_choice.startswith("Simulated") else None

with st.spinner("Running pipeline..."):
    result = resolve_and_score(mode_choice, watchlist, seed)
if result["error"]:
    st.warning(result["note"])

scored = result["scored"]
breaks = result["breaks"]
market = result["market"]

ticker_options = sorted(scored.keys())
# If the Command Center's correlation-network click-through set a ticker,
# jump straight to it here -- st.session_state is shared across pages
# within one session, so a click there carries through to this page.
preselected = st.session_state.get("_selected_ticker")
default_index = ticker_options.index(preselected) if preselected in ticker_options else 0
tkr = st.selectbox("Ticker", ticker_options, index=default_index)
df = scored[tkr]

# --------------------------------------------------------------------------
# Price / volume / return z-score detail
# --------------------------------------------------------------------------
st.markdown("#### Price, volume & return z-score")
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.05,
                     subplot_titles=("Price", "Volume", "Return z-score"))
fig.add_trace(go.Scatter(x=df.index, y=df["price"], name="Price", line=dict(color=tokens.CATEGORICAL[0], width=2)), row=1, col=1)
anomalies = df[df["any_model_flagged"]] if "any_model_flagged" in df.columns else df[df["is_anomaly"]]
fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["price"], mode="markers", name="Flagged",
                          marker=dict(color=tokens.CATEGORICAL[7], size=9, symbol="circle-open", line=dict(width=2))), row=1, col=1)
fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume", marker_color=tokens.CATEGORICAL[2]), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["return_zscore"], name="Return z", line=dict(color=tokens.CATEGORICAL[3], width=2)), row=3, col=1)
fig.add_hline(y=CONFIG.price_jump_sigma_alert, line_dash="dot", line_color=tokens.STATUS["High"], row=3, col=1)
fig.add_hline(y=-CONFIG.price_jump_sigma_alert, line_dash="dot", line_color=tokens.STATUS["High"], row=3, col=1)
fig = tokens.plotly_base_layout(fig, height=560)
fig.update_layout(showlegend=False, title=f"{tkr}: full detail view")
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Momentum (EWMA crossover)
# --------------------------------------------------------------------------
st.markdown("#### Momentum -- short/long EWMA crossover")
st.caption(
    "A short-window return EWMA crossing a long-window one by an amount that is itself a z-score outlier "
    "against this stock's own recent history -- trend divergence, not a single-bar price/volume spike "
    "(see detection/momentum.py). Flagged points feed the Alert Feed's `momentum_shift` alerts."
)
mdf = compute_momentum_signal(df["return"])
mfig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06,
                      subplot_titles=("Short vs. long EWMA", "Crossover z-score"))
mfig.add_trace(go.Scatter(x=mdf.index, y=mdf["ewma_short"], name=f"Short EWMA ({CONFIG.momentum_short_span})",
                           line=dict(color=tokens.CATEGORICAL[0], width=2)), row=1, col=1)
mfig.add_trace(go.Scatter(x=mdf.index, y=mdf["ewma_long"], name=f"Long EWMA ({CONFIG.momentum_long_span})",
                           line=dict(color=tokens.CATEGORICAL[4], width=2, dash="dot")), row=1, col=1)
mfig.add_trace(go.Scatter(x=mdf.index, y=mdf["crossover_zscore"], name="Crossover z-score",
                           line=dict(color=tokens.CATEGORICAL[3], width=2)), row=2, col=1)
shifted = mdf[mdf["is_shift"]]
mfig.add_trace(go.Scatter(x=shifted.index, y=shifted["crossover_zscore"], mode="markers", name="Shift flagged",
                           marker=dict(color=tokens.CATEGORICAL[7], size=9, symbol="circle-open", line=dict(width=2))),
               row=2, col=1)
mfig.add_hline(y=CONFIG.momentum_shift_zscore_alert, line_dash="dot", line_color=tokens.STATUS["High"], row=2, col=1)
mfig.add_hline(y=-CONFIG.momentum_shift_zscore_alert, line_dash="dot", line_color=tokens.STATUS["High"], row=2, col=1)
mfig = tokens.plotly_base_layout(mfig, height=420)
mfig.update_layout(title=f"{tkr}: momentum")
st.plotly_chart(mfig, use_container_width=True)

# --------------------------------------------------------------------------
# News relevance timeline
# --------------------------------------------------------------------------
st.markdown("#### News relevance timeline")
news_rows = []
if "news" in market and market["news"] is not None and not market["news"].empty:
    ticker_news = market["news"][market["news"]["ticker"] == tkr].sort_values("timestamp", ascending=False).head(15)
    for _, r in ticker_news.iterrows():
        news_rows.append({"timestamp": r["timestamp"], "headline": r["headline"],
                           "relevance": score_headline_relevance(r["headline"], tkr)})
elif mode_choice.startswith("Live"):
    try:
        from data.live_feed import fetch_recent_news
        for item in fetch_recent_news(tkr)[:15]:
            ts = item.get("published_at")
            news_rows.append({"timestamp": ts, "headline": item.get("title"),
                               "relevance": score_headline_relevance(item.get("title"), tkr)})
    except Exception:
        pass

if not news_rows:
    st.info("No recent headlines found for this ticker in the current lookback window.")
else:
    news_df = pd.DataFrame(news_rows)
    news_df["state"] = news_df["relevance"].map(
        lambda r: news_service.NEWS_EXPLAINS if r >= CONFIG.news_relevance_explained_threshold else news_service.NEWS_MISMATCH
    )
    news_df["badge"] = news_df["state"].map(lambda s: f"{tokens.NEWS_STATE_ICON.get(s, '')} {s}")
    news_df["relevance_pct"] = news_df["relevance"].map(lambda r: f"{r:.0%}")
    st.dataframe(
        news_df[["timestamp", "headline", "relevance_pct", "badge"]].rename(columns={
            "timestamp": "Time", "headline": "Headline", "relevance_pct": "Relevance", "badge": "Classification",
        }),
        use_container_width=True, hide_index=True, height=280,
    )

# --------------------------------------------------------------------------
# Correlation context
# --------------------------------------------------------------------------
st.markdown("#### Correlation context")
if breaks is None or breaks.empty:
    st.caption("No correlation-break history available yet for this window.")
else:
    latest = breaks.iloc[-1]
    in_cluster = latest["is_break"] and tkr in (latest.get("cluster") or [])
    if in_cluster:
        move_kind = latest.get("move_kind") or "correlated_group_move"
        st.success(
            f"As of the latest bar, {tkr} is part of a flagged correlation event "
            f"({KIND_LABELS.get(move_kind, move_kind)}, delta z-score {latest['delta_zscore']:.2f}, "
            f"concentration ratio {latest.get('concentration_ratio', 0):.2f}x)."
        )
    else:
        st.caption(f"{tkr} is not currently part of a flagged correlation cluster.")

# --------------------------------------------------------------------------
# This ticker's alert / investigation history
# --------------------------------------------------------------------------
st.markdown("#### Alert & investigation history for this ticker")
ticker_alerts = alert_service.fetch_alerts(ticker=tkr)
if ticker_alerts.empty:
    st.caption("No persisted alerts for this ticker yet.")
else:
    disp = ticker_alerts.copy()
    disp["window"] = disp.apply(lambda r: formatting.format_window(r["first_seen"], r["last_seen"]), axis=1)
    disp["type"] = disp["kind"].map(lambda k: KIND_LABELS.get(k, k))
    disp["status_badge"] = disp["status"].map(lambda s: f"{tokens.INVESTIGATION_ICON.get(s, '')} {s}")
    disp["severity_badge"] = disp["severity"].map(lambda s: f"{tokens.STATUS_ICON.get(s, '')} {s}")
    disp = disp.sort_values("last_seen", ascending=False)
    st.dataframe(
        disp[["window", "severity_badge", "type", "headline", "confidence", "status_badge"]].rename(columns={
            "window": "When", "severity_badge": "Severity", "type": "Type", "headline": "Alert",
            "confidence": "Confidence", "status_badge": "Investigation",
        }),
        use_container_width=True, hide_index=True, height=260,
    )
