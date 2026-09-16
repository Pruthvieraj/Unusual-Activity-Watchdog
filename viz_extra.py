"""
The three "extra" visualizations: a circular correlation-network diagram,
a 3D anomaly-feature-space landscape, and the browser voice-alert snippet.
Split out from app.py to keep the main dashboard script from turning into
one enormous file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SURFACE = "#1a1a19"
INK_SECONDARY = "#c3c2b7"
POS_EDGE = "#e66767"
NEG_EDGE = "#3987e5"
NODE_DEFAULT = "#3987e5"
NODE_HIGHLIGHT = "#e66767"


def build_correlation_network(returns_window: pd.DataFrame, highlight_cluster: list[str] | None = None) -> go.Figure:
    """A circular node-link diagram: one node per ticker, edges drawn only
    for pairs with meaningful correlation (weak pairs are hidden rather
    than drawn faint, for legibility), width/opacity scaled by |correlation|,
    color by sign. Tickers in `highlight_cluster` are drawn in the alert
    status color so the flagged group visually pops out of the network.
    """
    highlight_cluster = set(highlight_cluster or [])
    tickers = list(returns_window.columns)
    n = len(tickers)
    corr = returns_window.corr()

    angles = {tkr: 2 * np.pi * i / n - np.pi / 2 for i, tkr in enumerate(tickers)}
    pos = {tkr: (np.cos(a), np.sin(a)) for tkr, a in angles.items()}

    edge_traces = []
    for i in range(n):
        for j in range(i + 1, n):
            c = corr.iloc[i, j]
            if abs(c) < 0.15:
                continue
            x0, y0 = pos[tickers[i]]
            x1, y1 = pos[tickers[j]]
            # Pull each endpoint in slightly from the node center. Without
            # this, an edge's endpoint coordinate is EXACTLY coincident
            # with the node marker's own coordinate, and Plotly's
            # closest-point hit-testing (hover AND on_select click) can
            # resolve a click squarely on a node to the edge trace instead
            # -- the two points tie at distance 0, so hover ends up on an
            # arbitrary trace ordering instead of the node "on top" you'd
            # visually expect. A small inset keeps the edge visually
            # touching the node while leaving the node's own coordinate
            # unambiguous.
            inset = 0.08
            ex0, ey0 = x0 + (x1 - x0) * inset, y0 + (y1 - y0) * inset
            ex1, ey1 = x1 + (x0 - x1) * inset, y1 + (y0 - y1) * inset
            color = POS_EDGE if c > 0 else NEG_EDGE
            edge_traces.append(go.Scatter(
                x=[ex0, ex1, None], y=[ey0, ey1, None], mode="lines",
                line=dict(color=color, width=1 + abs(c) * 7),
                opacity=min(1.0, abs(c) + 0.25), hoverinfo="text",
                text=f"{tickers[i]}-{tickers[j]}: {c:+.2f}", showlegend=False,
            ))

    node_colors = [NODE_HIGHLIGHT if t in highlight_cluster else NODE_DEFAULT for t in tickers]
    node_sizes = [40 if t in highlight_cluster else 30 for t in tickers]
    node_trace = go.Scatter(
        x=[pos[t][0] for t in tickers], y=[pos[t][1] for t in tickers],
        mode="markers+text", text=tickers, textposition="top center",
        textfont=dict(color=INK_SECONDARY, size=13),
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color=SURFACE)),
        hoverinfo="text", showlegend=False,
        # `customdata` (not curve/point index) is what the click-through
        # panel reads from st.plotly_chart(..., on_select=...) -- carries
        # the ticker name straight through regardless of how many edge
        # traces precede this node trace in the figure.
        customdata=tickers,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_xaxes(visible=False, range=[-1.4, 1.4])
    fig.update_yaxes(visible=False, range=[-1.4, 1.4], scaleanchor="x", scaleratio=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=10, r=10, t=30, b=10), height=460,
        title="Live correlation network (red = flagged cluster, edge color = sign, width = strength)",
    )
    return fig


def build_3d_landscape(scored_df: pd.DataFrame, ticker: str) -> go.Figure:
    """3D scatter of every bar in feature space (return z / volume z /
    volatility), colored by the ensemble anomaly score, with dual-model
    consensus points drawn as larger diamonds -- flagged bars visibly float
    outside the dense "normal" cluster at the origin.
    """
    has_ensemble = "ensemble_score" in scored_df.columns
    color_col = "ensemble_score" if has_ensemble else "anomaly_score"
    consensus = scored_df["consensus"] if "consensus" in scored_df.columns else scored_df["is_anomaly"]
    flagged = scored_df["any_model_flagged"] if "any_model_flagged" in scored_df.columns else scored_df["is_anomaly"]

    sizes = np.where(consensus, 12, np.where(flagged, 7, 3.5))
    # Scatter3d doesn't support a per-point marker outline width, so
    # consensus points are distinguished by symbol (diamond) instead of a
    # ring -- still visually distinct, just via shape rather than outline.
    symbols = np.where(consensus, "diamond", "circle")

    fig = go.Figure(go.Scatter3d(
        x=scored_df["return_zscore"], y=scored_df["volume_zscore"], z=scored_df["volatility"],
        mode="markers",
        marker=dict(
            size=sizes, color=scored_df[color_col], symbol=symbols,
            colorscale=[[0, "#1c5cab"], [0.5, SURFACE], [1, "#e66767"]],
            colorbar=dict(title="Anomaly<br>score", tickfont=dict(color=INK_SECONDARY)),
            opacity=0.85,
        ),
        text=[t.strftime("%b %d %H:%M") for t in scored_df.index],
        hovertemplate="%{text}<br>return z=%{x:.2f} vol z=%{y:.2f} volat=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=0, r=0, t=40, b=0), height=520,
        title=f"{ticker}: anomaly feature-space landscape (diamond = dual-model consensus)",
        scene=dict(
            xaxis=dict(title="Return z-score", backgroundcolor=SURFACE, gridcolor="#2c2c2a", color=INK_SECONDARY),
            yaxis=dict(title="Volume z-score", backgroundcolor=SURFACE, gridcolor="#2c2c2a", color=INK_SECONDARY),
            zaxis=dict(title="Volatility", backgroundcolor=SURFACE, gridcolor="#2c2c2a", color=INK_SECONDARY),
        ),
    )
    return fig


def speak_snippet(text: str) -> str:
    """HTML/JS snippet using the browser's built-in SpeechSynthesis API to
    announce an alert out loud -- no external TTS service or API key
    needed. Pass the returned string to st.components.v1.html(..., height=0).
    """
    safe = text.replace('"', "'").replace("\n", " ")
    return f"""
    <script>
    try {{
        var msg = new SpeechSynthesisUtterance("{safe}");
        msg.rate = 1.05;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
    }} catch (e) {{ console.log("speech synthesis unavailable", e); }}
    </script>
    """
