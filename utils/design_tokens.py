"""
Single source of truth for color/spacing/typography tokens (Sprint 4).

Before this, app.py defined its dark-mode palette (SURFACE, PAGE, INK_*,
GRIDLINE, CATEGORICAL, STATUS, STATUS_ICON, ...) as module-level constants
that a new page under pages/ would have had to copy-paste to look
consistent -- the classic design-token-consolidation gap. Every page
(app.py included) now imports its palette from here instead.

app.py keeps its own module-level names (SURFACE = tokens.SURFACE, etc.)
for the rest of that already-large file's readability; new pages should
just `from utils import design_tokens as tokens` and use `tokens.X`
directly rather than re-aliasing.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Palette (validated dark-mode set). Categorical slots keep a fixed order
# across every page; status colors are reserved for severity and never
# reused as series colors.
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

INVESTIGATION_ICON = {
    "NEW": "\U0001F195",
    "INVESTIGATING": "\U0001F50E",
    "RESOLVED": "✅",
    "FALSE POSITIVE": "\U0001F6AB",
}

NEWS_STATE_ICON = {
    "NEWS EXPLAINS MOVEMENT": "\U0001F4F0",
    "POSSIBLE NEWS-MOVEMENT MISMATCH": "❓",
    "NO RELEVANT NEWS DETECTED": "\U0001F507",
}

HEALTH_ICON = {"OK": "✅", "WARN": "⚠️", "FAIL": "\U0001F534"}

# Spacing / sizing tokens for the handful of places that hardcode a pixel
# value (chart heights, card padding) -- named so a future redesign changes
# one number instead of grepping every page for "height=".
CHART_HEIGHT_SM = 220
CHART_HEIGHT_MD = 360
CHART_HEIGHT_LG = 420
CARD_BORDER_RADIUS_PX = 10

PAGE_CSS = f"""
<style>
  .stApp {{ background-color: {PAGE}; }}
  section[data-testid="stSidebar"] {{ background-color: {SURFACE}; }}
  div[data-testid="stMetric"] {{
      background-color: {SURFACE}; border: 1px solid {GRIDLINE};
      border-radius: {CARD_BORDER_RADIUS_PX}px; padding: 12px 16px;
  }}
  .watchdog-badge {{
      display: inline-block; padding: 2px 10px; border-radius: 999px;
      font-size: 0.78rem; font-weight: 600; margin-right: 6px;
  }}
</style>
"""


def plotly_base_layout(fig, height: int = CHART_HEIGHT_LG):
    """Applies this project's one dark Plotly theme -- every page should
    route its figures through this instead of setting template/colors
    inline, so a future palette change is a one-file edit."""
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


def badge_html(text: str, color: str) -> str:
    """One `<span class="watchdog-badge">` -- the same pill styling
    app.py's inline HTML badges (severity/confidence/consensus) already
    use, extracted so a new page doesn't hand-roll the CSS again."""
    return f"<span class='watchdog-badge' style='background:{color}22;color:{color};'>{text}</span>"
