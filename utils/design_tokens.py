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
# Palette -- "surveillance-desk" dark theme: a deep blue-black base (not a
# flat gray/black) with a single cyan accent, so the eye has one clear
# "system" color instead of the old scheme's low-contrast olive grays.
# Categorical slots keep a fixed order across every page; status colors are
# reserved for severity and never reused as series colors.
# --------------------------------------------------------------------------
PAGE = "#0a0e14"
SURFACE = "#111827"
SURFACE_RAISED = "#161f33"
ACCENT = "#22d3ee"
ACCENT_SOFT = "#38bdf8"
INK_PRIMARY = "#f1f5f9"
INK_SECONDARY = "#a8b3c7"
INK_MUTED = "#6b7688"
GRIDLINE = "#232f45"
BASELINE = "#2d3b55"
BORDER = "#1f2b40"

CATEGORICAL = ["#22d3ee", "#f97316", "#34d399", "#eab308", "#f472b6", "#a78bfa", "#60a5fa", "#fb7185"]

STATUS = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#7c8698"}
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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  :root {{
      --wd-page: {PAGE}; --wd-surface: {SURFACE}; --wd-surface-raised: {SURFACE_RAISED};
      --wd-accent: {ACCENT}; --wd-ink: {INK_PRIMARY}; --wd-ink-2: {INK_SECONDARY};
      --wd-muted: {INK_MUTED}; --wd-grid: {GRIDLINE}; --wd-border: {BORDER};
  }}

  html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
  code, pre, .stCode, [data-testid="stMetricValue"] {{ font-family: 'JetBrains Mono', 'SFMono-Regular', monospace !important; }}

  /* ---------------- Base canvas ---------------- */
  .stApp {{
      background:
          radial-gradient(1100px 520px at 8% -8%, rgba(34,211,238,0.07), transparent 60%),
          radial-gradient(900px 500px at 100% 0%, rgba(167,139,250,0.05), transparent 55%),
          {PAGE};
      color: var(--wd-ink);
  }}
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}

  /* ---------------- Typography ---------------- */
  h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.01em; color: var(--wd-ink); }}
  h1 {{ font-weight: 800 !important; font-size: 2.15rem !important; }}
  h2, h3 {{ font-weight: 700 !important; }}
  h4, h5 {{ font-weight: 600 !important; color: var(--wd-ink-2) !important; }}
  p, li, span, label {{ color: var(--wd-ink-2); }}
  [data-testid="stCaptionContainer"], .stCaption {{ color: var(--wd-muted) !important; font-size: 0.85rem !important; }}
  a {{ color: var(--wd-accent) !important; }}

  /* Top title gets a subtle underline rule + gradient wordmark feel */
  h1:first-of-type {{
      background: linear-gradient(90deg, {INK_PRIMARY} 0%, {ACCENT_SOFT} 65%, {ACCENT} 100%);
      -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
      padding-bottom: 0.35rem; border-bottom: 1px solid var(--wd-border); margin-bottom: 0.9rem !important;
  }}

  /* ---------------- Sidebar ---------------- */
  section[data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {SURFACE_RAISED} 0%, {SURFACE} 100%);
      border-right: 1px solid var(--wd-border);
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.4rem; }}
  section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
      font-size: 1rem !important; text-transform: uppercase; letter-spacing: 0.06em; color: var(--wd-accent) !important;
      -webkit-text-fill-color: var(--wd-accent) !important; background: none !important; border: none !important;
  }}

  /* ---------------- Metrics / KPI cards ---------------- */
  div[data-testid="stMetric"] {{
      background: linear-gradient(160deg, {SURFACE_RAISED} 0%, {SURFACE} 100%);
      border: 1px solid var(--wd-border);
      border-radius: {CARD_BORDER_RADIUS_PX + 4}px; padding: 16px 18px;
      box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 20px -12px rgba(0,0,0,0.6);
      transition: border-color 0.15s ease, transform 0.15s ease;
  }}
  div[data-testid="stMetric"]:hover {{ border-color: var(--wd-accent); transform: translateY(-1px); }}
  [data-testid="stMetricLabel"] {{
      color: var(--wd-muted) !important; font-size: 0.72rem !important; font-weight: 600 !important;
      text-transform: uppercase; letter-spacing: 0.06em;
  }}
  [data-testid="stMetricValue"] {{ color: var(--wd-ink) !important; font-weight: 700 !important; font-size: 1.7rem !important; }}
  [data-testid="stMetricDelta"] {{ font-weight: 600 !important; }}

  /* ---------------- Tabs ---------------- */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 4px; border-bottom: 1px solid var(--wd-border); background: transparent;
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent; border-radius: 8px 8px 0 0; color: var(--wd-muted);
      font-weight: 600; font-size: 0.92rem; padding: 10px 18px; border: none;
  }}
  .stTabs [aria-selected="true"] {{
      color: var(--wd-accent) !important; background: rgba(34,211,238,0.08);
      border-bottom: 2px solid var(--wd-accent);
  }}

  /* ---------------- Buttons ---------------- */
  .stButton > button, .stDownloadButton > button {{
      background: linear-gradient(180deg, {SURFACE_RAISED}, {SURFACE});
      color: var(--wd-ink); border: 1px solid var(--wd-border); border-radius: 8px;
      font-weight: 600; padding: 0.5rem 1.1rem; transition: all 0.15s ease;
  }}
  .stButton > button:hover, .stDownloadButton > button:hover {{
      border-color: var(--wd-accent); color: var(--wd-accent);
      box-shadow: 0 0 0 1px {ACCENT}33; transform: translateY(-1px);
  }}
  .stButton > button[kind="primary"] {{
      background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_SOFT} 100%);
      color: #06212a; border: none; font-weight: 700;
  }}
  .stButton > button[kind="primary"]:hover {{ filter: brightness(1.08); box-shadow: 0 0 16px {ACCENT}55; }}

  /* ---------------- Inputs / selects / sliders ---------------- */
  .stSelectbox [data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {{
      background-color: {SURFACE_RAISED} !important; border-color: var(--wd-border) !important;
      border-radius: 8px !important; color: var(--wd-ink) !important;
  }}
  .stSlider [data-baseweb="slider"] div[role="slider"] {{ background-color: var(--wd-accent) !important; }}
  .stRadio label, .stCheckbox label {{ color: var(--wd-ink-2) !important; }}

  /* ---------------- DataFrames / tables ---------------- */
  [data-testid="stDataFrame"], [data-testid="stTable"] {{
      border: 1px solid var(--wd-border) !important; border-radius: 10px !important; overflow: hidden;
  }}
  [data-testid="stDataFrame"] div {{ font-family: 'JetBrains Mono', monospace; font-size: 0.86rem; }}

  /* ---------------- Expanders ---------------- */
  details {{
      background: {SURFACE}; border: 1px solid var(--wd-border) !important; border-radius: 10px !important;
      overflow: hidden;
  }}
  summary {{ color: var(--wd-ink) !important; font-weight: 600; }}

  /* ---------------- Alert / status boxes ---------------- */
  div[data-testid="stAlert"] {{ border-radius: 10px; border: 1px solid var(--wd-border); }}

  /* ---------------- Plotly chart frames ---------------- */
  .js-plotly-plot {{ border-radius: 12px; overflow: hidden; }}

  /* ---------------- Divider ---------------- */
  hr {{ border-color: var(--wd-border) !important; }}

  /* ---------------- Badges ---------------- */
  .watchdog-badge {{
      display: inline-flex; align-items: center; gap: 4px; padding: 3px 11px; border-radius: 999px;
      font-size: 0.76rem; font-weight: 700; margin-right: 6px; letter-spacing: 0.01em;
      border: 1px solid currentColor;
  }}

  /* ---------------- Scrollbar ---------------- */
  ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
  ::-webkit-scrollbar-track {{ background: {PAGE}; }}
  ::-webkit-scrollbar-thumb {{ background: {SURFACE_RAISED}; border-radius: 6px; border: 2px solid {PAGE}; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--wd-accent); }}

  /* Hide default Streamlit chrome for a cleaner, product-like feel */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
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
        font=dict(color=INK_SECONDARY, size=13, family="Inter, sans-serif"),
        title_font=dict(color=INK_PRIMARY, size=15, family="Inter, sans-serif"),
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
