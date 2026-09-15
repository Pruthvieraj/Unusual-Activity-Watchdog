"""
One-click PDF "surveillance incident report" for a single alert -- the kind
of document an exchange's market-surveillance desk would actually file
when escalating something for human review.

Built with reportlab (layout/text) + matplotlib (the embedded chart,
rendered to PNG bytes in memory -- no temp files, no kaleido dependency).
"""

from __future__ import annotations

import hashlib
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless rendering, no display server needed
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                  Table, TableStyle)

DARK = colors.HexColor("#0b0b0b")
MUTED = colors.HexColor("#52514e")
ACCENT = colors.HexColor("#2a78d6")
STATUS_COLORS = {"High": colors.HexColor("#d03b3b"), "Medium": colors.HexColor("#c98500"), "Low": colors.HexColor("#898781")}


def _report_id(alert_row: dict) -> str:
    raw = f"{alert_row.get('kind')}|{alert_row.get('tickers')}|{alert_row.get('first_seen')}"
    return "WD-" + hashlib.sha1(raw.encode()).hexdigest()[:10].upper()


def _render_ticker_chart(scored_by_ticker: dict, ticker: str, ts) -> bytes | None:
    if ticker not in scored_by_ticker:
        return None
    df = scored_by_ticker[ticker]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3.2), sharex=True, height_ratios=[2, 1])
    ax1.plot(df.index, df["price"], color="#2a78d6", linewidth=1.2)
    if ts in df.index:
        ax1.scatter([ts], [df.loc[ts, "price"]], color="#d03b3b", zorder=5, s=45, marker="o", facecolors="none", linewidths=1.6)
    ax1.set_ylabel("Price", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax2.bar(df.index, df["volume"], color="#199e70", width=0.003)
    ax2.set_ylabel("Volume", fontsize=8)
    ax2.tick_params(labelsize=7)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_incident_report(alert_row: dict, scored_by_ticker: dict, output_path: str) -> str:
    """Writes a PDF incident report for one alert to `output_path`."""
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("WDTitle", parent=styles["Title"], textColor=DARK, fontSize=18, spaceAfter=2)
    sub_style = ParagraphStyle("WDSub", parent=styles["Normal"], textColor=MUTED, fontSize=9, spaceAfter=14)
    h2_style = ParagraphStyle("WDH2", parent=styles["Heading2"], textColor=DARK, fontSize=12, spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle("WDBody", parent=styles["Normal"], fontSize=10, leading=14)
    footer_style = ParagraphStyle("WDFooter", parent=styles["Normal"], fontSize=7.5, textColor=MUTED)

    doc = SimpleDocTemplate(output_path, pagesize=LETTER, topMargin=0.7 * inch, bottomMargin=0.6 * inch,
                              leftMargin=0.75 * inch, rightMargin=0.75 * inch)
    story = []

    report_id = _report_id(alert_row)
    story.append(Paragraph("UNUSUAL ACTIVITY WATCHDOG", title_style))
    story.append(Paragraph("Automated Surveillance Incident Report", sub_style))

    severity = alert_row.get("severity", "Medium")
    confidence = alert_row.get("confidence", 0.0) or 0.0
    meta_table = Table([
        ["Report ID", report_id, "Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Severity", severity, "Confidence", f"{confidence:.0f} / 100"],
        ["Consensus", "Yes -- 2 independent AI models agree" if alert_row.get("consensus") else "No -- single-model flag", "", ""],
    ], colWidths=[1.0 * inch, 2.4 * inch, 1.0 * inch, 2.1 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 0), (1, 0), STATUS_COLORS.get(severity, MUTED)),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#e1e0d9")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Alert Summary", h2_style))
    story.append(Paragraph(f"<b>{alert_row.get('headline', '')}</b>", body_style))
    story.append(Paragraph(alert_row.get("detail", ""), body_style))

    window = f"{alert_row.get('first_seen')} → {alert_row.get('last_seen')}" if alert_row.get("first_seen") != alert_row.get("last_seen") else str(alert_row.get("first_seen"))
    story.append(Spacer(1, 6))
    facts = Table([
        ["Type", str(alert_row.get("kind", "")).replace("_", " ").title()],
        ["Ticker(s)", alert_row.get("tickers", "")],
        ["Time window", window],
        ["Bars persisted", str(alert_row.get("occurrences", 1))],
    ], colWidths=[1.3 * inch, 5.2 * inch])
    facts.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e1e0d9")),
    ]))
    story.append(facts)
    story.append(Spacer(1, 10))

    evidence = alert_row.get("evidence") or {}
    drivers = evidence.get("drivers")
    if drivers:
        story.append(Paragraph("Model Evidence (SHAP driver attribution)", h2_style))
        rows = [["Feature", "Contribution"]] + [[k, f"{v:+.3f}"] for k, v in drivers]
        t = Table(rows, colWidths=[3.0 * inch, 2.0 * inch])
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e1e0d9")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(t)
    elif "corr_delta" in evidence:
        story.append(Paragraph("Model Evidence (correlation-break engine)", h2_style))
        concentration_line = ""
        if "concentration_ratio" in evidence:
            kind_note = ("genuine small cluster" if alert_row.get("kind") == "correlated_group_move"
                         else "broad market-wide move, not a targeted cluster")
            concentration_line = (f"<br/>Concentration ratio: <b>{evidence.get('concentration_ratio', 0):.2f}x</b> "
                                   f"({kind_note})")
        story.append(Paragraph(
            f"Pairwise correlation delta vs. baseline: <b>{evidence.get('corr_delta', 0):+.3f}</b><br/>"
            f"Unusualness (z-score): <b>{evidence.get('delta_zscore', 0):.2f}</b>{concentration_line}", body_style))

    tickers = str(alert_row.get("tickers", "")).split(", ")
    ts = alert_row.get("first_seen")
    for tkr in tickers[:1]:
        chart_bytes = _render_ticker_chart(scored_by_ticker, tkr, ts)
        if chart_bytes:
            story.append(Spacer(1, 10))
            story.append(Paragraph("Price / Volume Context", h2_style))
            story.append(Image(io.BytesIO(chart_bytes), width=6.3 * inch, height=3.0 * inch))

    story.append(Spacer(1, 24))
    story.append(Paragraph(
        "Generated automatically by the Unusual Activity Watchdog AI system (Synapse 1.0, AI in FinTech track). "
        "This report is produced for investigative triage by a human analyst and does not itself constitute a "
        "finding of market misconduct.", footer_style))

    doc.build(story)
    return output_path
