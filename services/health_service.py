"""
System Health checks (Sprint 3): answers "is the detection stack actually
working right now", independent of whether any alerts happen to be firing.

An empty Alert Feed is ambiguous on its own -- it could mean a quiet
market, or a silently broken detector. This module runs a fixed battery of
cheap, self-contained checks against the SAME pipeline result the Command
Center just computed (plus a couple of pure self-tests that don't need
live data at all) and reports each as OK / WARN / FAIL with a one-sentence
reason, so a demo presenter -- or an actual operator -- can tell the
difference at a glance instead of inferring it from alert counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import CONFIG
from services import alert_service

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_ICON = {OK: "✅", WARN: "⚠️", FAIL: "\U0001F534"}


@dataclass
class HealthCheck:
    name: str
    status: str
    detail: str

    @property
    def badge(self) -> str:
        return f"{_ICON.get(self.status, '')} {self.status}"


def check_data_source(pipeline_result: dict, refresh_interval: int, last_live_success_ts=None) -> HealthCheck:
    mode = pipeline_result.get("mode", "")
    error = pipeline_result.get("error")
    # `error` is the reliable signal here, not `mode`: resolve_market_data
    # (data_source.py) REWRITES "mode" to the Simulated label on a live-fetch
    # failure (so the rest of the pipeline runs on real fallback data under
    # its own honest name) -- but that means checking `mode.startswith("Live")`
    # to decide whether a live fetch just failed is checking the wrong field
    # AFTER a fallback: it's always False by the time we get here, so a
    # genuinely failed live fetch was silently reported as a healthy plain
    # "Simulated demo data" OK, indistinguishable from someone deliberately
    # choosing Simulated mode. `error` is set if and only if a live fetch was
    # attempted and failed this run, regardless of what `mode` got relabeled
    # to -- so it's checked first, before any mode-based branching.
    if error is not None:
        return HealthCheck("Data source", WARN, f"Live feed failed this run and fell back to simulated data: {pipeline_result.get('note', '')}")
    if mode.startswith("Live"):
        last_success = last_live_success_ts
        if last_success is not None:
            age_s = (pd.Timestamp.now() - last_success).total_seconds()
            stale_cutoff = CONFIG.stale_data_multiplier * refresh_interval
            if age_s > stale_cutoff:
                return HealthCheck(
                    "Data source", WARN,
                    f"Live mode, but the last successful refresh was {age_s:.0f}s ago "
                    f"(stale cutoff is {stale_cutoff:.0f}s) -- data may be delayed.",
                )
        return HealthCheck("Data source", OK, "Live yfinance feed responding normally this run.")
    return HealthCheck("Data source", OK, "Simulated demo data -- deterministic and always available, independent of market hours or network access.")


def check_autorefresh(mode_choice: str, live_autorefresh: bool, refresh_interval: int) -> HealthCheck:
    if not mode_choice.startswith("Live"):
        return HealthCheck("Continuous monitoring", OK, "Not applicable in Simulated mode (the pipeline recomputes on page load / manual refresh).")
    if not live_autorefresh:
        return HealthCheck("Continuous monitoring", WARN, "Auto-refresh is turned off in the sidebar -- the header and Alert Feed only update on a full page rerun.")
    return HealthCheck("Continuous monitoring", OK, f"Header + Alert Feed silently refresh every {refresh_interval}s via native st.fragment(run_every=...).")


def check_detectors(scored_by_ticker: dict) -> HealthCheck:
    if not scored_by_ticker:
        return HealthCheck("Detection ensemble", FAIL, "The pipeline returned no scored tickers at all.")
    missing = [f"{tkr}:{col}" for tkr, df in scored_by_ticker.items()
               for col in ("is_anomaly", "anomaly_score") if col not in df.columns]
    if missing:
        return HealthCheck("Detection ensemble", FAIL, f"Isolation Forest output is missing expected columns: {', '.join(missing[:5])}")
    ae_missing = [t for t, df in scored_by_ticker.items() if "ae_is_anomaly" not in df.columns]
    if ae_missing:
        return HealthCheck("Detection ensemble", WARN, f"Autoencoder columns missing for {len(ae_missing)} of {len(scored_by_ticker)} ticker(s) -- running on Isolation Forest alone for those.")
    return HealthCheck("Detection ensemble", OK, f"Isolation Forest + autoencoder ensemble scored all {len(scored_by_ticker)} watched ticker(s).")


def check_correlation_engine(breaks: Optional[pd.DataFrame]) -> HealthCheck:
    if breaks is None or breaks.empty:
        return HealthCheck("Correlation engine", WARN, "No correlation-break output for this window (too little history yet, or a very small watchlist).")
    required = {"is_break", "delta_zscore", "move_kind"}
    missing = required - set(breaks.columns)
    if missing:
        return HealthCheck("Correlation engine", FAIL, f"Missing expected columns: {', '.join(sorted(missing))}")
    return HealthCheck("Correlation engine", OK, f"{len(breaks)} bar(s) scored; {int(breaks['is_break'].sum())} break(s) flagged this run.")


def check_news_relevance() -> HealthCheck:
    try:
        from news_relevance import score_headline_relevance
        r_relevant = score_headline_relevance("AAPL announces merger and acquisition of a chip startup", "AAPL")
        r_irrelevant = score_headline_relevance("AAPL named among top workplaces for remote employees", "AAPL")
        if r_relevant > r_irrelevant:
            return HealthCheck("News relevance scorer", OK, f"Self-test passed (relevant={r_relevant:.2f} > irrelevant={r_irrelevant:.2f}).")
        return HealthCheck("News relevance scorer", WARN, "Self-test did not separate a relevant headline from an irrelevant one as expected.")
    except Exception as exc:  # noqa: BLE001 -- this IS the health check, any failure is the signal
        return HealthCheck("News relevance scorer", FAIL, f"Self-test raised an exception: {exc}")


def check_persistence() -> HealthCheck:
    try:
        counts = alert_service.status_counts()
        total = sum(counts.values())
        breakdown = ", ".join(f"{k}: {v}" for k, v in counts.items())
        return HealthCheck(
            "Alert persistence (SQLite)", OK,
            f"{total} alert(s) stored ({breakdown}). Note: on a host without a persistent disk (e.g. Render's "
            "free tier) this survives the running instance only -- it is wiped on redeploy or cold restart.",
        )
    except Exception as exc:  # noqa: BLE001
        return HealthCheck("Alert persistence (SQLite)", FAIL, f"Could not reach the alert store: {exc}")


def check_alert_volume(alerts: Optional[pd.DataFrame]) -> HealthCheck:
    if alerts is None or alerts.empty:
        return HealthCheck("Alert volume sanity", OK, "No alerts this run -- expected sometimes in a quiet market; treat as a problem only if it persists alongside other WARN/FAIL checks above.")
    high = int((alerts["severity"] == "High").sum())
    total = len(alerts)
    if total > 200:
        return HealthCheck("Alert volume sanity", WARN, f"{total} alert(s) in a single run is unusually high -- check for a misconfigured threshold or a pipeline runaway.")
    return HealthCheck("Alert volume sanity", OK, f"{total} alert(s) this run ({high} High severity) -- within the expected range.")


def run_all_checks(
    pipeline_result: dict,
    mode_choice: str,
    live_autorefresh: bool,
    refresh_interval: int,
    last_live_success_ts=None,
) -> list[HealthCheck]:
    return [
        check_data_source(pipeline_result, refresh_interval, last_live_success_ts),
        check_autorefresh(mode_choice, live_autorefresh, refresh_interval),
        check_detectors(pipeline_result.get("scored", {})),
        check_correlation_engine(pipeline_result.get("breaks")),
        check_news_relevance(),
        check_persistence(),
        check_alert_volume(pipeline_result.get("alerts")),
    ]
