"""
Sanity checks for the detection pipeline, run against the synthetic
generator's known ground truth. Not a rigorous statistical evaluation --
just enough to catch a broken pipeline before a demo.

Run with:  python3 -m pytest tests/ -v   (or just: python3 tests/test_pipeline.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import numpy as np

from alert_engine import _confidence_score, build_synthetic_news_lookup, generate_all_alerts, generate_correlation_alerts
from config import CONFIG
from correlation_watch import detect_correlation_breaks
from data.synthetic import generate_market_tape
from data_source import FALLBACK_SEED, SIMULATED_MODE_LABEL, resolve_market_data
from features import build_all_features
from models.ensemble import build_ensemble
from news_relevance import score_headline_relevance


def run_scenario(seed: int):
    market = generate_market_tape(seed=seed)
    features = build_all_features(market)
    scored = build_ensemble(features)
    breaks = detect_correlation_breaks(market["returns"])
    news_lookup = build_synthetic_news_lookup(market["news"], CONFIG.news_lookback_hours)
    alerts = generate_all_alerts(scored, breaks, news_lookup, merge_bar_minutes=CONFIG.simulate_bar_freq_minutes)
    return market, alerts


def check_ground_truth_recall(seed: int) -> float:
    market, alerts = run_scenario(seed)
    window = pd.Timedelta(minutes=90)
    hits = 0
    for g in market["ground_truth"]:
        ts = g["timestamp"]
        nearby = alerts[(alerts["first_seen"] <= ts + window) & (alerts["last_seen"] >= ts - window)]
        if g["kind"] == "correlated_group_move":
            match = nearby[nearby["kind"] == "correlated_group_move"]
            caught = any(len(set(g["tickers"]) & set(t.split(", "))) >= 2 for t in match["tickers"])
        else:
            match = nearby[nearby["tickers"].isin(g["tickers"])]
            caught = not match.empty
        hits += int(caught)
    return hits / len(market["ground_truth"])


def test_recall_across_seeds():
    recalls = [check_ground_truth_recall(seed) for seed in range(1, 11)]
    avg_recall = sum(recalls) / len(recalls)
    print(f"Average recall on injected anomalies across 10 seeds: {avg_recall:.0%}")
    assert avg_recall >= 0.6, "Detector is missing most injected anomalies -- check thresholds in config.py"


def test_pipeline_runs_without_error():
    market, alerts = run_scenario(seed=42)
    assert not market["prices"].empty
    assert isinstance(alerts, pd.DataFrame)


def test_no_alerts_on_pure_noise():
    """A watchlist with no injected anomalies should raise very few alerts
    (not zero -- Isolation Forest's contamination parameter guarantees some
    flags by construction -- but nowhere near the count seen with injected
    events)."""
    market = generate_market_tape(seed=99, n_anomalies=0)
    features = build_all_features(market)
    scored = build_ensemble(features)
    breaks = detect_correlation_breaks(market["returns"])
    alerts = generate_all_alerts(scored, breaks, merge_bar_minutes=CONFIG.simulate_bar_freq_minutes)
    high_severity = alerts[alerts["severity"] == "High"]
    print(f"High-severity alerts with zero injected anomalies: {len(high_severity)}")
    assert len(high_severity) < 15


def test_market_wide_move_not_flagged_as_cluster():
    """Judge-review regression test (Phase 8 / Phase 17 item 1): a uniform
    shock across the ENTIRE watchlist (not a small subset) must be
    classified as `market_wide_move`, never `correlated_group_move` --
    otherwise a Fed announcement / index-wide selloff and a genuine
    coordinated pump would look identical to the correlation engine.
    """
    market = generate_market_tape(seed=5)
    returns = market["returns"].copy()
    tickers = list(returns.columns)
    shock_start = 200
    span = CONFIG.simulate_group_shock_bars
    rng = np.random.default_rng(123)
    for offset in range(span):
        shock = rng.uniform(0.016, 0.028)
        for tkr in tickers:  # the WHOLE watchlist, not a 3-stock subset
            returns.iloc[shock_start + offset, returns.columns.get_loc(tkr)] += shock + rng.normal(0, 0.0015)

    breaks = detect_correlation_breaks(returns)
    shock_ts = returns.index[shock_start]
    window = breaks[(breaks.index >= shock_ts) & (breaks.index <= shock_ts + pd.Timedelta(minutes=90))]
    flagged = window[window["is_break"]]
    assert not flagged.empty, "the uniform shock should still register as a correlation break"
    kinds = set(flagged["move_kind"])
    print(f"Uniform whole-watchlist shock classified as: {kinds}")
    assert kinds == {"market_wide_move"}, f"expected only market_wide_move, got {kinds}"

    # And the alert layer must carry that through as a lower-urgency,
    # differently-worded alert kind, not a "suspicious cluster" one.
    alerts = generate_correlation_alerts(breaks)
    market_wide_alerts = [a for a in alerts if a.kind == "market_wide_move"]
    cluster_alerts = [a for a in alerts if a.kind == "correlated_group_move" and a.timestamp in window.index]
    assert market_wide_alerts, "expected at least one market_wide_move alert"
    assert not cluster_alerts, "the uniform shock leaked into a correlated_group_move alert"


def test_news_relevance_scoring():
    """Judge-review test (Phase 11 P1 / Phase 17 item 4): the relevance
    scorer should clearly separate a headline that plausibly explains a
    move from one that doesn't, and handle edge cases without throwing.
    """
    relevant = score_headline_relevance("AAPL announces merger and acquisition of a chip startup", "AAPL")
    irrelevant = score_headline_relevance("AAPL named among top workplaces for remote employees", "AAPL")
    borderline = score_headline_relevance("AAPL moves on scheduled earnings update", "AAPL")  # the synthetic generator's own wording

    print(f"relevance: relevant={relevant}, irrelevant={irrelevant}, borderline={borderline}")
    assert relevant > irrelevant, "an M&A headline should score more relevant than an unrelated one"
    assert relevant >= CONFIG.news_relevance_explained_threshold
    assert borderline >= CONFIG.news_relevance_explained_threshold, "the demo's own 'explained' headlines must clear the bar"

    # Edge cases: no news, empty string, malformed/None input -- must not throw.
    assert score_headline_relevance(None) == 0.0
    assert score_headline_relevance("") == 0.0
    assert score_headline_relevance("   ") == 0.0

    empty_lookup = build_synthetic_news_lookup(pd.DataFrame(), CONFIG.news_lookback_hours)
    result = empty_lookup("AAPL", pd.Timestamp.now())
    assert result == {"found": False, "headline": None, "relevance": 0.0}


def test_confidence_score_monotonic_and_consensus_boost():
    """Judge-review test (Phase 17 item 5): confidence must be monotonic in
    the underlying anomaly magnitude, and a consensus (dual-model) alert
    must score at or above the single-model baseline for the same evidence.
    """
    zs = [0.5, 1.5, 2.25, 3.0, 4.5, 6.0]
    scores = [_confidence_score(z) for z in zs]
    assert scores == sorted(scores), "confidence should be non-decreasing in |z|"

    for z in zs:
        single = _confidence_score(z, consensus=False)
        consensus = _confidence_score(z, consensus=True)
        assert consensus >= single, f"consensus score ({consensus}) should be >= single-model score ({single}) at z={z}"

    assert 0.0 <= _confidence_score(0.0) <= 100.0
    assert _confidence_score(50.0) == 100.0  # saturates, never exceeds 100


def test_live_data_failure_falls_back_to_simulated():
    """Judge-review test (Phase 11 P1): the live-mode -> simulated fallback
    path, extracted into data_source.resolve_market_data specifically so it
    doesn't rely on manual inspection of app.py's try/except (app.py itself
    isn't imported by this test suite).
    """
    def failing_fetch_live(tickers):
        raise RuntimeError("simulated network failure")

    result = resolve_market_data(
        mode="Live / historical (yfinance)", watchlist=CONFIG.watchlist, seed=None,
        fetch_live=failing_fetch_live,
    )
    assert result["error"] is not None
    assert result["mode"] == SIMULATED_MODE_LABEL
    assert result["seed"] == FALLBACK_SEED
    assert not result["market"]["prices"].empty
    assert "network failure" in result["note"]

    def working_fetch_live(tickers):
        return generate_market_tape(tickers=tickers, seed=1)  # stand-in for a real yfinance response

    ok_result = resolve_market_data(
        mode="Live / historical (yfinance)", watchlist=CONFIG.watchlist, seed=None,
        fetch_live=working_fetch_live,
    )
    assert ok_result["error"] is None
    assert ok_result["mode"] == "Live / historical (yfinance)"


def test_order_flow_surveillance():
    from data.orderbook_synthetic import generate_order_events
    from order_flow import run_surveillance

    result = generate_order_events("AAPL", seed=3)
    flags = run_surveillance(result["events"])
    caught_kinds = set(flags["kind"]) if not flags.empty else set()
    injected_kinds = {g["kind"] for g in result["ground_truth"]}
    missed = injected_kinds - caught_kinds
    print(f"Order-flow surveillance: injected {injected_kinds}, caught {caught_kinds}")
    assert not missed, f"Missed injected manipulation patterns: {missed}"


if __name__ == "__main__":
    test_pipeline_runs_without_error()
    print("test_pipeline_runs_without_error: PASS")
    test_no_alerts_on_pure_noise()
    print("test_no_alerts_on_pure_noise: PASS")
    test_order_flow_surveillance()
    print("test_order_flow_surveillance: PASS")
    test_recall_across_seeds()
    print("test_recall_across_seeds: PASS")
    test_market_wide_move_not_flagged_as_cluster()
    print("test_market_wide_move_not_flagged_as_cluster: PASS")
    test_news_relevance_scoring()
    print("test_news_relevance_scoring: PASS")
    test_confidence_score_monotonic_and_consensus_boost()
    print("test_confidence_score_monotonic_and_consensus_boost: PASS")
    test_live_data_failure_falls_back_to_simulated()
    print("test_live_data_failure_falls_back_to_simulated: PASS")
