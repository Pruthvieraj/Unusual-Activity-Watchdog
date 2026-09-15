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

from alert_engine import build_synthetic_news_lookup, generate_all_alerts
from config import CONFIG
from correlation_watch import detect_correlation_breaks
from data.synthetic import generate_market_tape
from features import build_all_features
from models.anomaly_model import score_all


def run_scenario(seed: int):
    market = generate_market_tape(seed=seed)
    features = build_all_features(market)
    scored = score_all(features)
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
    scored = score_all(features)
    breaks = detect_correlation_breaks(market["returns"])
    alerts = generate_all_alerts(scored, breaks, merge_bar_minutes=CONFIG.simulate_bar_freq_minutes)
    high_severity = alerts[alerts["severity"] == "High"]
    print(f"High-severity alerts with zero injected anomalies: {len(high_severity)}")
    assert len(high_severity) < 15


if __name__ == "__main__":
    test_pipeline_runs_without_error()
    print("test_pipeline_runs_without_error: PASS")
    test_no_alerts_on_pure_noise()
    print("test_no_alerts_on_pure_noise: PASS")
    test_recall_across_seeds()
    print("test_recall_across_seeds: PASS")
