"""
Automated coverage for the two Section-32 test-plan items that the
pre-existing tests/test_pipeline.py cannot cover without being modified
(and Section 32 requires that file pass UNMODIFIED):

  - "Composite score: unit test asserting monotonicity" -- this is about
    detection/score.py's NEW compute_composite_score() (Section 9), not
    alert_engine.py's pre-existing _confidence_score() (a different,
    older correlation-alert formula that tests/test_pipeline.py already
    covers under a different name).
  - "Persistence workflow" -- automates the create -> investigate -> note
    -> resolve -> reload round trip at the data/store.py + services/
    alert_service.py layer, as a fast, repeatable complement to the
    one-time manual browser walkthrough of the same flow in the Alert
    Feed UI.

Kept in its own file, deliberately never imported by or merged into
tests/test_pipeline.py, so that file's byte-for-byte contents (and thus
its "pass unmodified" guarantee) are never at risk from future edits here.

Run with: python3 tests/test_sprint_upgrades.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from data import store
from detection.score import compute_composite_score


def test_composite_score_monotonic_per_component():
    """Increasing any ONE input z-score, holding every other input fixed,
    must never decrease the composite total -- the exact wording of the
    plan's Section 32 test-plan row for this check."""
    baseline = dict(
        return_zscore=1.0, volume_zscore=1.0, volatility_zscore=1.0,
        trade_burst_rate_zscore=1.0, news_relevance=0.5,
        cross_stock_delta_zscore=1.0, in_active_cluster=True,
    )
    base_total = compute_composite_score(**baseline).total

    varying_components = [
        "return_zscore", "volume_zscore", "volatility_zscore",
        "trade_burst_rate_zscore", "cross_stock_delta_zscore",
    ]
    for component in varying_components:
        bumped = dict(baseline)
        bumped[component] = baseline[component] + 2.0
        bumped_total = compute_composite_score(**bumped).total
        assert bumped_total >= base_total, (
            f"increasing {component} from {baseline[component]} to {bumped[component]} "
            f"decreased the composite score ({base_total} -> {bumped_total})"
        )

    # news_relevance moves the OTHER way by design (a check that finds
    # LESS relevant news is more anomalous, not less) -- so its
    # monotonicity is asserted in the direction the module documents,
    # not blindly bumped like the z-score inputs above.
    less_relevant = dict(baseline, news_relevance=0.1)
    more_relevant = dict(baseline, news_relevance=0.9)
    assert (compute_composite_score(**less_relevant).total
            >= compute_composite_score(**more_relevant).total), (
        "a LESS relevant nearby headline should score at least as anomalous "
        "as a MORE relevant one, holding every other input fixed"
    )

    # Consensus boost: strictly additive, never decreases the total.
    single = compute_composite_score(**baseline, consensus=False).total
    consensus = compute_composite_score(**baseline, consensus=True).total
    assert consensus >= single

    # Never exceeds the documented 0-100 bound even at extreme inputs.
    saturated = compute_composite_score(
        return_zscore=50, volume_zscore=50, volatility_zscore=50,
        trade_burst_rate_zscore=50, news_relevance=0.0,
        cross_stock_delta_zscore=50, in_active_cluster=True, consensus=True,
    ).total
    assert 0.0 <= saturated <= 100.0


def test_composite_score_missing_signal_contributes_zero_not_fake_value():
    """A signal that was never computed for this alert kind (None) must
    contribute exactly 0, distinct from a signal that WAS checked and
    found nothing (0.0) -- the module's own documented contract."""
    without_burst = compute_composite_score(return_zscore=2.0, trade_burst_rate_zscore=None)
    assert without_burst.trade_burst == 0.0

    without_news_check = compute_composite_score(return_zscore=2.0, news_relevance=None)
    checked_but_irrelevant = compute_composite_score(return_zscore=2.0, news_relevance=0.0)
    assert without_news_check.news_divergence == 0.0
    assert checked_but_irrelevant.news_divergence > 0.0, (
        "a news check that ran and found NOTHING relevant should count as a "
        "real divergence signal, not be silently treated like no check ran"
    )


_FIXED_FIRST_SEEN = pd.Timestamp("2026-01-15 09:30:00")


def _sample_alerts_df(last_seen: pd.Timestamp | None = None) -> pd.DataFrame:
    # A FIXED first_seen (not pd.Timestamp.now(), which differs to the
    # microsecond between two calls) is what lets a second call -- standing
    # in for the next pipeline rerun picking up the "same" ongoing alert --
    # resolve to the exact same alert_key as the first.
    row = {
        "kind": "volume_burst", "tickers": "AAPL", "severity": "High",
        "headline": "Test alert", "detail": "Synthetic alert for the persistence round-trip test.",
        "evidence": {}, "score_breakdown": {}, "consensus": False,
        "confidence": 75.0, "agreement_pct": 100.0,
        "first_seen": _FIXED_FIRST_SEEN, "last_seen": last_seen or _FIXED_FIRST_SEEN, "occurrences": 1,
    }
    return pd.DataFrame([row])


def test_persistence_full_investigation_round_trip():
    """Automates the plan's "Persistence workflow" test-plan row: create an
    alert, mark Investigating, add a note, mark Resolved, RE-OPEN THE DB
    CONNECTION (the closest a script gets to "reload the page" -- a fresh
    connection re-reads from the same on-disk file rather than any
    in-memory state), and confirm both the status and the note survived.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "test_alerts.db")

        conn = store.get_connection(db_path)
        store.init_db(conn)
        inserted = store.upsert_alerts(conn, _sample_alerts_df())
        assert inserted == 1
        alert_id = store.get_id_by_key(conn, "volume_burst", "AAPL", _FIXED_FIRST_SEEN)
        assert alert_id is not None

        row = store.get_alert(conn, alert_id)
        assert row["status"] == store.STATUS_NEW

        store.set_status(conn, alert_id, store.STATUS_INVESTIGATING, note="Looking into this one.")
        store.set_status(conn, alert_id, store.STATUS_RESOLVED, note="Confirmed benign, closing out.")
        conn.close()

        # A brand-new connection against the same file -- nothing carried
        # over in memory -- is the script-level equivalent of a page reload.
        reopened = store.get_connection(db_path)
        row_after_reload = store.get_alert(reopened, alert_id)
        assert row_after_reload["status"] == store.STATUS_RESOLVED
        assert "Looking into this one." in row_after_reload["notes"]
        assert "Confirmed benign, closing out." in row_after_reload["notes"]

        # Re-running the pipeline's upsert for the SAME alert (as a rerun
        # would do every refresh, typically with a later last_seen) must
        # never clobber the investigation work that just survived the reload.
        store.upsert_alerts(reopened, _sample_alerts_df(last_seen=pd.Timestamp("2026-01-15 09:35:00")))
        row_after_rerun = store.get_alert(reopened, alert_id)
        assert row_after_rerun["status"] == store.STATUS_RESOLVED, (
            "a routine pipeline rerun overwrote an analyst's RESOLVED status back to NEW"
        )
        reopened.close()


if __name__ == "__main__":
    test_composite_score_monotonic_per_component()
    print("test_composite_score_monotonic_per_component: PASS")
    test_composite_score_missing_signal_contributes_zero_not_fake_value()
    print("test_composite_score_missing_signal_contributes_zero_not_fake_value: PASS")
    test_persistence_full_investigation_round_trip()
    print("test_persistence_full_investigation_round_trip: PASS")
