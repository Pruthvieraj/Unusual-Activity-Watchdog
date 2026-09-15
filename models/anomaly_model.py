"""
Per-ticker Isolation Forest anomaly scoring, with SHAP explanations for
every flagged point.

Isolation Forest is trained per ticker (not pooled globally) because
different stocks trade at very different volume/volatility scales -- a
pooled model would just learn "big-cap tech = normal, small-cap = weird".
Per-ticker models let each stock be judged against its OWN history, which
is both more correct and matches how a real analyst would think about it
("is this unusual FOR THIS STOCK").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from config import CONFIG

FEATURE_COLUMNS = ["return_zscore", "volume_zscore", "volatility"]


def fit_isolation_forest(feature_df: pd.DataFrame) -> IsolationForest:
    model = IsolationForest(
        n_estimators=CONFIG.isolation_forest_estimators,
        contamination=CONFIG.isolation_forest_contamination,
        random_state=CONFIG.random_state,
    )
    model.fit(feature_df[FEATURE_COLUMNS])
    return model


def score_ticker(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Fit + score one ticker's feature frame. Adds:
      anomaly_score  -- higher = more anomalous (sign-flipped so "higher is worse")
      is_anomaly     -- bool, IsolationForest's own -1/1 flag
      shap_<feature> -- per-feature SHAP contribution to that bar's anomaly score
    """
    X = feature_df[FEATURE_COLUMNS]
    model = fit_isolation_forest(feature_df)

    raw_score = model.decision_function(X)          # higher = more normal
    feature_df = feature_df.copy()
    feature_df["anomaly_score"] = -raw_score          # flip: higher = more anomalous
    feature_df["is_anomaly"] = model.predict(X) == -1

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # TreeExplainer on IsolationForest returns contributions toward the raw
    # (higher-is-normal) score; flip sign to match anomaly_score's direction.
    shap_values = -np.asarray(shap_values)
    for i, col in enumerate(FEATURE_COLUMNS):
        feature_df[f"shap_{col}"] = shap_values[:, i]

    return feature_df


def score_all(features_by_ticker: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {tkr: score_ticker(df) for tkr, df in features_by_ticker.items()}


def explain_point(scored_df: pd.DataFrame, timestamp) -> dict:
    """Human-readable explanation for one flagged bar: which feature(s)
    drove the anomaly score, ranked by |SHAP contribution|.
    """
    row = scored_df.loc[timestamp]
    contribs = {col: row[f"shap_{col}"] for col in FEATURE_COLUMNS}
    ranked = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))
    return {
        "timestamp": timestamp,
        "anomaly_score": float(row["anomaly_score"]),
        "ranked_drivers": ranked,
        "top_driver": ranked[0][0] if ranked else None,
    }
