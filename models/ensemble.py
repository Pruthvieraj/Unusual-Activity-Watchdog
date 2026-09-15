"""
Combines the Isolation Forest (models/anomaly_model.py) and the neural
autoencoder (models/autoencoder_model.py) into one ensemble score per bar,
and -- more usefully than the raw number -- flags whether both
architecturally-independent models agree.

`consensus == True` (both models flag the same bar) is the strongest
signal this project can produce and is surfaced as such in the UI and in
alert severity; a single-model flag is kept but visibly marked as weaker
evidence.
"""

from __future__ import annotations

import pandas as pd

from models.anomaly_model import score_all as score_all_isolation_forest
from models.autoencoder_model import score_all_autoencoder


def build_ensemble(features_by_ticker: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if_scored = score_all_isolation_forest(features_by_ticker)
    ae_scored = score_all_autoencoder(features_by_ticker)

    out = {}
    for tkr in features_by_ticker:
        if_df = if_scored[tkr]
        ae_df = ae_scored[tkr]

        merged = if_df.copy()
        merged["ae_error"] = ae_df["ae_error"]
        merged["ae_score"] = ae_df["ae_score"]
        merged["ae_is_anomaly"] = ae_df["ae_is_anomaly"]

        # Normalize the Isolation Forest's anomaly_score to a comparable
        # z-scale so it can be meaningfully averaged with the autoencoder's
        # (already z-scored) reconstruction error.
        std = merged["anomaly_score"].std() or 1.0
        if_score_z = (merged["anomaly_score"] - merged["anomaly_score"].mean()) / std

        merged["ensemble_score"] = (if_score_z + merged["ae_score"]) / 2
        merged["consensus"] = merged["is_anomaly"] & merged["ae_is_anomaly"]
        merged["any_model_flagged"] = merged["is_anomaly"] | merged["ae_is_anomaly"]
        out[tkr] = merged

    return out
