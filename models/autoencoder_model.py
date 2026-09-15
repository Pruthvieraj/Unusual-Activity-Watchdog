"""
Neural-network autoencoder: a second, architecturally independent anomaly
model that gets ensembled with the Isolation Forest in
models/anomaly_model.py.

Why a second model at all: any single detector has blind spots shaped by
its own assumptions. Isolation Forest is a tree-based partitioning method;
an autoencoder is a compression-based one. When both independently flag the
same bar, that's a much stronger signal than either alone -- this is the
same "ensemble of structurally different models" principle production
fraud/anomaly systems use, not just stacking two similar models for show.

Kept as a compact MLP autoencoder (scikit-learn's MLPRegressor trained to
reconstruct its own input) rather than a recurrent/deep net on purpose: the
model trains in well under a second per ticker on this data volume and adds
no heavy dependency (no PyTorch/TensorFlow) -- important for staying inside
a free-tier hosting deploy. Feeding it a short SLIDING WINDOW of bars
(flattened, `CONFIG.autoencoder_window` wide) rather than a single bar still
gives it short-range temporal context, which is what actually matters for
catching a multi-bar anomaly shape.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config import CONFIG

# At this data volume the autoencoder only needs a rough reconstruction
# baseline, not full convergence -- a few hundred fast iterations with a
# higher learning rate gets there; sklearn's convergence warning past that
# point is expected and not a sign of a problem, so it's suppressed rather
# than left to alarm anyone reading the demo's console output.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

WINDOW_COLUMNS = ["return_zscore", "volume_zscore", "volatility"]


def _build_windows(feature_df: pd.DataFrame, window: int) -> tuple[np.ndarray, pd.Index]:
    X = feature_df[WINDOW_COLUMNS].values
    n = len(X)
    if n <= window:
        return np.empty((0, window * len(WINDOW_COLUMNS))), feature_df.index[:0]
    windows = np.array([X[i - window + 1: i + 1].flatten() for i in range(window - 1, n)])
    idx = feature_df.index[window - 1:]
    return windows, idx


def score_ticker_autoencoder(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Adds ae_error, ae_score (z-scored reconstruction error), and
    ae_is_anomaly columns to a copy of feature_df.
    """
    window = CONFIG.autoencoder_window
    out = feature_df.copy()
    out["ae_error"] = 0.0
    out["ae_score"] = 0.0
    out["ae_is_anomaly"] = False

    X, idx = _build_windows(feature_df, window)
    if len(X) < 20:  # not enough data to train a meaningful autoencoder
        return out

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = MLPRegressor(
        hidden_layer_sizes=CONFIG.autoencoder_hidden_layers,
        activation="tanh",
        solver="adam",
        learning_rate_init=0.02,
        max_iter=250,
        random_state=CONFIG.random_state,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.15,
    )
    model.fit(Xs, Xs)
    reconstructed = model.predict(Xs)
    error = np.mean((Xs - reconstructed) ** 2, axis=1)

    err_series = pd.Series(error, index=idx)
    std = err_series.std() or 1.0
    z = (err_series - err_series.mean()) / std

    out.loc[idx, "ae_error"] = err_series.values
    out.loc[idx, "ae_score"] = z.values
    out.loc[idx, "ae_is_anomaly"] = z.values > CONFIG.autoencoder_zscore_alert
    return out


def score_all_autoencoder(features_by_ticker: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {tkr: score_ticker_autoencoder(df) for tkr, df in features_by_ticker.items()}
