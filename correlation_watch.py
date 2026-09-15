"""
Cross-stock correlation-break detection: "several stocks moving together in
a suspicious way".

Individual-ticker anomaly scoring (models/anomaly_model.py) can't see this
pattern by design -- each stock might look perfectly ordinary on its own,
and it's only the fact that several of them moved together, well outside
their normal co-movement, that's suspicious. So this is a separate engine
that looks at the whole watchlist at once.

Approach:
  1. At each bar, compare the SHORT rolling pairwise-correlation matrix
     (recent behaviour) against a LONGER trailing baseline correlation
     matrix ("how these stocks normally relate to each other") -> a
     "corr_delta" series.
  2. Pairwise correlation over a short window is inherently noisy, so
     instead of flagging on a fixed absolute jump, a bar is flagged when
     its delta is itself unusual relative to ITS OWN recent history (a
     z-score of the delta series). This self-calibrates per watchlist
     instead of needing a hand-tuned constant.
  3. When flagged, tickers are ranked by how much they contributed to the
     jump, so the alert can name the actual cluster involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONFIG


def _avg_offdiag(corr: pd.DataFrame) -> float:
    n = corr.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(corr.values[mask].mean())


def _top_k_offdiag_mean(delta_matrix: pd.DataFrame, k: int = 3) -> float:
    """Mean of the k largest POSITIVE off-diagonal deltas.

    Averaging the delta across *all* pairs (e.g. 21 pairs for a 7-stock
    watchlist) dilutes a genuine 3-stock cluster down to noise, since only
    the 3 pairs inside that cluster actually move -- the other 18 stay flat.
    Looking at the top-k pairs instead keys directly onto "is there a small
    group of pairs moving together much more than usual", which is exactly
    the pattern being hunted for, and it naturally requires more than one
    coincidentally-noisy pair to fire.
    """
    n = delta_matrix.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    values = delta_matrix.values[mask]
    values = values[values > 0]
    if len(values) == 0:
        return 0.0
    top = np.sort(values)[::-1][:k]
    return float(top.mean())


def detect_correlation_breaks(returns: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame indexed by timestamp with columns:
      avg_corr_current, avg_corr_baseline, corr_delta, delta_zscore,
      is_break, cluster
    `cluster` is a list of tickers most responsible for the break (empty
    list when is_break is False).
    """
    win = CONFIG.correlation_window
    base_win = CONFIG.correlation_baseline_window
    n = len(returns)

    delta_rows = []
    delta_matrices = {}
    for t in range(base_win + win, n):
        current = returns.iloc[t - win: t]
        baseline = returns.iloc[t - base_win - win: t - win]

        current_corr = current.corr().fillna(0)
        baseline_corr = baseline.corr().fillna(0)

        avg_current = _avg_offdiag(current_corr)
        avg_baseline = _avg_offdiag(baseline_corr)
        delta_matrix = current_corr - baseline_corr
        cluster_signal = _top_k_offdiag_mean(delta_matrix, k=3)

        ts = returns.index[t]
        delta_rows.append({
            "timestamp": ts,
            "avg_corr_current": avg_current,
            "avg_corr_baseline": avg_baseline,
            "corr_delta": cluster_signal,
        })
        delta_matrices[ts] = delta_matrix

    df = pd.DataFrame(delta_rows).set_index("timestamp")

    # Adaptive threshold: z-score of each delta against its OWN prior
    # history. This uses an *expanding* (whole-history-so-far) mean/std
    # rather than a short rolling window, deliberately: a short trailing
    # window would include the anomalous bars themselves, dragging the
    # "normal" baseline up and masking the very event it's supposed to
    # catch. Expanding stats barely move for a brief localized event, so
    # the anomaly still stands out. shift(1) keeps it causal -- a bar is
    # only judged against deltas seen strictly before it.
    min_hist = CONFIG.correlation_delta_history_window
    expanding_mean = df["corr_delta"].expanding(min_periods=min_hist).mean().shift(1)
    expanding_std = df["corr_delta"].expanding(min_periods=min_hist).std(ddof=0).shift(1)
    df["delta_zscore"] = ((df["corr_delta"] - expanding_mean) / expanding_std.replace(0, np.nan)).fillna(0)
    df["is_break"] = df["delta_zscore"] > CONFIG.correlation_break_zscore

    clusters = []
    for ts, is_break in df["is_break"].items():
        if not is_break:
            clusters.append([])
            continue
        delta_matrix = delta_matrices[ts]
        involvement = delta_matrix.clip(lower=0).sum(axis=1).sort_values(ascending=False)
        top_score = involvement.iloc[0] if len(involvement) else 0
        cluster = list(involvement[involvement >= top_score * 0.6].index[:4]) if top_score > 0 else []
        clusters.append(cluster)
    df["cluster"] = clusters

    return df
