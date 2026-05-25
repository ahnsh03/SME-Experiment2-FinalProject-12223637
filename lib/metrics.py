"""Position error metrics."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def position_errors(p_true: Array, p_pred: Array) -> Array:
    """Per-user Euclidean error, shape (N,)."""
    return np.hypot(p_pred[0] - p_true[0], p_pred[1] - p_true[1])


def summarize_errors(err: Array) -> dict:
    err = np.asarray(err, dtype=np.float64)
    return {
        "rmse_m": float(np.sqrt(np.mean(err**2))),
        "mae_m": float(np.mean(err)),
        "median_m": float(np.median(err)),
        "p90_m": float(np.percentile(err, 90)),
        "max_m": float(np.max(err)),
        "n": int(err.size),
    }
