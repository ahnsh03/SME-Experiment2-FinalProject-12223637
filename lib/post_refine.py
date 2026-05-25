"""Post-trilateration refinement (train-fold only, no GT at inference)."""
from __future__ import annotations

import numpy as np

Array = np.ndarray


def fit_pos_affine(p_hat: Array, p_true: Array) -> Array:
    """
    Fit p_true ≈ M @ [x_hat, y_hat, 1].
    Returns M shape (2, 3).
    """
    p_hat = np.asarray(p_hat, dtype=np.float64)
    p_true = np.asarray(p_true, dtype=np.float64)
    if p_hat.ndim == 1:
        p_hat = p_hat.reshape(2, -1)
    if p_true.ndim == 1:
        p_true = p_true.reshape(2, -1)
    n = p_hat.shape[1]
    X = np.column_stack([p_hat[0], p_hat[1], np.ones(n, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(X, p_true.T, rcond=None)
    return np.asarray(coef.T, dtype=np.float64)


def apply_pos_affine(xy: Array, M: Array) -> Array:
    v = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
    out = np.asarray(M, dtype=np.float64) @ v
    return out
