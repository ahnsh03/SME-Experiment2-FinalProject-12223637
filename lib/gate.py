"""Hard geometric gate (V15-style) on corrected distances."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .trilat import huber_trilat

Array = np.ndarray


def geometric_residuals(p_xy: Array, d_corr: Array, bs: Array) -> Array:
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    pred = np.sqrt((bs[0] - p_xy[0]) ** 2 + (bs[1] - p_xy[1]) ** 2)
    return np.abs(pred - np.asarray(d_corr, dtype=np.float64).ravel())


def apply_hard_gate(
    d_corr: Array,
    bs: Array,
    p0: Array,
    tau: float,
    *,
    min_keep: int = 4,
) -> Tuple[Array, Array, int]:
    """Return filtered (d, bs_sub), n_dropped."""
    r = geometric_residuals(p0, d_corr, bs)
    keep = r <= tau
    n_drop = int(18 - keep.sum())
    if keep.sum() < min_keep:
        return np.asarray(d_corr).ravel(), bs, 0
    d_f = np.asarray(d_corr).ravel()[keep]
    bs_f = bs[:, keep] if bs.shape[0] == 2 else bs[keep]
    return d_f, bs_f, n_drop


def collect_oof_gate_residuals(
    d_hat: Array,
    p: Array,
    bs: Array,
    calib_mode: str,
    *,
    loss: str = "huber",
    f_scale: float = 1.0,
) -> Array:
    """Pool |geom - d_corr| at pre-gate trilat for OOF tau search (uses GT only inside CV)."""
    from sklearn.model_selection import KFold

    from .calib import fit_calib

    n = d_hat.shape[1]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pool = []
    for tr, va in kf.split(np.arange(n)):
        train_idx = np.asarray(tr, dtype=np.int64)
        calib = fit_calib(d_hat, p, bs, train_idx, calib_mode)
        for u in va:
            dc = calib.apply(d_hat[:, u])
            p0, _ = huber_trilat(dc, bs, loss=loss, f_scale=f_scale)
            pool.extend(geometric_residuals(p0, dc, bs).tolist())
    return np.asarray(pool, dtype=np.float64)


def tau_candidates_from_oof(pool: Array) -> list[float]:
    if pool.size == 0:
        return [20.0]
    qs = [50, 60, 70, 75, 80, 85, 90, 92, 95]
    cands = [float(np.percentile(pool, q)) for q in qs]
    cands = sorted(set(max(3.0, round(t, 2)) for t in cands))
    return cands
