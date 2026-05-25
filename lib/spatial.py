"""
Canonical indoor space for InF_DH_FR1 (UE positions p).

Convention: ±HALF_X × ±HALF_Y meters from origin → total span (2*HALF_X) × (2*HALF_Y).
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray

# Design / measured half-extents (m)
HALF_X_M = 60.0
HALF_Y_M = 30.0
MARGIN_M = 2.0

X_BOUNDS = (-(HALF_X_M + MARGIN_M), HALF_X_M + MARGIN_M)  # (-62, 62)
Y_BOUNDS = (-(HALF_Y_M + MARGIN_M), HALF_Y_M + MARGIN_M)  # (-32, 32)

# BS deployment (from p_bs in mat)
BS_HALF_X_M = 50.0
BS_HALF_Y_M = 20.0

# UE zone by min(d_hat) — inference-safe proxy for NLOS severity (no GT)
MIN_DIST_ZONE_EDGES = (32.0, 55.0)  # near / mid / far


def bounds_from_positions(p: Array, margin: float = MARGIN_M) -> tuple[tuple[float, float], tuple[float, float]]:
    """Fold-safe bounds from training UE only."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(2, -1)
    xb = (float(p[0].min() - margin), float(p[0].max() + margin))
    yb = (float(p[1].min() - margin), float(p[1].max() + margin))
    # Clip to canonical envelope
    xb = (max(xb[0], X_BOUNDS[0]), min(xb[1], X_BOUNDS[1]))
    yb = (max(yb[0], Y_BOUNDS[0]), min(yb[1], Y_BOUNDS[1]))
    return xb, yb


def default_xy_bounds() -> tuple[tuple[float, float], tuple[float, float]]:
    return X_BOUNDS, Y_BOUNDS


def zone_from_min_dist(d_hat_u: Array, edges: tuple[float, float] = MIN_DIST_ZONE_EDGES) -> int:
    """0=near, 1=mid, 2=far from shortest RTT (LOS proxy)."""
    m = float(np.min(np.asarray(d_hat_u, dtype=np.float64)))
    if m < edges[0]:
        return 0
    if m < edges[1]:
        return 1
    return 2


def zone_from_true_radius(r_ue: float, edges: tuple[float, float] = (18.0, 35.0)) -> int:
    """Diagnostic zones from GT radius (train analysis only)."""
    if r_ue < edges[0]:
        return 0
    if r_ue < edges[1]:
        return 1
    return 2


def clip_xy(xy: Array) -> Array:
    out = np.asarray(xy, dtype=np.float64).reshape(2)
    out[0] = np.clip(out[0], X_BOUNDS[0], X_BOUNDS[1])
    out[1] = np.clip(out[1], Y_BOUNDS[0], Y_BOUNDS[1])
    return out


def space_summary(p: Array, bs: Array | None = None) -> dict:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(2, -1)
    s = {
        "half_extent_m": {"x": HALF_X_M, "y": HALF_Y_M},
        "total_span_m": {"x": 2 * HALF_X_M, "y": 2 * HALF_Y_M},
        "bounds_m": {"x": list(X_BOUNDS), "y": list(Y_BOUNDS)},
        "measured_ue_span_m": {
            "x": float(p[0].max() - p[0].min()),
            "y": float(p[1].max() - p[1].min()),
        },
        "measured_ue_half_extent_m": {
            "x": float(max(abs(p[0].min()), abs(p[0].max()))),
            "y": float(max(abs(p[1].min()), abs(p[1].max()))),
        },
    }
    if bs is not None:
        bs = np.asarray(bs, dtype=np.float64)
        if bs.shape[0] != 2:
            bs = bs.T
        s["measured_bs_span_m"] = {
            "x": float(bs[0].max() - bs[0].min()),
            "y": float(bs[1].max() - bs[1].min()),
        }
    return s
