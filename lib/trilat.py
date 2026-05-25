"""Robust trilateration."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

Array = np.ndarray
LossKind = Literal["linear", "huber"]

from .spatial import X_BOUNDS, Y_BOUNDS, bounds_from_positions, zone_from_min_dist

# Re-export for backward compatibility
__all__ = ["huber_trilat", "X_BOUNDS", "Y_BOUNDS", "bounds_from_positions"]


def _distance_weights(
    d_obs: Array,
    gamma: float,
    *,
    zone_near_boost: float = 1.0,
    zone_far_scale: float = 1.0,
    zone_ref: Optional[Array] = None,
    zone_edges: Optional[tuple[float, float]] = None,
) -> Array:
    """Soft weights: closer RTT → larger weight. Zone from raw d_hat (zone_ref)."""
    d = np.maximum(np.asarray(d_obs, dtype=np.float64).ravel(), 5.0)
    w = (1.0 / d) ** float(gamma)
    if zone_near_boost != 1.0 or zone_far_scale != 1.0:
        ref = np.asarray(zone_ref if zone_ref is not None else d_obs, dtype=np.float64)
        z = zone_from_min_dist(ref, edges=zone_edges) if zone_edges else zone_from_min_dist(ref)
        mult = (zone_near_boost, 1.0, zone_far_scale)[z]
        w = w * mult
    return w / (w.mean() + 1e-12)


def huber_trilat(
    d_obs: Array,
    bs: Array,
    *,
    loss: LossKind = "huber",
    f_scale: float = 1.0,
    x0: Optional[Array] = None,
    min_anchors: int = 3,
    weight_gamma: Optional[float] = None,
    asym_pos_weight: float = 1.0,
    xy_bounds: Optional[tuple[tuple[float, float], tuple[float, float]]] = None,
    zone_near_boost: float = 1.0,
    zone_far_scale: float = 1.0,
    zone_ref: Optional[Array] = None,
    zone_edges: Optional[tuple[float, float]] = None,
) -> Tuple[Array, dict]:
    """
    Return (xy, meta). meta includes success, n_anchors, final_cost.
    """
    d_obs = np.asarray(d_obs, dtype=np.float64).ravel()
    bs = np.asarray(bs, dtype=np.float64)
    if bs.shape[0] != 2:
        bs = bs.T
    n = d_obs.size
    if n < min_anchors:
        c = bs.mean(axis=1)
        return c.copy(), {"success": False, "n_anchors": n, "reason": "too_few_anchors"}

    if x0 is None:
        x0 = bs.mean(axis=1)
    else:
        x0 = np.asarray(x0, dtype=np.float64).reshape(2)

    bx, by = bs[0], bs[1]

    sqrt_w = None
    if weight_gamma is not None and float(weight_gamma) > 0:
        sqrt_w = np.sqrt(
            _distance_weights(
                d_obs,
                weight_gamma,
                zone_near_boost=zone_near_boost,
                zone_far_scale=zone_far_scale,
                zone_ref=zone_ref,
                zone_edges=zone_edges,
            )
        )

    def residual(xy: Array) -> Array:
        pred = np.sqrt((bx - xy[0]) ** 2 + (by - xy[1]) ** 2)
        r = pred - d_obs
        if asym_pos_weight != 1.0:
            w = np.where(r > 0, asym_pos_weight, 1.0)
            r = r * np.sqrt(w)
        if sqrt_w is not None:
            r = r * sqrt_w
        return r

    if xy_bounds is None:
        xy_bounds = (X_BOUNDS, Y_BOUNDS)
    xb, yb = xy_bounds
    loss_arg = "linear" if loss == "linear" else "huber"
    r = least_squares(
        residual,
        x0,
        loss=loss_arg,
        f_scale=f_scale if loss_arg == "huber" else 1.0,
        bounds=([xb[0], yb[0]], [xb[1], yb[1]]),
    )
    return r.x, {
        "success": bool(r.success),
        "n_anchors": n,
        "cost": float(r.cost),
        "nfev": int(r.nfev),
    }
