"""RTT distance calibration (incremental stages)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .io_mat import geometric_distances
from .spatial import MIN_DIST_ZONE_EDGES, zone_from_min_dist

Array = np.ndarray
CalibMode = Literal[
    "none",
    "global_affine",
    "per_bs_affine",
    "per_bs_quadratic",
    "per_bs_isotonic",
    "per_bs_isotonic_radial",
    "mlp_dist",
    "isotonic_mlp",
    "gbr_dist",
    "pos_mlp",
    "per_bs_isotonic_mindist",
    "per_bs_quantile",
    "isotonic_mlp_mindist",
    "quantile_mlp",
    "isotonic_mlp_far2",
]

ZONE_EDGES = (15.0, 30.0)
MIN_ZONE_SAMPLES = 25


@dataclass
class CalibParams:
    mode: CalibMode
    alpha: Array = field(default_factory=lambda: np.array(1.0))
    beta: Array = field(default_factory=lambda: np.array(0.0))
    # per_bs_quadratic: (18, 3) rows [a, b, c] -> a*d + b*d^2 + c
    quad: Optional[Array] = None
    # per_bs_isotonic: list of 18 (x_thresh, y_thresh) arrays
    iso_x: Optional[list] = None
    iso_y: Optional[list] = None
    centroid: Optional[Array] = None
    zone_edges: Optional[Array] = None
    iso_x_zone: Optional[list] = None
    iso_y_zone: Optional[list] = None
    iso_x_fb: Optional[list] = None
    iso_y_fb: Optional[list] = None
    mlp_bundle: Optional[Any] = None  # MLPBundle from mlp_calib
    gbr_bundle: Optional[Any] = None
    pos_bundle: Optional[Any] = None  # PosMLPBundle
    quantile_tau: Optional[float] = None
    quantile_models: Optional[list] = None  # per-BS QuantileRegressor
    far_iso_x: Optional[list] = None
    far_iso_y: Optional[list] = None
    far_min_edge: Optional[float] = None
    pos_affine: Optional[Array] = None  # (2, 3), set in CV/train not in calib.fit
    pos_affine_far: Optional[Array] = None  # second affine for far UE only

    def zone_index(self, r: float) -> int:
        e = self.zone_edges if self.zone_edges is not None else ZONE_EDGES
        if r < e[0]:
            return 0
        if r < e[1]:
            return 1
        return 2

    def apply(self, d_hat_u: Array, zone: Optional[int] = None) -> Array:
        d = np.asarray(d_hat_u, dtype=np.float64).reshape(-1)
        if self.mode == "none":
            return d.copy()
        if self.mode == "global_affine":
            return self.alpha * d + self.beta
        if self.mode == "per_bs_affine":
            return self.alpha * d + self.beta
        if self.mode == "per_bs_quadratic":
            q = np.asarray(self.quad, dtype=np.float64)
            return q[:, 0] * d + q[:, 1] * d**2 + q[:, 2]
        if self.mode == "per_bs_isotonic":
            out = np.empty(18, dtype=np.float64)
            for k in range(18):
                xk = np.asarray(self.iso_x[k], dtype=np.float64)
                yk = np.asarray(self.iso_y[k], dtype=np.float64)
                out[k] = np.interp(d[k], xk, yk, left=yk[0], right=yk[-1])
            return out
        if self.mode == "per_bs_isotonic_mindist":
            e = tuple(self.zone_edges) if self.zone_edges is not None else MIN_DIST_ZONE_EDGES
            z = zone_from_min_dist(d, edges=e) if zone is None else int(zone)
            out = np.empty(18, dtype=np.float64)
            for k in range(18):
                xk = np.asarray(self.iso_x_zone[k][z], dtype=np.float64)
                yk = np.asarray(self.iso_y_zone[k][z], dtype=np.float64)
                out[k] = np.interp(d[k], xk, yk, left=yk[0], right=yk[-1])
            return out
        if self.mode == "per_bs_quantile":
            assert self.quantile_models is not None
            out = np.empty(18, dtype=np.float64)
            for k, m in enumerate(self.quantile_models):
                out[k] = float(m.predict(d[k].reshape(1, -1))[0])
            return np.maximum(out, 1.0)
        if self.mode == "isotonic_mlp_mindist":
            e = tuple(self.zone_edges) if self.zone_edges is not None else MIN_DIST_ZONE_EDGES
            z = zone_from_min_dist(d, edges=e) if zone is None else int(zone)
            iso_d = CalibParams(
                "per_bs_isotonic_mindist",
                iso_x_zone=self.iso_x_zone,
                iso_y_zone=self.iso_y_zone,
                zone_edges=self.zone_edges,
                iso_x_fb=self.iso_x_fb,
                iso_y_fb=self.iso_y_fb,
            ).apply(d, zone=z)
            assert self.mlp_bundle is not None
            return self.mlp_bundle.apply(iso_d)
        if self.mode == "quantile_mlp":
            qd = CalibParams(
                "per_bs_quantile",
                quantile_models=self.quantile_models,
                quantile_tau=self.quantile_tau,
            ).apply(d)
            assert self.mlp_bundle is not None
            return self.mlp_bundle.apply(qd)
        if self.mode == "isotonic_mlp_far2":
            d1 = CalibParams(
                "isotonic_mlp",
                iso_x=self.iso_x,
                iso_y=self.iso_y,
                mlp_bundle=self.mlp_bundle,
            ).apply(d)
            if (
                self.far_min_edge is not None
                and self.far_iso_x is not None
                and float(np.min(d)) >= float(self.far_min_edge)
            ):
                out = np.empty(18, dtype=np.float64)
                for k in range(18):
                    xk = np.asarray(self.far_iso_x[k], dtype=np.float64)
                    yk = np.asarray(self.far_iso_y[k], dtype=np.float64)
                    out[k] = np.interp(d1[k], xk, yk, left=yk[0], right=yk[-1])
                return out
            return d1
        if self.mode == "per_bs_isotonic_radial":
            z = 1 if zone is None else int(zone)
            out = np.empty(18, dtype=np.float64)
            for k in range(18):
                xk = np.asarray(self.iso_x_zone[k][z], dtype=np.float64)
                yk = np.asarray(self.iso_y_zone[k][z], dtype=np.float64)
                out[k] = np.interp(d[k], xk, yk, left=yk[0], right=yk[-1])
            return out
        if self.mode == "mlp_dist":
            assert self.mlp_bundle is not None
            return self.mlp_bundle.apply(d)
        if self.mode == "isotonic_mlp":
            iso_d = CalibParams("per_bs_isotonic", iso_x=self.iso_x, iso_y=self.iso_y).apply(d)
            assert self.mlp_bundle is not None
            return self.mlp_bundle.apply(iso_d)
        if self.mode == "gbr_dist":
            assert self.gbr_bundle is not None
            return self.gbr_bundle.apply(d)
        if self.mode == "pos_mlp":
            assert self.pos_bundle is not None
            return self.pos_bundle.apply(d)
        raise ValueError(f"unknown calib mode {self.mode}")

    def apply_fallback(self, d_hat_u: Array) -> Array:
        """All-train isotonic per BS (radial pass-1 prior)."""
        d = np.asarray(d_hat_u, dtype=np.float64).ravel()
        out = np.empty(18, dtype=np.float64)
        for k in range(18):
            xk = np.asarray(self.iso_x_fb[k], dtype=np.float64)
            yk = np.asarray(self.iso_y_fb[k], dtype=np.float64)
            out[k] = np.interp(d[k], xk, yk, left=yk[0], right=yk[-1])
        return out

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "mode": self.mode,
            "alpha": np.asarray(self.alpha).tolist(),
            "beta": np.asarray(self.beta).tolist(),
        }
        if self.quad is not None:
            d["quad"] = np.asarray(self.quad).tolist()
        if self.iso_x is not None and self.iso_y is not None:
            d["iso_x"] = [np.asarray(x).tolist() for x in self.iso_x]
            d["iso_y"] = [np.asarray(y).tolist() for y in self.iso_y]
        if self.centroid is not None:
            d["centroid"] = np.asarray(self.centroid).tolist()
            d["zone_edges"] = np.asarray(self.zone_edges).tolist()
            d["iso_x_zone"] = [
                [np.asarray(x).tolist() for x in row] for row in self.iso_x_zone
            ]
            d["iso_y_zone"] = [
                [np.asarray(y).tolist() for y in row] for row in self.iso_y_zone
            ]
            if self.iso_x_fb is not None:
                d["iso_x_fb"] = [np.asarray(x).tolist() for x in self.iso_x_fb]
                d["iso_y_fb"] = [np.asarray(y).tolist() for y in self.iso_y_fb]
        if self.mlp_bundle is not None:
            from .mlp_calib import bundle_to_dict

            d["mlp"] = bundle_to_dict(self.mlp_bundle)
        if self.gbr_bundle is not None:
            d["gbr"] = True
        if self.pos_bundle is not None:
            from .pos_mlp import bundle_to_dict as pos_to_dict

            d["pos"] = pos_to_dict(self.pos_bundle)
        if self.pos_affine is not None:
            d["pos_affine"] = np.asarray(self.pos_affine).tolist()
        if self.pos_affine_far is not None:
            d["pos_affine_far"] = np.asarray(self.pos_affine_far).tolist()
        if self.far_min_edge is not None:
            d["far_min_edge"] = float(self.far_min_edge)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CalibParams":
        return cls(
            mode=d["mode"],
            alpha=np.asarray(d.get("alpha", 1.0), dtype=np.float64),
            beta=np.asarray(d.get("beta", 0.0), dtype=np.float64),
            quad=np.asarray(d["quad"], dtype=np.float64) if "quad" in d else None,
            iso_x=d.get("iso_x"),
            iso_y=d.get("iso_y"),
            centroid=np.asarray(d["centroid"], dtype=np.float64) if "centroid" in d else None,
            zone_edges=np.asarray(d["zone_edges"], dtype=np.float64)
            if "zone_edges" in d
            else None,
            iso_x_zone=d.get("iso_x_zone"),
            iso_y_zone=d.get("iso_y_zone"),
            iso_x_fb=d.get("iso_x_fb"),
            iso_y_fb=d.get("iso_y_fb"),
            pos_affine=np.asarray(d["pos_affine"], dtype=np.float64)
            if "pos_affine" in d
            else None,
            pos_affine_far=np.asarray(d["pos_affine_far"], dtype=np.float64)
            if "pos_affine_far" in d
            else None,
            far_min_edge=float(d["far_min_edge"]) if "far_min_edge" in d else None,
        )


def fit_calib(
    d_hat: Array,
    p: Array,
    bs: Array,
    train_idx: Array,
    mode: CalibMode,
    *,
    quantile_tau: Optional[float] = None,
    mlp_noise_std: float = 0.0,
) -> CalibParams:
    if mode == "none":
        return CalibParams("none", np.array(1.0), np.array(0.0))

    d_true = geometric_distances(p[:, train_idx], bs)
    dh = d_hat[:, train_idx]

    if mode == "global_affine":
        ab = _lstsq_affine(dh.ravel(), d_true.ravel())
        return CalibParams("global_affine", np.array(ab[0]), np.array(ab[1]))

    if mode == "per_bs_affine":
        alphas = np.zeros(18, dtype=np.float64)
        betas = np.zeros(18, dtype=np.float64)
        for k in range(18):
            ab = _lstsq_affine(dh[k], d_true[k])
            alphas[k], betas[k] = ab
        return CalibParams("per_bs_affine", alphas, betas)

    if mode == "per_bs_quadratic":
        quad = np.zeros((18, 3), dtype=np.float64)
        for k in range(18):
            quad[k] = _lstsq_quad(dh[k], d_true[k])
        return CalibParams("per_bs_quadratic", quad=quad)

    if mode == "per_bs_isotonic":
        iso_x, iso_y = [], []
        for k in range(18):
            xk, yk = _fit_isotonic_1d(dh[k], d_true[k])
            iso_x.append(xk)
            iso_y.append(yk)
        return CalibParams("per_bs_isotonic", iso_x=iso_x, iso_y=iso_y)

    if mode == "mlp_dist":
        from .mlp_calib import fit_mlp_calib

        bundle = fit_mlp_calib(d_hat, p, bs, train_idx)
        return CalibParams("mlp_dist", mlp_bundle=bundle)

    if mode == "isotonic_mlp":
        from .mlp_calib import fit_mlp_on_iso

        iso = fit_calib(d_hat, p, bs, train_idx, "per_bs_isotonic")
        bundle = fit_mlp_on_iso(
            d_hat,
            p,
            bs,
            train_idx,
            lambda du: iso.apply(du),
            epochs=100,
            noise_std=mlp_noise_std,
        )
        return CalibParams(
            "isotonic_mlp",
            iso_x=iso.iso_x,
            iso_y=iso.iso_y,
            mlp_bundle=bundle,
        )

    if mode == "isotonic_mlp_mindist":
        from .mlp_calib import fit_mlp_on_iso

        iso = fit_calib(d_hat, p, bs, train_idx, "per_bs_isotonic_mindist")

        def _apply(du: Array) -> Array:
            e = tuple(float(x) for x in iso.zone_edges)
            z = zone_from_min_dist(du, edges=e)
            return iso.apply(du, zone=z)

        bundle = fit_mlp_on_iso(d_hat, p, bs, train_idx, _apply, epochs=100)
        return CalibParams(
            "isotonic_mlp_mindist",
            zone_edges=iso.zone_edges,
            iso_x_zone=iso.iso_x_zone,
            iso_y_zone=iso.iso_y_zone,
            iso_x_fb=iso.iso_x_fb,
            iso_y_fb=iso.iso_y_fb,
            mlp_bundle=bundle,
        )

    if mode == "quantile_mlp":
        from .mlp_calib import fit_mlp_on_iso

        tau = float(quantile_tau) if quantile_tau is not None else 0.35
        q = fit_calib(d_hat, p, bs, train_idx, "per_bs_quantile", quantile_tau=tau)
        bundle = fit_mlp_on_iso(
            d_hat, p, bs, train_idx, lambda du: q.apply(du), epochs=100
        )
        return CalibParams(
            "quantile_mlp",
            quantile_tau=q.quantile_tau,
            quantile_models=q.quantile_models,
            mlp_bundle=bundle,
        )

    if mode == "isotonic_mlp_far2":
        base = fit_calib(
            d_hat, p, bs, train_idx, "isotonic_mlp", mlp_noise_std=mlp_noise_std
        )
        mins = np.array(
            [float(np.min(d_hat[:, int(u)])) for u in train_idx], dtype=np.float64
        )
        e_far = float(np.percentile(mins, 66))
        far_mask = mins >= e_far
        if int(far_mask.sum()) < MIN_ZONE_SAMPLES:
            base.mode = "isotonic_mlp_far2"
            base.far_min_edge = e_far
            return base

        far_users = train_idx[far_mask]
        d_true = geometric_distances(p[:, train_idx], bs)
        dh = d_hat[:, train_idx]
        far_x, far_y = [], []
        for k in range(18):
            xs, ys = [], []
            for j, u in enumerate(train_idx):
                if not far_mask[j]:
                    continue
                d1 = base.apply(d_hat[:, int(u)])
                xs.append(d1[k])
                ys.append(d_true[k, j])
            if len(xs) < MIN_ZONE_SAMPLES:
                xk, yk = _fit_isotonic_1d(dh[k], d_true[k])
            else:
                xk, yk = _fit_isotonic_1d(np.asarray(xs), np.asarray(ys))
            far_x.append(xk)
            far_y.append(yk)

        base.mode = "isotonic_mlp_far2"
        base.far_iso_x = far_x
        base.far_iso_y = far_y
        base.far_min_edge = e_far
        return base

    if mode == "per_bs_isotonic_mindist":
        mins = np.array(
            [float(np.min(d_hat[:, int(u)])) for u in train_idx], dtype=np.float64
        )
        e1, e2 = float(np.percentile(mins, 33)), float(np.percentile(mins, 66))
        edges = (e1, e2)
        zones = np.array(
            [
                zone_from_min_dist(d_hat[:, int(u)], edges=edges) for u in train_idx
            ],
            dtype=np.int64,
        )
        iso_x_zone: list = [[None, None, None] for _ in range(18)]
        iso_y_zone: list = [[None, None, None] for _ in range(18)]
        iso_x_fb, iso_y_fb = [], []
        for k in range(18):
            xk_fb, yk_fb = _fit_isotonic_1d(dh[k], d_true[k])
            iso_x_fb.append(xk_fb)
            iso_y_fb.append(yk_fb)
            for z in range(3):
                mask = zones == z
                if int(mask.sum()) >= MIN_ZONE_SAMPLES:
                    xk, yk = _fit_isotonic_1d(dh[k, mask], d_true[k, mask])
                else:
                    xk, yk = xk_fb, yk_fb
                iso_x_zone[k][z] = xk
                iso_y_zone[k][z] = yk
        return CalibParams(
            "per_bs_isotonic_mindist",
            zone_edges=np.array(edges, dtype=np.float64),
            iso_x_zone=iso_x_zone,
            iso_y_zone=iso_y_zone,
            iso_x_fb=iso_x_fb,
            iso_y_fb=iso_y_fb,
        )

    if mode == "per_bs_quantile":
        from sklearn.linear_model import QuantileRegressor

        tau = float(quantile_tau) if quantile_tau is not None else 0.35
        models = []
        for k in range(18):
            qr = QuantileRegressor(quantile=tau, alpha=0.01, solver="highs")
            qr.fit(dh[k].reshape(-1, 1), d_true[k])
            models.append(qr)
        return CalibParams(
            "per_bs_quantile", quantile_tau=tau, quantile_models=models
        )

    if mode == "gbr_dist":
        from .gbr_calib import fit_gbr_calib

        return CalibParams("gbr_dist", gbr_bundle=fit_gbr_calib(d_hat, p, bs, train_idx))

    if mode == "pos_mlp":
        from .pos_mlp import fit_pos_mlp

        return CalibParams("pos_mlp", pos_bundle=fit_pos_mlp(d_hat, p, train_idx))

    if mode == "per_bs_isotonic_radial":
        pt = p[:, train_idx]
        centroid = np.array([pt[0].mean(), pt[1].mean()], dtype=np.float64)
        r_ue = np.hypot(pt[0] - centroid[0], pt[1] - centroid[1])
        zones = np.array([_zone_of_r(r) for r in r_ue], dtype=np.int64)
        iso_x_zone: list = [[None, None, None] for _ in range(18)]
        iso_y_zone: list = [[None, None, None] for _ in range(18)]
        iso_x_fb, iso_y_fb = [], []
        for k in range(18):
            xk_fb, yk_fb = _fit_isotonic_1d(dh[k], d_true[k])
            iso_x_fb.append(xk_fb)
            iso_y_fb.append(yk_fb)
            for z in range(3):
                mask = zones == z
                if int(mask.sum()) >= MIN_ZONE_SAMPLES:
                    xk, yk = _fit_isotonic_1d(dh[k, mask], d_true[k, mask])
                else:
                    xk, yk = xk_fb, yk_fb
                iso_x_zone[k][z] = xk
                iso_y_zone[k][z] = yk
        return CalibParams(
            "per_bs_isotonic_radial",
            centroid=centroid,
            zone_edges=np.array(ZONE_EDGES, dtype=np.float64),
            iso_x_zone=iso_x_zone,
            iso_y_zone=iso_y_zone,
            iso_x_fb=iso_x_fb,
            iso_y_fb=iso_y_fb,
        )

    raise ValueError(f"unknown calib mode {mode}")


def _zone_of_r(r: float) -> int:
    if r < ZONE_EDGES[0]:
        return 0
    if r < ZONE_EDGES[1]:
        return 1
    return 2


def _fit_isotonic_1d(x: Array, y: Array) -> tuple[Array, Array]:
    ir = IsotonicRegression(out_of_bounds="clip")
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ir.fit(x, y)
    return (
        np.asarray(ir.X_thresholds_, dtype=np.float64),
        np.asarray(ir.y_thresholds_, dtype=np.float64),
    )


def distance_mae_after_calib(
    d_hat: Array,
    p: Array,
    bs: Array,
    user_idx: Array,
    calib: CalibParams,
) -> float:
    d_true = geometric_distances(p[:, user_idx], bs)
    errs = []
    for j, u in enumerate(user_idx):
        dc = calib.apply(d_hat[:, u])
        errs.append(np.mean(np.abs(dc - d_true[:, j])))
    return float(np.mean(errs))


def _lstsq_affine(x: Array, y: Array) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    a = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _lstsq_quad(x: Array, y: Array) -> Array:
    """Fit y ~ a*x + b*x^2 + c."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    a = np.column_stack([x, x**2, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return np.array([coef[0], coef[1], coef[2]], dtype=np.float64)
