"""K-Fold CV with per-stage RMSE logging."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .calib import CalibParams, distance_mae_after_calib, fit_calib
from .metrics import position_errors, summarize_errors
from .pipeline import PipelineConfig, localize_user

Array = np.ndarray
CV_SEED = 42
N_SPLITS = 5

# Versions excluded from production (deprecated)
SKIP_IN_CATALOG_DEFAULT = {"v04"}


def clone_pipeline_cfg(base: PipelineConfig, **kwargs) -> PipelineConfig:
    """Preserve v25 options (pos_refine_affine, zone weights, etc.) when tuning."""
    kw = dict(kwargs)
    kw.setdefault("gate", False)
    return replace(base, **kw)


def run_fold_cv(
    d_hat: Array,
    p: Array,
    bs: Array,
    cfg: PipelineConfig,
    *,
    n_splits: int = N_SPLITS,
    seed: int = CV_SEED,
) -> Dict[str, Any]:
    n = d_hat.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows = []
    oof_err = np.full(n, np.nan)
    stage_err_lists: Dict[str, List[float]] = {}
    dist_mae_list: List[float] = []

    for fold_id, (tr, va) in enumerate(kf.split(np.arange(n))):
        train_idx = np.asarray(tr, dtype=np.int64)
        val_idx = np.asarray(va, dtype=np.int64)
        calib = fit_calib(
            d_hat,
            p,
            bs,
            train_idx,
            cfg.calib,
            quantile_tau=cfg.quantile_tau,
            mlp_noise_std=cfg.mlp_noise_std,
        )
        if cfg.pos_blend_weight is not None and cfg.calib != "pos_mlp":
            from .pos_mlp import fit_pos_mlp

            calib.pos_bundle = fit_pos_mlp(d_hat, p, train_idx, epochs=100)

        from .spatial import bounds_from_positions

        if cfg.pos_refine_affine and cfg.calib != "pos_mlp":
            from .post_refine import apply_pos_affine, fit_pos_affine

            xy_bounds_pre = bounds_from_positions(p[:, train_idx])
            p_tr = np.zeros((2, len(train_idx)), dtype=np.float64)
            cfg_no_aff = clone_pipeline_cfg(cfg, pos_refine_affine=False)
            mins = np.array(
                [float(np.min(d_hat[:, int(u)])) for u in train_idx], dtype=np.float64
            )
            far_edge = float(np.percentile(mins, 66))
            for j, u in enumerate(train_idx):
                xy, _ = localize_user(
                    d_hat[:, u],
                    bs,
                    calib,
                    cfg_no_aff,
                    xy_bounds=xy_bounds_pre,
                )
                p_tr[:, j] = xy
            calib.pos_affine = fit_pos_affine(p_tr, p[:, train_idx])
            if cfg.pos_refine_far:
                far_mask = mins >= far_edge
                if int(far_mask.sum()) >= 30:
                    p1 = np.zeros((2, int(far_mask.sum())), dtype=np.float64)
                    p_true_f = p[:, train_idx[far_mask]]
                    for j, idx in enumerate(np.where(far_mask)[0]):
                        u = int(train_idx[idx])
                        p1[:, j] = apply_pos_affine(p_tr[:, idx], calib.pos_affine)
                    calib.pos_affine_far = fit_pos_affine(p1, p_true_f)
                    calib.far_min_edge = far_edge

        xy_bounds = None
        if cfg.calib != "pos_mlp":
            xy_bounds = bounds_from_positions(p[:, train_idx])

        dm = None
        if cfg.calib != "pos_mlp":
            dm = distance_mae_after_calib(d_hat, p, bs, val_idx, calib)
            dist_mae_list.append(dm)

        preds = np.zeros((2, len(val_idx)))
        for j, u in enumerate(val_idx):
            from .io_mat import geometric_distances

            d_true_u = geometric_distances(p[:, [u]], bs)[:, 0]
            xy, lg = localize_user(
                d_hat[:, u],
                bs,
                calib,
                cfg,
                p_true_u=p[:, u],
                d_true_u=d_true_u,
                xy_bounds=xy_bounds,
            )
            preds[:, j] = xy
            for k, v in lg.rmse_if_stop_at.items():
                stage_err_lists.setdefault(k, []).append(v)

        err = position_errors(p[:, val_idx], preds)
        oof_err[val_idx] = err
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "val_rmse_m": float(np.sqrt(np.mean(err**2))),
                "val_dist_mae_after_calib_m": dm,
            }
        )

    oof_summary = summarize_errors(oof_err)
    stage_summary = {}
    for k, vals in stage_err_lists.items():
        if vals:
            stage_summary[k] = summarize_errors(np.asarray(vals))

    return {
        "version": cfg.version,
        "description": cfg.description,
        "config": {
            "loss": cfg.loss,
            "huber_f_scale": cfg.huber_f_scale,
            "calib": cfg.calib,
            "gate": cfg.gate,
            "gate_tau": cfg.gate_tau,
            "top_k": cfg.top_k,
            "weight_gamma": cfg.weight_gamma,
            "asym_pos_weight": cfg.asym_pos_weight,
            "pos_blend_weight": cfg.pos_blend_weight,
        },
        "cv": {"n_splits": n_splits, "seed": seed},
        "oof_position": oof_summary,
        "oof_stage_position": stage_summary,
        "oof_dist_mae_after_calib_mean_m": float(np.mean(dist_mae_list)) if dist_mae_list else None,
        "folds": fold_rows,
    }


def tune_huber_f_scale(
    d_hat: Array,
    p: Array,
    bs: Array,
    base_cfg: PipelineConfig,
    scales: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if scales is None:
        scales = [0.5, 0.7, 0.8, 1.0, 1.2, 1.35, 1.5, 2.0]

    rows = []
    best_scale = base_cfg.huber_f_scale
    best_rmse = float("inf")
    for fs in scales:
        cfg = clone_pipeline_cfg(base_cfg, huber_f_scale=fs)
        res = run_fold_cv(d_hat, p, bs, cfg)
        rmse = res["oof_position"]["rmse_m"]
        rows.append({"huber_f_scale": fs, "cv_rmse_m": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_scale = fs

    return {
        "grid": rows,
        "best_huber_f_scale": best_scale,
        "best_cv_rmse_m": best_rmse,
    }


def tune_weight_gamma(
    d_hat: Array,
    p: Array,
    bs: Array,
    base_cfg: PipelineConfig,
    gammas: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if gammas is None:
        gammas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    rows = []
    best_g = 0.0
    best_rmse = float("inf")
    for g in gammas:
        cfg = clone_pipeline_cfg(
            base_cfg, weight_gamma=float(g) if g > 0 else None
        )
        res = run_fold_cv(d_hat, p, bs, cfg)
        rmse = res["oof_position"]["rmse_m"]
        rows.append({"weight_gamma": g, "cv_rmse_m": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_g = float(g)

    return {
        "grid": rows,
        "best_weight_gamma": best_g if best_g > 0 else None,
        "best_cv_rmse_m": best_rmse,
    }


def tune_asym_pos_weight(
    d_hat: Array,
    p: Array,
    bs: Array,
    base_cfg: PipelineConfig,
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if weights is None:
        weights = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]

    rows = []
    best_w = 1.0
    best_rmse = float("inf")
    for w in weights:
        cfg = clone_pipeline_cfg(base_cfg, asym_pos_weight=float(w))
        res = run_fold_cv(d_hat, p, bs, cfg)
        rmse = res["oof_position"]["rmse_m"]
        rows.append({"asym_pos_weight": w, "cv_rmse_m": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = float(w)

    return {
        "grid": rows,
        "best_asym_pos_weight": best_w,
        "best_cv_rmse_m": best_rmse,
    }


def tune_top_k(
    d_hat: Array,
    p: Array,
    bs: Array,
    base_cfg: PipelineConfig,
    ks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if ks is None:
        ks = [3, 4, 5, 6, 8, 10, 12, 18]

    rows = []
    best_k = 18
    best_rmse = float("inf")
    for k in ks:
        cfg = clone_pipeline_cfg(base_cfg, top_k=int(k) if k < 18 else None)
        res = run_fold_cv(d_hat, p, bs, cfg)
        rmse = res["oof_position"]["rmse_m"]
        rows.append({"top_k": k, "cv_rmse_m": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_k = int(k)

    return {
        "grid": rows,
        "best_top_k": best_k if best_k < 18 else None,
        "best_cv_rmse_m": best_rmse,
    }


def tune_pos_blend_weight(
    d_hat: Array,
    p: Array,
    bs: Array,
    base_cfg: PipelineConfig,
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if weights is None:
        weights = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    rows = []
    best_w = 0.7
    best_rmse = float("inf")
    for w in weights:
        cfg = clone_pipeline_cfg(base_cfg, pos_blend_weight=float(w))
        res = run_fold_cv(d_hat, p, bs, cfg)
        rmse = res["oof_position"]["rmse_m"]
        rows.append({"pos_blend_weight": w, "cv_rmse_m": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = float(w)

    return {
        "grid": rows,
        "best_pos_blend_weight": best_w,
        "best_cv_rmse_m": best_rmse,
    }


def tune_quantile_tau(
    d_hat: Array,
    p: Array,
    bs: Array,
    base_cfg: PipelineConfig,
    taus: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if taus is None:
        taus = [0.25, 0.30, 0.35, 0.40, 0.45]

    rows = []
    best_tau = 0.35
    best_rmse = float("inf")
    for tau in taus:
        cfg = clone_pipeline_cfg(base_cfg, quantile_tau=float(tau))
        res = run_fold_cv(d_hat, p, bs, cfg)
        rmse = res["oof_position"]["rmse_m"]
        rows.append({"quantile_tau": tau, "cv_rmse_m": rmse})
        if rmse < best_rmse:
            best_rmse = rmse
            best_tau = float(tau)

    return {
        "grid": rows,
        "best_quantile_tau": best_tau,
        "best_cv_rmse_m": best_rmse,
    }


def fit_full_and_predict(
    d_hat: Array,
    p: Array,
    bs: Array,
    cfg: PipelineConfig,
    train_all: bool = True,
) -> tuple[CalibParams, Array, Dict[str, Any]]:
    from .pipeline import localize_batch

    n = d_hat.shape[1]
    idx = np.arange(n, dtype=np.int64)
    calib = fit_calib(
        d_hat,
        p,
        bs,
        idx,
        cfg.calib,
        quantile_tau=cfg.quantile_tau,
        mlp_noise_std=cfg.mlp_noise_std,
    )
    if cfg.pos_blend_weight is not None and cfg.calib != "pos_mlp":
        from .pos_mlp import fit_pos_mlp

        calib.pos_bundle = fit_pos_mlp(d_hat, p, idx, epochs=120)
    if cfg.pos_refine_affine and cfg.calib != "pos_mlp":
        from .post_refine import apply_pos_affine, fit_pos_affine
        from .spatial import bounds_from_positions

        xy_bounds = bounds_from_positions(p)
        cfg_no = clone_pipeline_cfg(cfg, pos_refine_affine=False)
        mins = np.array([float(np.min(d_hat[:, u])) for u in range(n)], dtype=np.float64)
        far_edge = float(np.percentile(mins, 66))
        p_tr = np.zeros((2, n), dtype=np.float64)
        for u in range(n):
            xy, _ = localize_user(d_hat[:, u], bs, calib, cfg_no, xy_bounds=xy_bounds)
            p_tr[:, u] = xy
        calib.pos_affine = fit_pos_affine(p_tr, p)
        if cfg.pos_refine_far:
            far_mask = mins >= far_edge
            if int(far_mask.sum()) >= 30:
                p1 = np.zeros((2, int(far_mask.sum())), dtype=np.float64)
                for j, u in enumerate(np.where(far_mask)[0]):
                    p1[:, j] = apply_pos_affine(p_tr[:, u], calib.pos_affine)
                calib.pos_affine_far = fit_pos_affine(p1, p[:, far_mask])
                calib.far_min_edge = far_edge
    p_hat, logs = localize_batch(d_hat, bs, calib, cfg, p=p if train_all else None)
    summary: Dict[str, Any] = {
        "n_user": n,
        "calib": calib.to_dict(),
        "config": cfg.version,
    }
    if train_all:
        err = position_errors(p, p_hat)
        summary["train_fit"] = summarize_errors(err)
    return calib, p_hat, summary


def save_version_outputs(
    out_dir: Path,
    version: str,
    cv_result: Dict[str, Any],
    *,
    predictions: Optional[Array] = None,
    indices: Optional[Array] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{version}_summary.json").write_text(
        json.dumps(cv_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(cv_result["folds"]).to_csv(
        out_dir / f"{version}_cv_folds.csv", index=False, encoding="utf-8-sig"
    )
    if predictions is not None:
        df = pd.DataFrame(
            {
                "user": np.arange(predictions.shape[1]),
                "pred_x": predictions[0],
                "pred_y": predictions[1],
            }
        )
        if indices is not None:
            df.insert(1, "index", indices)
        df.to_csv(
            out_dir / f"{version}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
