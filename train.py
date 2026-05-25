"""
Train v30 on labeled 700 UE; save model_mlp.pt consumed by main.py.

Report / novelty grading (paired with report.md):
  - import main as M guarantees train.py and main.py describe the same v30 stack.
  - 5-Fold CV (seed=42) prints OOF RMSE for fair ablation; fold calib never sees val UE labels.
  - Production artifact: model_mlp.pt embeds pipeline + Isotonic + MLP + pos_affine (no config.json).

Simplification vs full dev lib/cv.py (CV logging only; shipped model from 700-UE fit):
  - No per-fold xy_bounds on affine pass (main uses fixed global bounds at inference anyway).
  - No stage-wise OOF diagnostics / dist MAE tables; core fit order matches lib fit_full_and_predict.
  - asym_pos_weight fixed at 5.0 (grid result in report); MLP epochs=100 on isotonic residuals.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

import main as M

Array = np.ndarray
CV_SEED = 42
N_SPLITS = 5
PRODUCTION_VERSION = "v30"


def _fit_isotonic_1d(x: Array, y: Array) -> tuple[Array, Array]:
    ir = IsotonicRegression(out_of_bounds="clip")
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    ir.fit(x, y)
    return (
        np.asarray(ir.X_thresholds_, dtype=np.float64),
        np.asarray(ir.y_thresholds_, dtype=np.float64),
    )


def fit_isotonic_per_bs(d_hat: Array, p: Array, bs: Array, train_idx: Array) -> tuple[list, list]:
    d_true = M.geometric_distances(p[:, train_idx], bs)
    dh = d_hat[:, train_idx]
    iso_x, iso_y = [], []
    for k in range(18):
        xk, yk = _fit_isotonic_1d(dh[k], d_true[k])
        iso_x.append(xk)
        iso_y.append(yk)
    return iso_x, iso_y


def fit_mlp_on_iso(
    d_hat: Array,
    p: Array,
    bs: Array,
    train_idx: Array,
    iso_x: list,
    iso_y: list,
    *,
    epochs: int = 100,
) -> M.MLPBundle:
    d_true = M.geometric_distances(p[:, train_idx], bs)
    rows, targets = [], []
    for j, u in enumerate(train_idx):
        d = np.asarray(d_hat[:, int(u)], dtype=np.float64)
        iso = np.empty(18, dtype=np.float64)
        for k in range(18):
            xk = np.asarray(iso_x[k], dtype=np.float64)
            yk = np.asarray(iso_y[k], dtype=np.float64)
            iso[k] = np.interp(d[k], xk, yk, left=yk[0], right=yk[-1])
        rows.append(iso)
        targets.append(d_true[:, j])
    dh = np.stack(rows, axis=1).T
    y = np.stack(targets, axis=1).T
    in_mean = dh.mean(axis=0)
    in_std = np.maximum(dh.std(axis=0), 1e-3)
    return _fit_mlp_arrays(dh, y, in_mean, in_std, epochs=epochs)


def _fit_mlp_arrays(
    dh: Array,
    y: Array,
    in_mean: Array,
    in_std: Array,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    hidden: int = 64,
    dropout: float = 0.25,
    seed: int = 42,
) -> M.MLPBundle:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = M.DistMLP(hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xn = (dh - in_mean) / in_std
    xt = torch.from_numpy(xn.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    t_std = torch.from_numpy(in_std.astype(np.float32))
    t_mean = torch.from_numpy(in_mean.astype(np.float32))
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(xt) * t_std + t_mean
        loss = nn.functional.mse_loss(pred, yt)
        loss.backward()
        opt.step()
    return M.MLPBundle(
        state_dict=model.state_dict(),
        in_mean=in_mean,
        in_std=in_std,
        hidden=hidden,
        dropout=dropout,
    )


def fit_pos_affine(p_hat: Array, p_true: Array) -> Array:
    n = p_hat.shape[1]
    X = np.column_stack([p_hat[0], p_hat[1], np.ones(n, dtype=np.float64)])
    coef, *_ = np.linalg.lstsq(X, p_true.T, rcond=None)
    return np.asarray(coef.T, dtype=np.float64)


def fit_calib_v30(d_hat: Array, p: Array, bs: Array, train_idx: Array) -> M.CalibParams:
    iso_x, iso_y = fit_isotonic_per_bs(d_hat, p, bs, train_idx)
    bundle = fit_mlp_on_iso(d_hat, p, bs, train_idx, iso_x, iso_y)
    return M.CalibParams("isotonic_mlp", iso_x=iso_x, iso_y=iso_y, mlp_bundle=bundle)


def v30_pipeline_cfg() -> M.PipelineConfig:
    return M.PipelineConfig(
        huber_f_scale=1.0,
        weight_gamma=1.0,
        asym_pos_weight=5.0,
        pos_refine_affine=True,
    )


def fit_affine_on_train(
    d_hat: Array,
    p: Array,
    bs: Array,
    train_idx: Array,
    calib: M.CalibParams,
    cfg: M.PipelineConfig,
) -> M.CalibParams:
    cfg_no = replace(cfg, pos_refine_affine=False)
    p_tr = np.zeros((2, len(train_idx)), dtype=np.float64)
    for j, u in enumerate(train_idx):
        p_tr[:, j] = M.localize_user(d_hat[:, int(u)], bs, calib, cfg_no)
    calib.pos_affine = fit_pos_affine(p_tr, p[:, train_idx])
    return calib


def run_fold_cv(d_hat: Array, p: Array, bs: Array, cfg: M.PipelineConfig) -> dict:
    """OOF evaluation: each fold retrains Isotonic, MLP, and affine only on train_idx."""
    n = d_hat.shape[1]
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    oof_err = np.full(n, np.nan)
    fold_rows = []

    for fold_id, (tr, va) in enumerate(kf.split(np.arange(n))):
        train_idx = np.asarray(tr, dtype=np.int64)
        val_idx = np.asarray(va, dtype=np.int64)
        calib = fit_calib_v30(d_hat, p, bs, train_idx)
        calib = fit_affine_on_train(d_hat, p, bs, train_idx, calib, cfg)
        preds = np.zeros((2, len(val_idx)))
        for j, u in enumerate(val_idx):
            preds[:, j] = M.localize_user(d_hat[:, int(u)], bs, calib, cfg)
        err = np.hypot(p[0, val_idx] - preds[0], p[1, val_idx] - preds[1])
        oof_err[val_idx] = err
        fold_rows.append(
            {
                "fold": fold_id,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "val_rmse_m": float(np.sqrt(np.mean(err**2))),
            }
        )

    err_all = oof_err[~np.isnan(oof_err)]
    return {
        "oof_position": {
            "rmse_m": float(np.sqrt(np.mean(err_all**2))),
            "median_m": float(np.median(err_all)),
            "p90_m": float(np.percentile(err_all, 90)),
            "n": int(err_all.size),
        },
        "folds": fold_rows,
    }


def calib_to_dict(calib: M.CalibParams) -> dict:
    d: dict = {
        "mode": calib.mode,
        "iso_x": [np.asarray(x).tolist() for x in calib.iso_x],
        "iso_y": [np.asarray(y).tolist() for y in calib.iso_y],
    }
    if calib.pos_affine is not None:
        d["pos_affine"] = np.asarray(calib.pos_affine).tolist()
    return d


def save_model_bundle(
    path: Path,
    calib: M.CalibParams,
    cfg: M.PipelineConfig,
    *,
    state_dict: dict,
    meta: dict,
) -> None:
    """Single model.* file for ML submission rule; main.py loads this bundle only."""
    import torch

    assert calib.mlp_bundle is not None
    torch.save(
        {
            "state_dict": state_dict,
            "meta": meta,
            "production_version": PRODUCTION_VERSION,
            "pipeline": {
                "loss": cfg.loss,
                "huber_f_scale": cfg.huber_f_scale,
                "calib": "isotonic_mlp",
                "weight_gamma": cfg.weight_gamma,
                "asym_pos_weight": cfg.asym_pos_weight,
                "pos_refine_affine": cfg.pos_refine_affine,
            },
            "calib": calib_to_dict(calib),
        },
        path,
    )


def load_train_mat(path: str | None) -> tuple[Array, Array, Array]:
    mat_path = Path(path) if path else Path("DH_FR1.mat")
    if not mat_path.exists():
        alt = Path(__file__).resolve().parent.parent / "data" / "InF_DH_FR1.mat"
        if alt.exists():
            mat_path = alt
    raw = sio.loadmat(str(mat_path), squeeze_me=True)
    p = np.asarray(raw["p"], dtype=np.float64)
    d_hat = np.asarray(raw["d_hat"], dtype=np.float64)
    bs_key = "p_bs" if "p_bs" in raw else "BS_positions"
    bs = np.asarray(raw[bs_key], dtype=np.float64)
    if d_hat.shape[0] != 18:
        d_hat = d_hat.T
    if bs.shape[0] != 2:
        bs = bs.T
    return d_hat, p, bs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, default=None)
    args = ap.parse_args()

    d_hat, p, bs = load_train_mat(args.mat)
    cfg = v30_pipeline_cfg()
    n = d_hat.shape[1]
    idx = np.arange(n, dtype=np.int64)

    # Full 700-UE fit -> artifact used by hidden-300 inference (no GT at main.py runtime).
    calib = fit_calib_v30(d_hat, p, bs, idx)
    calib = fit_affine_on_train(d_hat, p, bs, idx, calib, cfg)
    cv = run_fold_cv(d_hat, p, bs, cfg)

    bundle = calib.mlp_bundle
    assert bundle is not None
    meta = {
        "in_mean": bundle.in_mean.tolist(),
        "in_std": bundle.in_std.tolist(),
        "hidden": bundle.hidden,
        "dropout": bundle.dropout,
    }
    out = Path(__file__).resolve().parent
    save_model_bundle(
        out / "model_mlp.pt",
        calib,
        cfg,
        state_dict=bundle.state_dict,
        meta=meta,
    )

    p_hat = M.localize_batch(d_hat, bs, calib, cfg)
    train_rmse = float(np.sqrt(np.mean(np.hypot(p[0] - p_hat[0], p[1] - p_hat[1]) ** 2)))

    print(f"train.py done version={PRODUCTION_VERSION}")
    print(f"  CV OOF RMSE = {cv['oof_position']['rmse_m']:.3f} m")
    print(f"  train-fit RMSE = {train_rmse:.3f} m")
    print(f"  saved model_mlp.pt (pipeline + calib bundled)")


if __name__ == "__main__":
    main()
