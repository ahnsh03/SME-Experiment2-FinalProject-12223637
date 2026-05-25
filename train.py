"""
Train final localization model (700 UE) and save artifacts for main.py.

Picks best version from outputs/version_catalog.json unless --version set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.cv import (
    fit_full_and_predict,
    run_fold_cv,
    tune_asym_pos_weight,
    tune_huber_f_scale,
    tune_pos_blend_weight,
    tune_quantile_tau,
    tune_top_k,
    tune_weight_gamma,
)
from lib.io_mat import load_dataset, project_root
from lib.pipeline import VERSION_REGISTRY, PipelineConfig

PRODUCTION_VERSION = "v30"


def _pick_best_from_catalog() -> str:
    cat_path = project_root() / "outputs" / "version_catalog.json"
    if not cat_path.exists():
        return PRODUCTION_VERSION
    cat = json.loads(cat_path.read_text(encoding="utf-8"))
    if "best" in cat:
        return cat["best"]["version"]
    rows = cat.get("versions", [])
    if not rows:
        return PRODUCTION_VERSION
    # exclude deprecated gate versions from auto-pick
    rows = [r for r in rows if r.get("version") != "v04" and not r.get("gate")]
    if not rows:
        return PRODUCTION_VERSION
    return min(rows, key=lambda r: r["cv_rmse_m"])["version"]


def _resolve_config(version: str, d_hat, p, bs) -> tuple[PipelineConfig, dict | None]:
    cfg = VERSION_REGISTRY[version]
    extra = None
    if version in ("v09", "v10"):
        tune = tune_huber_f_scale(d_hat, p, bs, cfg)
        cfg = PipelineConfig(
            version=version,
            loss=cfg.loss,
            huber_f_scale=tune["best_huber_f_scale"],
            calib=cfg.calib,
            gate=False,
            description=cfg.description,
        )
        extra = {"f_scale_tuning": tune}
    elif version == "v11":
        tune = tune_weight_gamma(d_hat, p, bs, cfg)
        cfg = PipelineConfig(
            version="v11",
            loss=cfg.loss,
            huber_f_scale=cfg.huber_f_scale,
            calib=cfg.calib,
            gate=False,
            weight_gamma=tune["best_weight_gamma"],
            description=cfg.description,
        )
        extra = {"weight_tuning": tune}
    elif version == "v14":
        tune = tune_asym_pos_weight(d_hat, p, bs, cfg)
        cfg = PipelineConfig(
            version="v14",
            loss=cfg.loss,
            calib=cfg.calib,
            asym_pos_weight=tune["best_asym_pos_weight"],
            description=cfg.description,
        )
        extra = {"asym_tuning": tune}
    elif version == "v15":
        tune = tune_top_k(d_hat, p, bs, cfg)
        cfg = PipelineConfig(
            version="v15",
            loss=cfg.loss,
            calib=cfg.calib,
            top_k=tune["best_top_k"],
            description=cfg.description,
        )
        extra = {"top_k_tuning": tune}
    elif version == "v17":
        tune = tune_weight_gamma(d_hat, p, bs, cfg)
        cfg = PipelineConfig(
            version="v17",
            loss=cfg.loss,
            calib=cfg.calib,
            weight_gamma=tune["best_weight_gamma"],
            description=cfg.description,
        )
        extra = {"weight_tuning": tune}
    elif version == "v20":
        tune = tune_pos_blend_weight(d_hat, p, bs, cfg)
        cfg = PipelineConfig(
            version="v20",
            loss=cfg.loss,
            calib=cfg.calib,
            pos_blend_weight=tune["best_pos_blend_weight"],
            description=cfg.description,
        )
        extra = {"blend_tuning": tune}
    elif version in ("v25", "v26"):
        cfg = VERSION_REGISTRY[version]
        if version == "v25":
            cfg = PipelineConfig(
                version="v25",
                calib="isotonic_mlp",
                weight_gamma=1.0,
                pos_refine_affine=True,
            )
        else:
            cfg = PipelineConfig(
                version="v26",
                calib="isotonic_mlp_far2",
                weight_gamma=1.0,
            )
    elif version == "v30":
        cfg = VERSION_REGISTRY["v30"]
    elif version == "v27":
        tune = tune_quantile_tau(d_hat, p, bs, VERSION_REGISTRY["v27"])
        cfg = PipelineConfig(
            version="v27",
            calib="quantile_mlp",
            weight_gamma=1.0,
            quantile_tau=tune["best_quantile_tau"],
        )
        extra = {"tau_tuning": tune}
    return cfg, extra


def _save_pos_pt(path: Path, calib) -> None:
    if calib.pos_bundle is None:
        return
    import torch
    from lib.pos_mlp import bundle_to_dict

    torch.save(
        {"state_dict": calib.pos_bundle.state_dict, "meta": bundle_to_dict(calib.pos_bundle)},
        path,
    )


def _save_mlp_pt(path: Path, calib) -> None:
    if calib.mlp_bundle is None:
        return
    import torch
    from lib.mlp_calib import bundle_to_dict

    torch.save(
        {
            "state_dict": calib.mlp_bundle.state_dict,
            "meta": bundle_to_dict(calib.mlp_bundle),
        },
        path,
    )


def _save_calib_npz(path: Path, calib) -> None:
    payload = {"mode": calib.mode}
    if calib.quad is not None:
        payload["quad"] = np.asarray(calib.quad)
    if calib.mode in ("global_affine", "per_bs_affine"):
        payload["alpha"] = np.asarray(calib.alpha)
        payload["beta"] = np.asarray(calib.beta)
    if calib.iso_x is not None:
        payload["iso_x"] = np.array(calib.iso_x, dtype=object)
        payload["iso_y"] = np.array(calib.iso_y, dtype=object)
    if calib.centroid is not None:
        payload["centroid"] = np.asarray(calib.centroid)
        payload["zone_edges"] = np.asarray(calib.zone_edges)
        payload["iso_x_zone"] = np.array(calib.iso_x_zone, dtype=object)
        payload["iso_y_zone"] = np.array(calib.iso_y_zone, dtype=object)
    np.savez(path, **payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=str, default=None)
    ap.add_argument("--mat", type=str, default=None)
    args = ap.parse_args()

    version = args.version or _pick_best_from_catalog()
    if version not in VERSION_REGISTRY:
        version = PRODUCTION_VERSION

    data = load_dataset(args.mat)
    d_hat, p, bs = data["d_hat"], data["p"], data["bs"]
    cfg, extra = _resolve_config(version, d_hat, p, bs)

    calib, p_hat, fit_summary = fit_full_and_predict(d_hat, p, bs, cfg)
    cv_check = run_fold_cv(d_hat, p, bs, cfg)

    out = project_root()
    config = {
        "production_version": version,
        "pipeline": {
            "loss": cfg.loss,
            "huber_f_scale": cfg.huber_f_scale,
            "calib": cfg.calib,
            "gate": False,
            "weight_gamma": cfg.weight_gamma,
            "asym_pos_weight": cfg.asym_pos_weight,
            "top_k": cfg.top_k,
            "pos_blend_weight": cfg.pos_blend_weight,
            "pos_refine_affine": cfg.pos_refine_affine,
            "quantile_tau": cfg.quantile_tau,
        },
        "calib": calib.to_dict(),
        "cv_oof_rmse_m": cv_check["oof_position"]["rmse_m"],
        "train_fit_rmse_m": fit_summary.get("train_fit", {}).get("rmse_m"),
        "tuning": extra,
    }
    (out / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _save_calib_npz(out / "model_calib.npz", calib)
    if calib.mlp_bundle is not None:
        _save_mlp_pt(out / "model_mlp.pt", calib)
    if calib.pos_bundle is not None:
        _save_pos_pt(out / "model_pos.pt", calib)

    pred_path = out / "outputs" / f"{version}_train700_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame({"pred_x": p_hat[0], "pred_y": p_hat[1], "true_x": p[0], "true_y": p[1]})
    if data["indices"] is not None:
        df.insert(0, "index", data["indices"])
    df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print(f"train.py done version={version}")
    print(f"  CV OOF RMSE = {config['cv_oof_rmse_m']:.3f} m")
    print(f"  train-fit RMSE = {config['train_fit_rmse_m']:.3f} m")
    print(f"  saved config.json, model_calib.npz")
    if calib.mlp_bundle is not None:
        print(f"  saved model_mlp.pt")


if __name__ == "__main__":
    main()
