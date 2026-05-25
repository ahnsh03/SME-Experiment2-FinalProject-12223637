"""
Final project inference — grading entry point.

Loads DH_FR1.mat from cwd, applies trained calibration + Huber trilat.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from lib.calib import CalibParams
from lib.io_mat import load_dataset
from lib.pipeline import PipelineConfig, localize_batch


def _attach_pos_bundle(calib: CalibParams, root: Path) -> CalibParams:
    pos_path = root / "model_pos.pt"
    if calib.mode == "pos_mlp" and not pos_path.exists():
        raise FileNotFoundError(f"{pos_path} required for pos_mlp")
    if not pos_path.exists():
        return calib
    import torch
    from lib.pos_mlp import bundle_from_dict

    z = torch.load(pos_path, map_location="cpu", weights_only=False)
    calib.pos_bundle = bundle_from_dict(z["meta"], z["state_dict"])
    return calib


def _attach_mlp_bundle(calib: CalibParams, root: Path) -> CalibParams:
    if calib.mode not in ("mlp_dist", "isotonic_mlp"):
        return calib
    mlp_path = root / "model_mlp.pt"
    if not mlp_path.exists():
        raise FileNotFoundError(f"{mlp_path} required for {calib.mode}")
    import torch
    from lib.mlp_calib import bundle_from_dict

    z = torch.load(mlp_path, map_location="cpu", weights_only=False)
    bundle = bundle_from_dict(z["meta"], z["state_dict"])
    calib.mlp_bundle = bundle
    return calib


def _load_production_config(root: Path) -> tuple[PipelineConfig, CalibParams]:
    cfg_path = root / "config.json"
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        pipe = raw["pipeline"]
        calib = CalibParams.from_dict(raw["calib"])
        calib = _attach_mlp_bundle(calib, root)
        calib = _attach_pos_bundle(calib, root)
        wg = pipe.get("weight_gamma")
        pcfg = PipelineConfig(
            version=raw.get("production_version", "v02"),
            loss=pipe.get("loss", "huber"),
            huber_f_scale=float(pipe.get("huber_f_scale", 1.0)),
            calib=pipe.get("calib", "global_affine"),
            gate=False,
            weight_gamma=float(wg) if wg is not None else None,
            asym_pos_weight=float(pipe.get("asym_pos_weight", 1.0)),
            top_k=pipe.get("top_k"),
            pos_blend_weight=pipe.get("pos_blend_weight"),
            pos_refine_affine=bool(pipe.get("pos_refine_affine", False)),
        )
        return pcfg, calib

    npz_path = root / "model_calib.npz"
    if npz_path.exists():
        z = np.load(npz_path, allow_pickle=True)
        mode = str(z["mode"])
        kw = {"mode": mode}
        if "quad" in z.files:
            kw["quad"] = z["quad"]
        if "alpha" in z.files:
            kw["alpha"] = z["alpha"]
            kw["beta"] = z["beta"]
        if "iso_x" in z.files:
            kw["iso_x"] = list(z["iso_x"])
            kw["iso_y"] = list(z["iso_y"])
        if "centroid" in z.files:
            kw["centroid"] = z["centroid"]
            kw["zone_edges"] = z["zone_edges"]
            kw["iso_x_zone"] = list(z["iso_x_zone"])
            kw["iso_y_zone"] = list(z["iso_y_zone"])
        calib = CalibParams(**kw)
        pcfg = PipelineConfig(version="v02", loss="huber", calib=mode, gate=False)
        return pcfg, calib

    raise FileNotFoundError("config.json or model_calib.npz required — run train.py first")


def your_algorithm(d_hat_u: np.ndarray, p_bs: np.ndarray) -> np.ndarray:
    """Per-user localization (used when config not preloaded)."""
    root = Path(__file__).resolve().parent
    pcfg, calib = _load_production_config(root)
    p_hat, _ = localize_batch(
        d_hat_u.reshape(18, 1),
        p_bs,
        calib,
        pcfg,
    )
    return p_hat[:, 0]


def main() -> np.ndarray:
    mat_path = Path("DH_FR1.mat")
    if not mat_path.exists():
        mat_path = Path(__file__).resolve().parent / "data" / "InF_DH_FR1.mat"

    data = load_dataset(mat_path)
    d_hat = data["d_hat"]
    bs = data["bs"]
    num_user = d_hat.shape[1]

    root = Path(__file__).resolve().parent
    pcfg, calib = _load_production_config(root)
    p_hat, _ = localize_batch(d_hat, bs, calib, pcfg)

    return np.asarray(p_hat, dtype=np.float64)


if __name__ == "__main__":
    result = main()
    print(f"main() returned shape {result.shape}")
