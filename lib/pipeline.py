"""Versioned localization pipeline with per-stage diagnostics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .calib import CalibParams, CalibMode, distance_mae_after_calib, fit_calib
from .gate import apply_hard_gate, geometric_residuals
from .metrics import position_errors, summarize_errors
from .post_refine import apply_pos_affine
from .spatial import clip_xy, zone_from_min_dist
from .trilat import LossKind, bounds_from_positions, huber_trilat

Array = np.ndarray


@dataclass
class PipelineConfig:
    version: str
    loss: LossKind = "huber"
    huber_f_scale: float = 1.0
    calib: CalibMode = "none"
    gate: bool = False
    gate_tau: Optional[float] = None
    top_k: Optional[int] = None  # shortest d_hat channels (applied before calib)
    weight_gamma: Optional[float] = None  # soft 1/d^gamma weights in trilat (not HardGate)
    asym_pos_weight: float = 1.0  # penalize pred > d_obs (NLOS-consistent)
    pos_blend_weight: Optional[float] = None  # p = w*trilat + (1-w)*pos_mlp(d_hat)
    quantile_tau: Optional[float] = None
    mlp_noise_std: float = 0.0
    zone_near_boost: float = 1.0
    zone_far_scale: float = 1.0
    pos_refine_affine: bool = False
    pos_refine_far: bool = False  # far UE: 2nd affine after global
    description: str = ""

    def stages_enabled(self) -> List[str]:
        s = ["trilat_raw"]
        if self.calib != "none":
            s.append("calib")
        if self.top_k is not None:
            s.append(f"top_k_{self.top_k}")
        s.append("trilat_main")
        if self.gate:
            s.append("gate_refine")
        return s


@dataclass
class UserStageLog:
    dist_mae_after_calib: Optional[float] = None
    n_anchors_main: int = 18
    n_anchors_gated: Optional[int] = None
    n_dropped: int = 0
    gate_applied: bool = False
    rmse_if_stop_at: Dict[str, float] = field(default_factory=dict)


def _select_top_k(d_hat_u: Array, bs: Array, k: int) -> tuple[Array, Array, Array]:
    order = np.argsort(d_hat_u)[:k]
    return d_hat_u[order], bs[:, order], order


def localize_user(
    d_hat_u: Array,
    bs: Array,
    calib: CalibParams,
    cfg: PipelineConfig,
    *,
    p_true_u: Optional[Array] = None,
    d_true_u: Optional[Array] = None,
    xy_bounds: Optional[tuple] = None,
) -> tuple[Array, UserStageLog]:
    log = UserStageLog()
    d_work = np.asarray(d_hat_u, dtype=np.float64).ravel()
    bs_work = np.asarray(bs, dtype=np.float64)
    if bs_work.shape[0] != 2:
        bs_work = bs_work.T

    if calib.mode == "pos_mlp":
        assert calib.pos_bundle is not None
        p_out = calib.pos_bundle.apply(d_work)
        if p_true_u is not None:
            log.rmse_if_stop_at["after_pos_mlp"] = float(
                np.hypot(p_out[0] - p_true_u[0], p_out[1] - p_true_u[1])
            )
        return p_out, log

    wg = cfg.weight_gamma
    apw = cfg.asym_pos_weight
    znb = cfg.zone_near_boost
    zfs = cfg.zone_far_scale

    # Stage: raw trilat (diagnostic — same anchors as main input)
    p_raw, _ = huber_trilat(
        d_work,
        bs_work,
        loss=cfg.loss,
        f_scale=cfg.huber_f_scale,
        weight_gamma=wg,
        asym_pos_weight=apw,
        xy_bounds=xy_bounds,
        zone_near_boost=znb,
        zone_far_scale=zfs,
    )
    if p_true_u is not None:
        log.rmse_if_stop_at["after_raw_trilat"] = float(
            np.hypot(p_raw[0] - p_true_u[0], p_raw[1] - p_true_u[1])
        )

    if calib.mode == "per_bs_isotonic_radial":
        assert calib.centroid is not None and calib.iso_x_fb is not None
        d_pre = calib.apply_fallback(d_work)
        p0, _ = huber_trilat(
            d_pre,
            bs_work,
            loss=cfg.loss,
            f_scale=cfg.huber_f_scale,
            weight_gamma=wg,
            asym_pos_weight=apw,
            xy_bounds=xy_bounds,
            zone_near_boost=znb,
            zone_far_scale=zfs,
        )
        r_ue = float(np.hypot(p0[0] - calib.centroid[0], p0[1] - calib.centroid[1]))
        z = calib.zone_index(r_ue)
        d_corr = calib.apply(d_work, zone=z)
    elif calib.mode in ("per_bs_isotonic_mindist", "isotonic_mlp_mindist"):
        e = (
            tuple(float(x) for x in calib.zone_edges)
            if calib.zone_edges is not None
            else None
        )
        z = zone_from_min_dist(d_work, edges=e) if e else zone_from_min_dist(d_work)
        d_corr = calib.apply(d_work, zone=z)
    else:
        d_corr = calib.apply(d_work)

    sel_idx = None
    if cfg.top_k is not None:
        d_corr, bs_work, sel_idx = _select_top_k(d_corr, bs_work, cfg.top_k)
        log.n_anchors_main = cfg.top_k

    if p_true_u is not None and d_true_u is not None:
        d_ref = d_true_u[sel_idx] if sel_idx is not None else d_true_u
        log.dist_mae_after_calib = float(np.mean(np.abs(d_corr - d_ref)))

    p_main, meta = huber_trilat(
        d_corr,
        bs_work,
        loss=cfg.loss,
        f_scale=cfg.huber_f_scale,
        weight_gamma=wg,
        asym_pos_weight=apw,
        xy_bounds=xy_bounds,
        zone_near_boost=znb,
        zone_far_scale=zfs,
        zone_ref=d_work,
    )
    log.n_anchors_main = int(meta.get("n_anchors", d_corr.size))
    if p_true_u is not None:
        log.rmse_if_stop_at["after_main_trilat"] = float(
            np.hypot(p_main[0] - p_true_u[0], p_main[1] - p_true_u[1])
        )

    p_out = p_main
    if cfg.pos_blend_weight is not None and calib.pos_bundle is not None:
        w = float(cfg.pos_blend_weight)
        p_pos = calib.pos_bundle.apply(d_work)
        p_out = w * p_main + (1.0 - w) * p_pos
        if p_true_u is not None:
            log.rmse_if_stop_at["after_pos_blend"] = float(
                np.hypot(p_out[0] - p_true_u[0], p_out[1] - p_true_u[1])
            )
    if calib.pos_affine is not None:
        p_out = apply_pos_affine(p_out, calib.pos_affine)
        if (
            cfg.pos_refine_far
            and calib.pos_affine_far is not None
            and calib.far_min_edge is not None
            and float(np.min(d_work)) >= float(calib.far_min_edge)
        ):
            p_out = apply_pos_affine(p_out, calib.pos_affine_far)
        p_out = clip_xy(p_out)
        if p_true_u is not None:
            log.rmse_if_stop_at["after_pos_affine"] = float(
                np.hypot(p_out[0] - p_true_u[0], p_out[1] - p_true_u[1])
            )

    if cfg.gate and cfg.gate_tau is not None:
        d_g, bs_g, n_drop = apply_hard_gate(d_corr, bs_work, p_main, cfg.gate_tau)
        log.n_dropped = n_drop
        log.gate_applied = n_drop > 0
        if d_g.size >= 4 and n_drop > 0:
            p_g, meta_g = huber_trilat(
                d_g, bs_g, loss=cfg.loss, f_scale=cfg.huber_f_scale, x0=p_main
            )
            p_out = p_g
            log.n_anchors_gated = int(meta_g.get("n_anchors", d_g.size))
        if p_true_u is not None:
            log.rmse_if_stop_at["after_gate"] = float(
                np.hypot(p_out[0] - p_true_u[0], p_out[1] - p_true_u[1])
            )

    return p_out, log


def localize_batch(
    d_hat: Array,
    bs: Array,
    calib: CalibParams,
    cfg: PipelineConfig,
    user_indices: Optional[Array] = None,
    p: Optional[Array] = None,
) -> tuple[Array, List[UserStageLog]]:
    n = d_hat.shape[1]
    if user_indices is None:
        user_indices = np.arange(n, dtype=np.int64)
    p_hat = np.zeros((2, n), dtype=np.float64)
    logs: List[UserStageLog] = []
    d_true_all = None
    if p is not None:
        from .io_mat import geometric_distances

        d_true_all = geometric_distances(p, bs)

    for u in range(n):
        p_true_u = p[:, u] if p is not None else None
        d_true_u = d_true_all[:, u] if d_true_all is not None else None
        xy, lg = localize_user(
            d_hat[:, u], bs, calib, cfg, p_true_u=p_true_u, d_true_u=d_true_u
        )
        p_hat[:, u] = xy
        logs.append(lg)
    return p_hat, logs


VERSION_REGISTRY: Dict[str, PipelineConfig] = {
    "v00": PipelineConfig(
        version="v00",
        loss="linear",
        calib="none",
        description="B0: LS 삼변, 보정 없음",
    ),
    "v01": PipelineConfig(
        version="v01",
        loss="huber",
        calib="none",
        description="B1: Huber 삼변, 보정 없음",
    ),
    "v02": PipelineConfig(
        version="v02",
        loss="huber",
        calib="global_affine",
        description="Huber + 전역 아핀 보정 (α,β 2개)",
    ),
    "v03": PipelineConfig(
        version="v03",
        loss="huber",
        calib="per_bs_affine",
        description="Huber + BS별 아핀 보정 (α_k,β_k 36개)",
    ),
    "v04": PipelineConfig(
        version="v04",
        loss="huber",
        calib="per_bs_affine",
        gate=True,
        gate_tau=20.0,
        description="[deprecated] v03 + HardGate — 기말 미사용",
    ),
    "v05": PipelineConfig(
        version="v05",
        loss="huber",
        calib="per_bs_quadratic",
        description="Huber + BS별 2차 보정 (a·d + b·d² + c)",
    ),
    "v06": PipelineConfig(
        version="v06",
        loss="huber",
        calib="per_bs_isotonic",
        description="Huber + BS별 Isotonic (단조) 보정",
    ),
    "v09": PipelineConfig(
        version="v09",
        loss="huber",
        calib="per_bs_quadratic",
        huber_f_scale=1.0,
        description="v05 + Huber f_scale CV 튜닝",
    ),
    "v10": PipelineConfig(
        version="v10",
        loss="huber",
        calib="per_bs_isotonic",
        huber_f_scale=1.0,
        description="v06 + Huber f_scale CV 튜닝",
    ),
    "v11": PipelineConfig(
        version="v11",
        loss="huber",
        calib="per_bs_isotonic",
        description="v06 + soft 거리 가중 삼변 (gamma CV)",
    ),
    "v12": PipelineConfig(
        version="v12",
        loss="huber",
        calib="per_bs_isotonic_radial",
        description="v06 + 반경 3구간 Isotonic (near/mid/far)",
    ),
    "v13": PipelineConfig(
        version="v13",
        loss="huber",
        calib="mlp_dist",
        description="Huber + PyTorch MLP 거리 보정 (18→64→64→18 residual)",
    ),
    "v14": PipelineConfig(
        version="v14",
        loss="huber",
        calib="per_bs_isotonic",
        description="v06 + NLOS 비대칭 Huber (pred>d_obs 강한 penalty, CV)",
    ),
    "v15": PipelineConfig(
        version="v15",
        loss="huber",
        calib="per_bs_isotonic",
        description="v06 + Isotonic 후 shortest-k BS만 삼변 (k CV)",
    ),
    "v16": PipelineConfig(
        version="v16",
        loss="huber",
        calib="isotonic_mlp",
        description="Isotonic + MLP 잔차 보정 (2단계 학습)",
    ),
    "v17": PipelineConfig(
        version="v17",
        loss="huber",
        calib="isotonic_mlp",
        description="v16 + soft 거리 가중 γ (CV)",
    ),
    "v18": PipelineConfig(
        version="v18",
        loss="huber",
        calib="pos_mlp",
        description="PyTorch 직접 좌표 회귀 d_hat→(x,y)",
    ),
    "v19": PipelineConfig(
        version="v19",
        loss="huber",
        calib="gbr_dist",
        description="HistGradientBoosting BS별 거리 보정",
    ),
    "v20": PipelineConfig(
        version="v20",
        loss="huber",
        calib="isotonic_mlp",
        description="v16 + PosMLP 블렌드 (w·trilat + (1-w)·pos, CV)",
    ),
    # v17 base: isotonic_mlp + weight_gamma=1.0 (set in runner)
    "v21": PipelineConfig(
        version="v21",
        loss="huber",
        calib="isotonic_mlp_mindist",
        weight_gamma=1.0,
        description="v17 + min(d_hat) 3구간 Isotonic (far 전용 보정)",
    ),
    "v22": PipelineConfig(
        version="v22",
        loss="huber",
        calib="quantile_mlp",
        weight_gamma=1.0,
        quantile_tau=0.35,
        description="v17 + quantile(τ=0.35) 거리 보정 + MLP",
    ),
    "v23": PipelineConfig(
        version="v23",
        loss="huber",
        calib="isotonic_mlp",
        weight_gamma=1.0,
        zone_near_boost=2.5,
        zone_far_scale=0.55,
        description="v17 + raw min(d_hat) zone 가중 (near 강화)",
    ),
    "v24": PipelineConfig(
        version="v24",
        loss="huber",
        calib="isotonic_mlp",
        weight_gamma=1.0,
        mlp_noise_std=0.03,
        description="v17 + MLP 학습 시 d_hat 노이즈 증강",
    ),
    "v25": PipelineConfig(
        version="v25",
        loss="huber",
        calib="isotonic_mlp",
        weight_gamma=1.0,
        pos_refine_affine=True,
        description="v17 + 2-pass 위치 아핀 (1차 삼변 후 잔차 보정)",
    ),
    "v26": PipelineConfig(
        version="v26",
        loss="huber",
        calib="isotonic_mlp_far2",
        weight_gamma=1.0,
        description="v17 + far UE 2차 Isotonic (pass1 보정 거리)",
    ),
    "v27": PipelineConfig(
        version="v27",
        loss="huber",
        calib="quantile_mlp",
        weight_gamma=1.0,
        description="quantile+MLP, tau CV 0.25-0.45",
    ),
    "v28": PipelineConfig(
        version="v28",
        loss="huber",
        calib="isotonic_mlp",
        pos_refine_affine=True,
        description="v25 + gamma CV (재튜닝, MLP+아핀)",
    ),
    "v29": PipelineConfig(
        version="v29",
        loss="huber",
        calib="isotonic_mlp",
        pos_refine_affine=True,
        description="v25 + Huber f_scale CV",
    ),
    "v30": PipelineConfig(
        version="v30",
        loss="huber",
        calib="isotonic_mlp",
        weight_gamma=1.0,
        asym_pos_weight=5.0,
        pos_refine_affine=True,
        description="v25 + asym Huber w=5 (NLOS) + 위치 아핀",
    ),
    "v31": PipelineConfig(
        version="v31",
        loss="huber",
        calib="isotonic_mlp",
        pos_refine_affine=True,
        description="v25 + top_k CV (보정 후 k BS)",
    ),
    "v32": PipelineConfig(
        version="v32",
        loss="huber",
        calib="isotonic_mlp",
        weight_gamma=1.5,
        pos_refine_affine=True,
        description="v25 + gamma=1.5 고정 (v11 통찰+MLP)",
    ),
    "v33": PipelineConfig(
        version="v33",
        loss="huber",
        calib="isotonic_mlp",
        weight_gamma=1.5,
        huber_f_scale=0.8,
        pos_refine_affine=True,
        description="v25 + gamma1.5 + f0.8 (v10+v11 조합)",
    ),
    "v34": PipelineConfig(
        version="v34",
        loss="huber",
        calib="isotonic_mlp",
        weight_gamma=1.0,
        pos_refine_affine=True,
        pos_refine_far=True,
        description="v25 + far UE 2차 위치 아핀",
    ),
    "v35": PipelineConfig(
        version="v35",
        loss="huber",
        calib="isotonic_mlp_mindist",
        weight_gamma=1.0,
        pos_refine_affine=True,
        description="v21 mindist Iso+MLP + v25 아핀",
    ),
}

# Default batch run: no HardGate (v04)
ACTIVE_VERSIONS = [
    "v00",
    "v01",
    "v02",
    "v03",
    "v05",
    "v06",
    "v10",
    "v11",
    "v12",
    "v13",
    "v14",
    "v15",
    "v16",
    "v17",
    "v18",
    "v19",
    "v20",
]


def config_to_dict(cfg: PipelineConfig) -> dict:
    d = asdict(cfg)
    return d
