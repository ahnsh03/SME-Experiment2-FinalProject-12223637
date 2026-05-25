"""Fixed hold-out split for unbiased hyperparameter / model selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .calib import fit_calib
from .cv import run_fold_cv
from .io_mat import load_dataset, project_root
from .metrics import position_errors, summarize_errors
from .pipeline import PipelineConfig, localize_batch

Array = np.ndarray
HOLDOUT_SEED = 42
HOLDOUT_FRAC = 0.2  # 140 / 700


def holdout_split(n: int, seed: int = HOLDOUT_SEED) -> Tuple[Array, Array]:
    idx = np.arange(n, dtype=np.int64)
    tr, ho = train_test_split(idx, test_size=HOLDOUT_FRAC, random_state=seed, shuffle=True)
    return np.asarray(tr, dtype=np.int64), np.asarray(ho, dtype=np.int64)


def eval_holdout(
    d_hat: Array,
    p: Array,
    bs: Array,
    cfg: PipelineConfig,
    train_idx: Array,
    holdout_idx: Array,
) -> Dict[str, Any]:
    """Train on train_idx only; report RMSE on holdout_idx (no label leakage at infer)."""
    calib = fit_calib(d_hat, p, bs, train_idx, cfg.calib)
    p_hat, _ = localize_batch(
        d_hat[:, holdout_idx], bs, calib, cfg, user_indices=holdout_idx, p=None
    )
    err = position_errors(p[:, holdout_idx], p_hat)
    return {
        "holdout_position": summarize_errors(err),
        "n_train": int(train_idx.size),
        "n_holdout": int(holdout_idx.size),
    }


def eval_cv_train_only(
    d_hat: Array,
    p: Array,
    bs: Array,
    cfg: PipelineConfig,
    train_idx: Array,
) -> Dict[str, Any]:
    """5-fold CV on train subset only (for tuning without holdout peeking)."""
    sub_d = d_hat[:, train_idx]
    sub_p = p[:, train_idx]
    return run_fold_cv(sub_d, sub_p, bs, cfg)


def save_holdout_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_holdout_path() -> Path:
    return project_root() / "outputs" / "holdout_eval.json"
