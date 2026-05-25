"""Load and normalize InF_DH_FR1.mat / DH_FR1.mat."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy.io as sio

Array = np.ndarray


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_train_mat_path() -> Path:
    return project_root() / "data" / "InF_DH_FR1.mat"


def load_dataset(mat_path: Path | str | None = None) -> Dict[str, Any]:
    path = Path(mat_path) if mat_path is not None else default_train_mat_path()
    raw = sio.loadmat(path, squeeze_me=True)
    p = np.asarray(raw["p"], dtype=np.float64)
    d_hat = np.asarray(raw["d_hat"], dtype=np.float64)
    bs = np.asarray(raw.get("p_bs", raw["BS_positions"]), dtype=np.float64)
    indices = raw.get("indices")
    if indices is not None:
        indices = np.asarray(indices).ravel().astype(np.int64)

    if p.ndim == 1:
        p = p.reshape(2, -1)
    if d_hat.shape[0] != 18:
        d_hat = d_hat.T
    if bs.shape[0] != 2:
        bs = bs.T

    n_user = d_hat.shape[1]
    if p.shape[1] != n_user:
        raise ValueError(f"p columns {p.shape[1]} != d_hat users {n_user}")

    return {
        "path": str(path.resolve()),
        "p": p,
        "d_hat": d_hat,
        "bs": bs,
        "indices": indices,
        "n_user": n_user,
        "n_bs": 18,
    }


def geometric_distances(p: Array, bs: Array) -> Array:
    """(18, N) true distances from UE positions p (2,N) and bs (2,18)."""
    px, py = p[0], p[1]
    bx, by = bs[0], bs[1]
    return np.sqrt((px[None, :] - bx[:, None]) ** 2 + (py[None, :] - by[:, None]) ** 2)
