"""sklearn HistGradientBoosting distance calibrator (per-BS)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .io_mat import geometric_distances

Array = np.ndarray


@dataclass
class GBRBundle:
    models: list

    def apply(self, d_hat_u: Array) -> Array:
        d = np.asarray(d_hat_u, dtype=np.float64).reshape(-1)
        out = np.empty(18, dtype=np.float64)
        for k, m in enumerate(self.models):
            out[k] = float(m.predict(d[k].reshape(1, -1))[0])
        return np.maximum(out, 1.0)


def fit_gbr_calib(
    d_hat: Array,
    p: Array,
    bs: Array,
    train_idx: Array,
    *,
    max_depth: int = 6,
    max_iter: int = 200,
    seed: int = 42,
) -> GBRBundle:
    d_true = geometric_distances(p[:, train_idx], bs)
    models = []
    for k in range(18):
        m = HistGradientBoostingRegressor(
            max_depth=max_depth,
            max_iter=max_iter,
            random_state=seed + k,
            early_stopping=True,
            validation_fraction=0.15,
        )
        m.fit(d_hat[k, train_idx].reshape(-1, 1), d_true[k])
        models.append(m)
    return GBRBundle(models=models)
