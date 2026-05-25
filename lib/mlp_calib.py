"""PyTorch distance calibrator (train.py / v13+)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io_mat import geometric_distances

Array = np.ndarray

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError("PyTorch required for MLP calibration") from e


class DistMLP(nn.Module):
    """Residual MLP: d_out = d_in + delta(d_in)."""

    def __init__(self, hidden: int = 64, dropout: float = 0.25) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 18),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


@dataclass
class MLPBundle:
    state_dict: dict
    in_mean: Array
    in_std: Array
    hidden: int = 64
    dropout: float = 0.25

    def apply(self, d_hat_u: Array) -> Array:
        d = np.asarray(d_hat_u, dtype=np.float64).reshape(-1)
        x = (d - self.in_mean) / self.in_std
        model = DistMLP(self.hidden, self.dropout)
        model.load_state_dict(self.state_dict)
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
            out = model(xt).numpy().reshape(-1) * self.in_std + self.in_mean
        return np.maximum(out, 1.0)


def fit_mlp_calib(
    d_hat: Array,
    p: Array,
    bs: Array,
    train_idx: Array,
    *,
    epochs: int = 80,
    lr: float = 1e-3,
    hidden: int = 64,
    dropout: float = 0.25,
    seed: int = 42,
) -> MLPBundle:
    torch.manual_seed(seed)
    d_true = geometric_distances(p[:, train_idx], bs)
    dh = d_hat[:, train_idx].T  # (N, 18)
    y = d_true.T

    in_mean = dh.mean(axis=0)
    in_std = np.maximum(dh.std(axis=0), 1e-3)
    xn = (dh - in_mean) / in_std
    yn = y

    model = DistMLP(hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    xt = torch.from_numpy(xn.astype(np.float32))
    yt = torch.from_numpy(yn.astype(np.float32))

    t_std = torch.from_numpy(in_std.astype(np.float32))
    t_mean = torch.from_numpy(in_mean.astype(np.float32))
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(xt) * t_std + t_mean
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()

    return MLPBundle(
        state_dict=model.state_dict(),
        in_mean=in_mean,
        in_std=in_std,
        hidden=hidden,
        dropout=dropout,
    )


def fit_mlp_on_iso(
    d_hat: Array,
    p: Array,
    bs: Array,
    train_idx: Array,
    iso_apply_fn,
    **kwargs,
) -> MLPBundle:
    """Train MLP on isotonic-preprocessed distances (v16)."""
    d_true = geometric_distances(p[:, train_idx], bs)
    rows = []
    targets = []
    for j, u in enumerate(train_idx):
        rows.append(iso_apply_fn(d_hat[:, u]))
        targets.append(d_true[:, j])
    dh = np.stack(rows, axis=1).T
    y = np.stack(targets, axis=1).T
    fake_d = d_hat.copy()
    fake_p = p[:, train_idx]
    # reuse fit with synthetic d_hat replaced
    in_mean = dh.mean(axis=0)
    in_std = np.maximum(dh.std(axis=0), 1e-3)
    return _fit_from_arrays(dh, y, in_mean, in_std, **kwargs)


def _fit_from_arrays(
    dh: Array,
    y: Array,
    in_mean: Array,
    in_std: Array,
    *,
    epochs: int = 80,
    lr: float = 1e-3,
    hidden: int = 64,
    dropout: float = 0.25,
    seed: int = 42,
    noise_std: float = 0.0,
) -> MLPBundle:
    torch.manual_seed(seed)
    model = DistMLP(hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xn = (dh - in_mean) / in_std
    xt = torch.from_numpy(xn.astype(np.float32))
    yt = torch.from_numpy(y.astype(np.float32))
    t_std = torch.from_numpy(in_std.astype(np.float32))
    t_mean = torch.from_numpy(in_mean.astype(np.float32))
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        xb = xt
        if noise_std > 0:
            xb = xt + torch.randn_like(xt) * float(noise_std)
        pred = model(xb) * t_std + t_mean
        loss = nn.functional.mse_loss(pred, yt)
        loss.backward()
        opt.step()
    return MLPBundle(
        state_dict=model.state_dict(),
        in_mean=in_mean,
        in_std=in_std,
        hidden=hidden,
        dropout=dropout,
    )


def bundle_to_dict(b: MLPBundle) -> dict:
    return {
        "in_mean": b.in_mean.tolist(),
        "in_std": b.in_std.tolist(),
        "hidden": b.hidden,
        "dropout": b.dropout,
    }


def bundle_from_dict(d: dict, state_dict: dict) -> MLPBundle:
    return MLPBundle(
        state_dict=state_dict,
        in_mean=np.asarray(d["in_mean"], dtype=np.float64),
        in_std=np.asarray(d["in_std"], dtype=np.float64),
        hidden=int(d.get("hidden", 64)),
        dropout=float(d.get("dropout", 0.25)),
    )
