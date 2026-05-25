"""Direct position regressor: d_hat (18) -> (x, y). Inference uses d_hat only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError("PyTorch required for PosMLP") from e


class PosMLP(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(18, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PosMLPBundle:
    state_dict: dict
    in_mean: Array
    in_std: Array
    out_mean: Array
    out_std: Array
    hidden: int = 128
    dropout: float = 0.3

    def apply(self, d_hat_u: Array) -> Array:
        d = np.asarray(d_hat_u, dtype=np.float64).reshape(-1)
        x = (d - self.in_mean) / self.in_std
        model = PosMLP(self.hidden, self.dropout)
        model.load_state_dict(self.state_dict)
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
            y = model(xt).numpy().reshape(2)
        out = y * self.out_std + self.out_mean
        from .spatial import clip_xy

        return clip_xy(out)


def fit_pos_mlp(
    d_hat: Array,
    p: Array,
    train_idx: Array,
    *,
    epochs: int = 150,
    lr: float = 1e-3,
    hidden: int = 128,
    dropout: float = 0.3,
    seed: int = 42,
) -> PosMLPBundle:
    torch.manual_seed(seed)
    dh = d_hat[:, train_idx].T
    pt = p[:, train_idx].T

    in_mean = dh.mean(axis=0)
    in_std = np.maximum(dh.std(axis=0), 1e-3)
    out_mean = pt.mean(axis=0)
    out_std = np.maximum(pt.std(axis=0), 1e-3)

    xn = (dh - in_mean) / in_std
    yn = (pt - out_mean) / out_std

    model = PosMLP(hidden, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(xn.astype(np.float32))
    yt = torch.from_numpy(yn.astype(np.float32))
    t_out_std = torch.from_numpy(out_std.astype(np.float32))
    t_out_mean = torch.from_numpy(out_mean.astype(np.float32))

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(xt) * t_out_std + t_out_mean
        loss = nn.functional.smooth_l1_loss(pred, yt)
        loss.backward()
        opt.step()

    return PosMLPBundle(
        state_dict=model.state_dict(),
        in_mean=in_mean,
        in_std=in_std,
        out_mean=out_mean,
        out_std=out_std,
        hidden=hidden,
        dropout=dropout,
    )


def bundle_to_dict(b: PosMLPBundle) -> dict:
    return {
        "in_mean": b.in_mean.tolist(),
        "in_std": b.in_std.tolist(),
        "out_mean": b.out_mean.tolist(),
        "out_std": b.out_std.tolist(),
        "hidden": b.hidden,
        "dropout": b.dropout,
    }


def bundle_from_dict(d: dict, state_dict: dict) -> PosMLPBundle:
    return PosMLPBundle(
        state_dict=state_dict,
        in_mean=np.asarray(d["in_mean"], dtype=np.float64),
        in_std=np.asarray(d["in_std"], dtype=np.float64),
        out_mean=np.asarray(d["out_mean"], dtype=np.float64),
        out_std=np.asarray(d["out_std"], dtype=np.float64),
        hidden=int(d.get("hidden", 128)),
        dropout=float(d.get("dropout", 0.3)),
    )
