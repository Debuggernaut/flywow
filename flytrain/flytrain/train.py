"""Stage B: fit Tier A type-gains (and a diagnostic DN→6 linear head).

The deliverable is θ, not the diagnostic head. The head only answers:
"do cached DN rates already contain the button bits?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .connectome import Connectome
from .constants import BUTTONS, GAIN, THETA_HI, THETA_LO, VISION_HZ_MAX, WSYN
from .cuts import CutMatrices, extract_standard_cuts


def _ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary average precision, no sklearn dependency."""
    y_true = y_true.astype(np.float64)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score)
    yt = y_true[order]
    tp = np.cumsum(yt)
    prec = tp / np.arange(1, len(yt) + 1)
    rec_inc = yt / yt.sum()
    return float((prec * rec_inc).sum())


class TypeGainHop(nn.Module):
    """W_ij = W0_ij * exp(θ[type_pre]) or exp(θ[type_pre, type_post])."""

    def __init__(self, cut: CutMatrices, pair: bool = False):
        super().__init__()
        self.pair = pair
        self.register_buffer("pre_i", torch.from_numpy(cut.pre_i.astype(np.int64)))
        self.register_buffer("post_i", torch.from_numpy(cut.post_i.astype(np.int64)))
        self.register_buffer(
            "rows", torch.from_numpy(cut.edge_post_local.astype(np.int64))
        )
        self.register_buffer(
            "cols", torch.from_numpy(cut.edge_pre_local.astype(np.int64))
        )
        self.register_buffer("w0", torch.from_numpy(cut.edge_w0.astype(np.float32)))
        self.n_post = len(cut.post_i)
        self.n_pre = len(cut.pre_i)

        pre_types = cut.edge_pre_type.astype(str)
        post_types = cut.edge_post_type.astype(str)
        if len(pre_types) == 0:
            keys = np.array([], dtype=object)
            uniq, inv = np.array([], dtype=object), np.array([], dtype=np.int64)
        elif pair:
            keys = np.array([f"{a}||{b}" for a, b in zip(pre_types, post_types)])
            uniq, inv = np.unique(keys, return_inverse=True)
        else:
            keys = pre_types
            uniq, inv = np.unique(keys, return_inverse=True)
        self.keys = [str(k) for k in uniq]
        self.register_buffer("theta_ix", torch.from_numpy(np.asarray(inv, dtype=np.int64)))
        self.theta = nn.Parameter(torch.zeros(len(self.keys), dtype=torch.float32))
        self.cut_name = cut.name

    def gains(self) -> torch.Tensor:
        th = self.theta.clamp(THETA_LO, THETA_HI)
        return torch.exp(th)

    def edge_gains(self) -> torch.Tensor:
        return self.gains()[self.theta_ix]

    def forward(self, x_pre: torch.Tensor) -> torch.Tensor:
        """x_pre: (B, n_pre) rates of the cut's pre population."""
        if self.w0.numel() == 0 or self.n_pre == 0 or self.n_post == 0:
            return x_pre.new_zeros((x_pre.shape[0], max(self.n_post, 0)))
        w = self.w0 * self.edge_gains()
        idx = torch.stack([self.rows, self.cols], dim=0)
        with torch.sparse.check_sparse_tensor_invariants(False):
            W = torch.sparse_coo_tensor(
                idx, w, (self.n_post, self.n_pre), dtype=x_pre.dtype, device=x_pre.device
            ).coalesce()
        return torch.sparse.mm(W, x_pre.transpose(0, 1)).transpose(0, 1)


class MultiHop(nn.Module):
    def __init__(
        self,
        cuts: dict[str, CutMatrices],
        net: Connectome,
        pair: bool = False,
        use_tier_c: bool = False,
    ):
        super().__init__()
        self.ol_vpn = TypeGainHop(cuts["ol_vpn"], pair=pair)
        self.vpn_dn = TypeGainHop(cuts["vpn_dn"], pair=pair)
        self.dn_vnc = TypeGainHop(cuts["dn_vnc"], pair=pair)
        self.use_tier_c = use_tier_c and cuts["ol_dn"].W_compact.nnz > 0
        self.ol_dn = TypeGainHop(cuts["ol_dn"], pair=pair) if self.use_tier_c else None

        # Map compact VNC outputs onto the six pools.
        vnc_post = cuts["dn_vnc"].post_i
        pos = {int(g): k for k, g in enumerate(vnc_post)}
        pool_cols = []
        for name in BUTTONS:
            idx = [pos[int(g)] for g in net.pool_i[name] if int(g) in pos]
            pool_cols.append(np.asarray(idx, dtype=np.int64))
        self.pool_cols = pool_cols
        self.pool_scale = nn.Parameter(torch.full((6,), 0.05, dtype=torch.float32))
        self.pool_bias = nn.Parameter(torch.zeros(6, dtype=torch.float32))
        self.syn_scale = float(WSYN * GAIN)

    def hops(self) -> list[TypeGainHop]:
        hs = [self.ol_vpn, self.vpn_dn, self.dn_vnc]
        if self.ol_dn is not None:
            hs.append(self.ol_dn)
        return hs

    def forward_rates(self, hex_rate: torch.Tensor) -> dict[str, torch.Tensor]:
        x = hex_rate / float(VISION_HZ_MAX)
        s = self.syn_scale
        vpn = F.softplus(s * self.ol_vpn(x))
        dn = F.softplus(s * self.vpn_dn(vpn))
        if self.ol_dn is not None:
            dn = dn + F.softplus(s * self.ol_dn(x))
        vnc = F.softplus(s * self.dn_vnc(dn))
        pools = []
        for i, cols in enumerate(self.pool_cols):
            if len(cols) == 0:
                pools.append(hex_rate.new_zeros(hex_rate.shape[0]))
            else:
                pools.append(vnc[:, torch.as_tensor(cols, device=vnc.device)].mean(dim=1))
        raw = torch.stack(pools, dim=1)
        logits = self.pool_scale * raw + self.pool_bias
        return {"vpn": vpn, "dn": dn, "vnc": vnc, "pool_raw": raw, "logits": logits}

    def forward(self, hex_rate: torch.Tensor) -> torch.Tensor:
        return self.forward_rates(hex_rate)["logits"]


class DiagnosticHead(nn.Module):
    def __init__(self, n_dn: int):
        super().__init__()
        self.lin = nn.Linear(n_dn, 6)

    def forward(self, dn: torch.Tensor) -> torch.Tensor:
        return self.lin(dn)


def export_gains(model: MultiHop) -> pd.DataFrame:
    rows = []
    for hop in model.hops():
        g = hop.gains().detach().cpu().numpy()
        for key, val in zip(hop.keys, g):
            if hop.pair and "||" in key:
                a, b = key.split("||", 1)
            else:
                a, b = key, "*"
            rows.append(
                {
                    "cut": hop.cut_name,
                    "type_pre": a,
                    "type_post": b,
                    "theta": float(np.log(max(val, 1e-12))),
                    "exp_theta": float(val),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class TrainResult:
    metrics: dict
    gains: pd.DataFrame


def _split_idx(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    split = np.array(meta["split"])
    tr = np.flatnonzero(split == "train")
    va = np.flatnonzero(split == "val")
    if len(va) == 0:
        va = tr
    return tr, va


def train_tier_a(
    net: Connectome,
    cache: dict,
    meta: dict,
    out_dir: Path,
    steps: int = 2000,
    lr: float = 1e-2,
    l2: float = 1e-4,
    pair: bool = False,
    use_tier_c: bool = False,
    device: str = "cpu",
) -> TrainResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cuts = extract_standard_cuts(net)
    for name, c in cuts.items():
        print(
            f"  cut {name:8s}  edges={c.W_compact.nnz:7d}  "
            f"pre={len(c.pre_i):6d}  post={len(c.post_i):6d}"
        )

    empty = [name for name, c in cuts.items() if c.W_compact.nnz == 0 and name != "ol_dn"]
    if empty:
        print("EMPTY required cut(s):", empty)
        print("Do not raise GAIN. Inspect inventory and consider Tier C.")

    hex_drive = cache["hex_drive_hz"]
    hex_pos = {int(g): k for k, g in enumerate(net.hex_i)}
    take = np.array(
        [hex_pos.get(int(g), -1) for g in cuts["ol_vpn"].pre_i], dtype=np.int64
    )
    aligned = np.zeros((hex_drive.shape[0], len(take)), dtype=np.float32)
    ok = take >= 0
    if ok.any():
        aligned[:, ok] = hex_drive[:, take[ok]]
    hex_x = torch.from_numpy(aligned).float().to(device)
    dn_x = torch.from_numpy(cache["dn_hz"]).float().to(device)
    y = torch.from_numpy(cache["y"]).float().to(device)
    tr, va = _split_idx(meta)
    tr_t = torch.from_numpy(tr).long()
    va_t = torch.from_numpy(va).long()

    model = MultiHop(cuts, net, pair=pair, use_tier_c=use_tier_c).to(device)
    diag = DiagnosticHead(dn_x.shape[1]).to(device)
    opt = torch.optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if "theta" in n], "lr": lr},
            {
                "params": [p for n, p in model.named_parameters() if "theta" not in n],
                "lr": lr * 0.5,
            },
            {"params": diag.parameters(), "lr": 1e-2},
        ]
    )

    history = []
    for step in range(1, steps + 1):
        opt.zero_grad()
        logits = model(hex_x[tr_t])
        dlog = diag(dn_x[tr_t])
        bce = F.binary_cross_entropy_with_logits(logits, y[tr_t])
        bce_d = F.binary_cross_entropy_with_logits(dlog, y[tr_t])
        th = torch.cat([h.theta for h in model.hops()])
        reg = l2 * (th * th).mean()
        loss = bce + bce_d + reg
        loss.backward()
        opt.step()
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            with torch.no_grad():
                v_log = model(hex_x[va_t])
                v_bce = float(F.binary_cross_entropy_with_logits(v_log, y[va_t]))
                v_prob = torch.sigmoid(v_log).cpu().numpy()
                yv = y[va_t].cpu().numpy()
                aps = {
                    BUTTONS[i]: _ap(yv[:, i], v_prob[:, i]) for i in range(6)
                }
                d_prob = torch.sigmoid(diag(dn_x[va_t])).cpu().numpy()
                d_aps = {
                    BUTTONS[i]: _ap(yv[:, i], d_prob[:, i]) for i in range(6)
                }
            row = {
                "step": step,
                "train_bce": float(bce.detach()),
                "val_bce": v_bce,
                "diag_train_bce": float(bce_d.detach()),
                "reg": float(reg.detach()),
                "ap": aps,
                "diag_ap": d_aps,
            }
            history.append(row)
            print(
                f"step {step:5d}  trainBCE={row['train_bce']:.4f}  "
                f"valBCE={v_bce:.4f}  "
                f"AP={ {k: None if np.isnan(v) else round(v, 3) for k, v in aps.items()} }  "
                f"diagAP={ {k: None if np.isnan(v) else round(v, 3) for k, v in d_aps.items()} }"
            )

    gains = export_gains(model)
    gains.to_csv(out_dir / "gains_type.csv", index=False)

    metrics = {
        "history": history,
        "val_bce": history[-1]["val_bce"] if history else None,
        "per_button_ap": history[-1]["ap"] if history else {},
        "diagnostic_per_button_ap": history[-1]["diag_ap"] if history else {},
        "n_params_theta": int(sum(h.theta.numel() for h in model.hops())),
        "pair": pair,
        "use_tier_c": bool(model.ol_dn is not None),
        "gain_global": GAIN,
        "probes": meta.get("probes", {}),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    return TrainResult(metrics=metrics, gains=gains)
