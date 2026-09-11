#!/usr/bin/env python3
"""Drive MaleCNS locally: load tables, build a signed sparse graph, inject input, read output.

Requires the three public Feather dumps in DATA_DIR:
  body-annotations.feather
  body-neurotransmitters.feather
  connectome-weights.feather

This is anatomy plus a toy dynamical model. It is not a fly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path(r"C:\Dev\flywow\data")
MIN_WEIGHT = 5
EXPERIMENT = "giant_fiber"  # "giant_fiber" | "orn" | "visual"

# Linear-rate model
RATE_DRIVE = 1.0
RATE_STEPS = 4

# LIF model (optional second pass)
RUN_LIF = False
LIF_HZ = 100.0
LIF_MS = 100
LIF_GAIN = 0.05
DT = 0.001
TAU = 0.020
VREST, VTH, VRESET = -52.0, -45.0, -52.0

FAST_NT = {
    "acetylcholine": 1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def nt_sign(row: pd.Series) -> int:
    c = row["consensus_nt"]
    if c in FAST_NT:
        return FAST_NT[c]
    conf = row["predicted_nt_confidence"]
    if pd.notna(conf) and conf >= 0.5 and row["predicted_nt"] in FAST_NT:
        return FAST_NT[row["predicted_nt"]]
    return FAST_NT.get(row["celltype_predicted_nt"], 0)


def load_cells(data_dir: Path) -> pd.DataFrame:
    ann = pd.read_feather(data_dir / "body-annotations.feather").rename(
        columns={"bodyId": "body"}
    )
    nt = pd.read_feather(data_dir / "body-neurotransmitters.feather")
    keep_nt = [
        "body",
        "consensus_nt",
        "predicted_nt",
        "predicted_nt_confidence",
        "celltype_predicted_nt",
        "celltype_predicted_nt_confidence",
    ]
    cells = ann.merge(nt[keep_nt], on="body", how="left")
    cells["sign"] = cells.apply(nt_sign, axis=1)
    cells = cells.dropna(subset=["body"]).drop_duplicates("body")
    return cells.reset_index(drop=True)


def load_edges(data_dir: Path, legal_bodies: set[int]) -> pd.DataFrame:
    w = pd.read_feather(data_dir / "connectome-weights.feather").rename(
        columns={"body_pre": "pre", "body_post": "post"}
    )
    w = w[
        (w["weight"] >= MIN_WEIGHT)
        & w["pre"].isin(legal_bodies)
        & w["post"].isin(legal_bodies)
        & (w["pre"] != w["post"])
    ]
    return w


def build_W(nodes: pd.DataFrame, edges: pd.DataFrame) -> sparse.csr_matrix:
    idx = {int(b): i for i, b in enumerate(nodes["body"])}
    pre_i = edges["pre"].map(idx).to_numpy()
    post_i = edges["post"].map(idx).to_numpy()
    sign = nodes.set_index("body")["sign"].reindex(edges["pre"]).fillna(0).to_numpy()
    val = edges["weight"].to_numpy(dtype=np.float32) * sign.astype(np.float32)
    n = len(nodes)
    # W[target, source] so x_next = f(W @ x)
    return sparse.csr_matrix((val, (post_i, pre_i)), shape=(n, n), dtype=np.float32)


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

def ports(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    def m(pred) -> np.ndarray:
        return pred.to_numpy()

    return {
        "visual": m((nodes.superclass == "ol_sensory") & (nodes["class"] == "visual")),
        "orn": m(nodes["class"] == "olfactory"),
        "alpn": m(nodes["class"] == "ALPN"),
        "vpn": m(nodes.superclass == "visual_projection"),
        "dn": m(nodes.superclass == "descending_neuron"),
        "mn": m(nodes.superclass.isin(["vnc_motor", "cb_motor"])),
        "mn_fl": m((nodes.superclass == "vnc_motor") & (nodes["subclass"] == "fl")),
        "gf": m(nodes["type"] == "DNp01"),
    }


def choose_io(nodes: pd.DataFrame, p: dict[str, np.ndarray], experiment: str):
    if experiment == "giant_fiber":
        return p["gf"], p["mn"], "DNp01 (Giant Fiber)", "all motor neurons"
    if experiment == "visual":
        return p["visual"], p["vpn"], "photoreceptors", "visual projection neurons"
    if experiment == "orn":
        counts = nodes.loc[nodes["class"].eq("olfactory"), "type"].value_counts()
        name = counts.index[0]
        is_in = (nodes["class"].eq("olfactory") & nodes["type"].eq(name)).to_numpy()
        return is_in, p["alpn"], f"ORN type {name}", "ALPNs"
    raise ValueError(f"unknown experiment {experiment}")


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def drive_rate(W: sparse.csr_matrix, is_in: np.ndarray, drive: float, steps: int) -> np.ndarray:
    x = np.zeros(W.shape[0], dtype=np.float32)
    x[is_in] = drive
    for _ in range(steps):
        x = np.tanh(W @ x)
        x[is_in] = drive
    return x


def drive_lif(W: sparse.csr_matrix, is_in: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    v = np.full(n, VREST, dtype=np.float32)
    spikes = np.zeros(n, dtype=np.float32)
    spikes[is_in] = LIF_HZ * DT
    rate = np.zeros(n, dtype=np.float32)
    steps = int(LIF_MS)
    for _ in range(steps):
        v += DT * (-(v - VREST) / TAU) + (LIF_GAIN * (W @ spikes))
        fired = v >= VTH
        rate += fired
        v[fired] = VRESET
        spikes = fired.astype(np.float32)
        spikes[is_in] = LIF_HZ * DT
    return rate / (LIF_MS * DT)


def top_hits(nodes: pd.DataFrame, value: np.ndarray, is_out: np.ndarray, k: int = 20) -> pd.DataFrame:
    hit = nodes.loc[is_out, ["body", "type", "instance", "superclass", "class", "subclass", "exitNerve"]].copy()
    hit["value"] = value[is_out]
    return hit.sort_values("value", key=np.abs, ascending=False).head(k)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"loading from {DATA_DIR}")
    nodes = load_cells(DATA_DIR)
    edges = load_edges(DATA_DIR, set(nodes["body"].astype(int)))
    print(f"neurons {len(nodes):,}  edges>={MIN_WEIGHT} {len(edges):,}")
    print("sign counts:\n", nodes["sign"].value_counts().to_string())

    W = build_W(nodes, edges)
    print(f"W shape {W.shape}  nnz {W.nnz:,}")

    p = ports(nodes)
    for name, m in p.items():
        print(f"  port {name:8} {int(m.sum()):6}")

    is_in, is_out, in_name, out_name = choose_io(nodes, p, EXPERIMENT)

    # after W and is_in exist
    x0 = np.zeros(W.shape[0], dtype=np.float32)
    x0[is_in] = 1.0 #Set only the two Giant Fiber cells to 1
    current = W @ x0          # signed synapse-count onto each cell

    hit = nodes.assign(current=current)
    typed = hit["type"].notna()
    mn = hit.superclass.isin(["vnc_motor", "cb_motor"])

    print(hit.loc[typed & (hit.current != 0)]
            .sort_values("current", key=np.abs, ascending=False)
            [["body", "type", "instance", "superclass", "subclass", "current"]]
            .head(25)
            .to_string(index=False))

    print("\nmotor only")
    print(hit.loc[mn & typed]
            .sort_values("current", key=np.abs, ascending=False)
            [["body", "type", "subclass", "exitNerve", "current"]]
            .head(20)
            .to_string(index=False))
    # print(f"\nexperiment={EXPERIMENT}")
    # print(f"drive {in_name}  n={int(is_in.sum())}")
    # print(f"read  {out_name}  n={int(is_out.sum())}")
    # if is_in.sum() == 0 or is_out.sum() == 0:
    #     raise SystemExit("empty input or output port — check type/superclass strings")

    # x = drive_rate(W, is_in, RATE_DRIVE, RATE_STEPS)
    # print(f"\nlinear-rate readout ({RATE_STEPS} steps, drive={RATE_DRIVE})")
    # print(top_hits(nodes, x, is_out).to_string(index=False))

    # # also show where the signal went, regardless of the chosen output port
    # hidden = top_hits(nodes, x, np.ones(len(nodes), dtype=bool), k=15)
    # print("\ntop 15 cells anywhere")
    # print(hidden.to_string(index=False))

    # if RUN_LIF:
    #     hz = drive_lif(W, is_in)
    #     print(f"\nLIF readout ({LIF_MS} ms, input {LIF_HZ} Hz, gain={LIF_GAIN})")
    #     print(top_hits(nodes, hz, is_out).to_string(index=False))


if __name__ == "__main__":
    main()
