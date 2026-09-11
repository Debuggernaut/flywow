#!/usr/bin/env python3
"""MaleCNS 100 Hz LIF loop.

Sugar/water GRNs emit independent ~1 Hz Poisson spikes.
Those spikes propagate through the signed connectome as leaky
integrate-and-fire units (Shiu / fly-brain-minecraft parameters).

Not a fly. Not flavor. A live cartoon of the published graph.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(r"C:\Dev\flywow\data")
MIN_WEIGHT = 5

# Loop: 100 Hz biological time (10 ms per tick)
TICK_HZ = 100
DT = 1.0 / TICK_HZ
PRINT_EVERY = TICK_HZ  # one status line per simulated second
MAX_SECONDS = 30       # None = run until Ctrl+C

# GRN drive: each selected cell spikes independently at ~1 Hz
GRN_HZ = 91.0

# Shiu 2024 / minecraft-style LIF
TAU_M = 0.020
TAU_S = 0.005
VREST = -52.0
VTH = -45.0
VRESET = -52.0
TREF = 0.0022
WSYN = 0.275          # mV per signed synapse count
GAIN = 0.65           # minecraft extra scale; turn down if the net explodes

FAST_NT = {
    "acetylcholine": 1,
    "dopamine": 1,
    "octopamine": 1,
    "serotonin": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
}


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
    return w[
        (w["weight"] >= MIN_WEIGHT)
        & w["pre"].isin(legal_bodies)
        & w["post"].isin(legal_bodies)
        & (w["pre"] != w["post"])
    ]


def build_W(nodes: pd.DataFrame, edges: pd.DataFrame) -> sparse.csr_matrix:
    idx = {int(b): i for i, b in enumerate(nodes["body"])}
    pre_i = edges["pre"].map(idx).to_numpy()
    post_i = edges["post"].map(idx).to_numpy()
    sign = nodes.set_index("body")["sign"].reindex(edges["pre"]).fillna(0).to_numpy()
    val = edges["weight"].to_numpy(dtype=np.float32) * sign.astype(np.float32)
    n = len(nodes)
    return sparse.csr_matrix((val, (post_i, pre_i)), shape=(n, n), dtype=np.float32)


def sugar_water_mask(nodes: pd.DataFrame) -> np.ndarray:
    t = nodes["type"].astype(str)
    by_type = t.str.fullmatch(r"LB3a|LB3b|LB3c")
    if int(by_type.sum()) >= 10:
        return by_type.to_numpy()
    by_type = t.str.contains(r"LB3[abc]|Gr64f|Gr5a", case=False, na=False)
    if int(by_type.sum()) >= 10:
        print("LB3 exact names missing; using type contains LB3/Gr64/Gr5a")
        return by_type.to_numpy()
    gust = nodes["class"].eq("gustatory")
    print("falling back to all class==gustatory")
    return gust.to_numpy()


def print_top_motor(
    nodes: pd.DataFrame,
    mn_i: np.ndarray,
    mn_spikes_win: np.ndarray,
    mn_g_abs_sum: np.ndarray,
    v: np.ndarray,
    window_ticks: int,
    k: int = 10,
) -> None:
    """Top motor neurons in this 1 s window: spikes first, then |synaptic current|."""
    if len(mn_i) == 0:
        print("         motor    none annotated")
        return
    spikes = mn_spikes_win[mn_i]
    g_mean = mn_g_abs_sum[mn_i] / window_ticks
    # substantial: spiked, or mean |g| above a quiet ripple
    order = np.lexsort((-g_mean, -spikes))
    take = order[:k]
    rows = []
    for j in take:
        i = int(mn_i[j])
        rows.append(
            {
                "spikes/s": int(spikes[j]),
                "Hz": spikes[j] / 1.0,
                "|g|mV": round(float(g_mean[j]), 4),
                "V": round(float(v[i]), 2),
                "type": nodes.at[i, "type"],
                "subclass": nodes.at[i, "subclass"],
                "exitNerve": nodes.at[i, "exitNerve"],
                "body": int(nodes.at[i, "body"]),
            }
        )
    n_spiking = int((spikes > 0).sum())
    n_driven = int((g_mean > 0.01).sum())
    print(f"         motor    spiking={n_spiking}/{len(mn_i)}  |g|>0.01mV={n_driven}")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    print(f"loading from {DATA_DIR}")
    nodes = load_cells(DATA_DIR)
    edges = load_edges(DATA_DIR, set(nodes["body"].astype(int)))
    print(f"neurons {len(nodes):,}  edges>={MIN_WEIGHT} {len(edges):,}")

    W = build_W(nodes, edges)
    n = W.shape[0]
    print(f"W shape {W.shape}  nnz {W.nnz:,}")

    is_grn = sugar_water_mask(nodes)
    grn_i = np.flatnonzero(is_grn)
    print(f"sugar/water GRNs driven at {GRN_HZ} Hz: {len(grn_i)}")
    print(nodes.loc[is_grn, "type"].value_counts().head(15).to_string())

    mn9_i = np.flatnonzero(nodes["type"].astype(str).eq("MN9"))
    dand_i = np.flatnonzero(nodes["type"].astype(str).eq("AN13B002"))
    is_mn = nodes.superclass.isin(["vnc_motor", "cb_motor"]).to_numpy()
    mn_i = np.flatnonzero(is_mn)
    print(f"readout  MN9={len(mn9_i)}  Dandelion/AN13B002={len(dand_i)}  motor={len(mn_i)}")

    rng = np.random.default_rng(0)
    v = np.full(n, VREST, dtype=np.float32)
    g = np.zeros(n, dtype=np.float32)       # synaptic current, tau_s
    refr = np.zeros(n, dtype=np.float32)
    spikes = np.zeros(n, dtype=np.float32)

    types = nodes["type"].astype(str).to_numpy()
    p_grn = GRN_HZ * DT

    print(f"\nrunning {TICK_HZ} Hz LIF  dt={DT*1000:.1f} ms  Ctrl+C to stop")
    t0 = time.perf_counter()
    tick = 0
    n_ticks = None if MAX_SECONDS is None else int(MAX_SECONDS * TICK_HZ)

    grn_spikes_win = 0
    net_spikes_win = 0
    mn9_spikes_win = 0
    dand_spikes_win = 0
    mn_spikes_win = np.zeros(n, dtype=np.int32)
    mn_g_abs_sum = np.zeros(n, dtype=np.float64)
    v_sum = 0.0
    g_abs_sum = 0.0
    g_max = 0.0
    near_th_sum = 0

    try:
        while n_ticks is None or tick < n_ticks:
            # 1. sensors: independent Poisson, ~1 Hz
            grn_fire = rng.random(len(grn_i)) < p_grn
            spikes.fill(0.0)
            if grn_fire.any():
                spikes[grn_i[grn_fire]] = 1.0
            grn_spikes_win += int(grn_fire.sum())

            # 2. synaptic current from whoever spiked last tick (incl. GRNs)
            g += (WSYN * GAIN) * (W @ spikes)
            g *= np.float32(np.exp(-DT / TAU_S))

            # 3. leak + incoming current
            active = refr <= 0
            v[active] += np.float32(DT) * ((VREST - v[active]) / TAU_M) + g[active]
            v[~active] = VRESET
            refr = np.maximum(0.0, refr - DT)

            # 4. threshold
            fired = active & (v >= VTH)
            if fired.any():
                v[fired] = VRESET
                g[fired] = 0.0
                refr[fired] = TREF
            net_spikes_win += int(fired.sum())
            if len(mn9_i):
                mn9_spikes_win += int(fired[mn9_i].sum())
            if len(dand_i):
                dand_spikes_win += int(fired[dand_i].sum())
            if len(mn_i):
                mn_spikes_win[mn_i] += fired[mn_i].astype(np.int32)
                mn_g_abs_sum[mn_i] += np.abs(g[mn_i])

            # GRNs that fired this tick are sources even if they missed VTH
            # (they are clamped sensory spikes, like optogenetic drive)
            spikes = fired.astype(np.float32)
            if grn_fire.any():
                spikes[grn_i[grn_fire]] = 1.0

            v_sum += float(v.mean())
            g_abs_sum += float(np.abs(g).mean())
            g_max = max(g_max, float(np.abs(g).max()))
            near_th_sum += int((v > (VTH - 2.0)).sum())

            tick += 1
            if tick % PRINT_EVERY == 0:
                wall = time.perf_counter() - t0
                bio = tick * DT
                n_grn = max(len(grn_i), 1)
                grn_each = grn_spikes_win / n_grn
                net_hz = net_spikes_win / n
                v_mean = v_sum / PRINT_EVERY
                g_mean = g_abs_sum / PRINT_EVERY
                near_pct = 100.0 * (near_th_sum / PRINT_EVERY) / n
                top = ""
                if net_spikes_win:
                    names = pd.Series(types[fired]).value_counts().head(6)
                    top = "  " + ", ".join(f"{k}:{v}" for k, v in names.items())
                print(
                    f"t={bio:6.1f}s  wall={wall:5.1f}s  "
                    f"GRN {grn_spikes_win:4d}/s "
                    f"({grn_each:.2f} Hz/cell, n={len(grn_i)})  "
                    f"net {net_spikes_win:6d}/s ({net_hz*1000:.3f} mHz/cell)  "
                    f"Dandelion {dand_spikes_win:3d}/s  MN9 {mn9_spikes_win:3d}/s"
                )
                print(
                    f"         activity  Vmean={v_mean:7.2f} mV  "
                    f"|g|mean={g_mean:.4f} mV  |g|max={g_max:.2f} mV  "
                    f"near_th={near_pct:.4f}%  "
                    f"{'QUIET' if net_spikes_win == 0 else 'SPIKING'}"
                    f"{top}"
                )
                print_top_motor(
                    nodes, mn_i, mn_spikes_win, mn_g_abs_sum, v, PRINT_EVERY
                )
                grn_spikes_win = net_spikes_win = mn9_spikes_win = dand_spikes_win = 0
                mn_spikes_win[:] = 0
                mn_g_abs_sum[:] = 0
                v_sum = g_abs_sum = g_max = 0.0
                near_th_sum = 0
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
