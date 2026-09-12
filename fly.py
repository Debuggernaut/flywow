#!/usr/bin/env python3
"""MaleCNS 100 Hz LIF loop with live screen vision + sugar GRNs.

Not a fly. A live cartoon of the published graph.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path(r"C:\Dev\flywow\data")
GUIDANCE = Path(r"C:\Dev\flywow\guidance.png")
MIN_WEIGHT = 5

TICK_HZ = 100
INNER_STEPS = 2
INNER_DT = 0.01 / INNER_STEPS
DT = INNER_DT
PRINT_EVERY = TICK_HZ
MAX_SECONDS = 30

GRN_HZ = 91.0
VISION_HZ_MAX = 80.0
FOV_PX = 256
DRIVE_VISION = True
DRIVE_GRN = True

TAU_M = 0.020
TAU_S = 0.005
VREST = -52.0
VTH = -45.0
VRESET = -52.0
TREF = 0.0022
WSYN = 0.275
GAIN = 0.2

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
    print("falling back to all class==gustatory")
    return nodes["class"].eq("gustatory").to_numpy()


def photoreceptor_mask(nodes: pd.DataFrame) -> np.ndarray:
    vis = (nodes.superclass == "ol_sensory") & (nodes["class"] == "visual")
    t = nodes["type"].astype(str)
    looks_r = t.str.contains(r"^R[1-8]|R1-6|photoreceptor", case=False, na=False)
    m = vis & looks_r
    if int(m.sum()) >= 100:
        return m.to_numpy()
    print("photoreceptor type strings thin; using all visual ol_sensory")
    return vis.to_numpy()


def channel_kind(types: np.ndarray) -> np.ndarray:
    """0=luma (R1-6), 1=uv (R7), 2=blue (R8p), 3=green (R8y)."""
    out = np.zeros(len(types), dtype=np.int8)
    for i, raw in enumerate(types):
        t = str(raw).upper()
        if "R7" in t:
            out[i] = 1
        elif "R8" in t:
            if any(s in t for s in ("Y", "YELLOW", "RH6")):
                out[i] = 3
            else:
                out[i] = 2
        else:
            out[i] = 0
    return out


def hex_pixel_xy(h1: np.ndarray, h2: np.ndarray, canvas: int) -> tuple[np.ndarray, np.ndarray]:
    """Map assignedOlHex1/2 into a square canvas. Missing hex -> center."""
    x = h2.astype(float) - h1.astype(float)
    y = h1.astype(float) + h2.astype(float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 10:
        return np.full(len(h1), canvas / 2.0), np.full(len(h1), canvas / 2.0)
    x = np.where(valid, x, np.nan)
    y = np.where(valid, y, np.nan)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    px = (x - xmin) / span * (canvas * 0.92) + canvas * 0.04
    py = (y - ymin) / span * (canvas * 0.92) + canvas * 0.04
    px = np.where(np.isfinite(px), px, canvas / 2.0)
    py = np.where(np.isfinite(py), py, canvas / 2.0)
    return px, py


def fit_frame_to_fov(im, canvas: int):
    from PIL import Image

    scale = min(canvas / im.width, canvas / im.height)
    nw = max(1, int(im.width * scale))
    nh = max(1, int(im.height * scale))
    small = im.resize((nw, nh), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (canvas, canvas), (0, 0, 0))
    out.paste(small, ((canvas - nw) // 2, (canvas - nh) // 2))
    return out


class ScreenEye:
    def __init__(self, nodes: pd.DataFrame, pr_i: np.ndarray) -> None:
        import dxcam
        from PIL import Image

        self.Image = Image
        self.cam = dxcam.create(output_color="RGB")
        if self.cam is None:
            raise RuntimeError("dxcam.create failed")
        self.overlay = None
        #if GUIDANCE.exists():
        #    self.overlay = Image.open(GUIDANCE).convert("RGBA")
        #    print(f"vision overlay {GUIDANCE}")

        self.pr_i = pr_i
        sub = nodes.iloc[pr_i]
        self.kind = channel_kind(sub["type"].astype(str).to_numpy())
        h1 = pd.to_numeric(sub.get("assignedOlHex1"), errors="coerce").to_numpy()
        h2 = pd.to_numeric(sub.get("assignedOlHex2"), errors="coerce").to_numpy()
        self.px, self.py = hex_pixel_xy(h1, h2, FOV_PX)
        self.rate = np.zeros(len(pr_i), dtype=np.float32)
        print(
            f"photoreceptors {len(pr_i)}  "
            f"luma={int((self.kind==0).sum())}  "
            f"UV={int((self.kind==1).sum())}  "
            f"B={int((self.kind==2).sum())}  "
            f"G={int((self.kind==3).sum())}"
        )

    def grab_rates(self) -> np.ndarray:
        frame = self.cam.grab(new_frame_only=False)
        if frame is None:
            return self.rate
        im = self.Image.fromarray(frame).convert("RGBA")
        if self.overlay is not None:
            ov = self.overlay
            if ov.size != im.size:
                ov = ov.resize(im.size, self.Image.Resampling.NEAREST)
                self.overlay = ov
            im = self.Image.alpha_composite(im, ov)
        fov = np.asarray(fit_frame_to_fov(im.convert("RGB"), FOV_PX), dtype=np.float32)
        xi = np.clip(self.px.astype(int), 0, FOV_PX - 1)
        yi = np.clip(self.py.astype(int), 0, FOV_PX - 1)
        rgb = fov[yi, xi] / 255.0
        uv, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        luma = (uv + green + blue) / 3.0
        ch = np.empty(len(self.pr_i), dtype=np.float32)
        ch[self.kind == 0] = luma[self.kind == 0]
        ch[self.kind == 1] = uv[self.kind == 1]
        ch[self.kind == 2] = blue[self.kind == 2]
        ch[self.kind == 3] = green[self.kind == 3]
        self.rate = ch * np.float32(VISION_HZ_MAX)
        return self.rate


def print_top_motor(
    nodes: pd.DataFrame,
    mn_i: np.ndarray,
    mn_spikes_win: np.ndarray,
    mn_g_abs_sum: np.ndarray,
    v: np.ndarray,
    window_ticks: int,
    k: int = 10,
) -> None:
    if len(mn_i) == 0:
        print("         motor    none annotated")
        return
    spikes = mn_spikes_win[mn_i]
    g_mean = mn_g_abs_sum[mn_i] / max(window_ticks, 1)
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

    W = build_W(nodes, edges).tocsc()
    n = W.shape[0]
    indptr = W.indptr
    indices = W.indices
    data = W.data
    print(f"W shape {W.shape}  nnz {W.nnz:,}  (CSC spike gather)")

    def dump_spikes(g: np.ndarray, fired_idx: np.ndarray) -> None:
        scale = np.float32(WSYN * GAIN)
        for i in fired_idx:
            a = int(indptr[i])
            b = int(indptr[i + 1])
            if a == b:
                continue
            g[indices[a:b]] += scale * data[a:b]

    is_grn = sugar_water_mask(nodes)
    grn_i = np.flatnonzero(is_grn)
    print(f"sugar/water GRNs driven at {GRN_HZ} Hz: {len(grn_i)}")
    print(nodes.loc[is_grn, "type"].value_counts().head(10).to_string())

    pr_i = np.flatnonzero(photoreceptor_mask(nodes))
    eye = ScreenEye(nodes, pr_i) if DRIVE_VISION else None

    mn9_i = np.flatnonzero(nodes["type"].astype(str).eq("MN9"))
    dand_i = np.flatnonzero(nodes["type"].astype(str).eq("AN13B002"))
    l1_i = np.flatnonzero(nodes["type"].astype(str).str.fullmatch(r"L1|L1_.*", case=False))
    vpn_i = np.flatnonzero(nodes.superclass.eq("visual_projection"))
    is_mn = nodes.superclass.isin(["vnc_motor", "cb_motor"]).to_numpy()
    mn_i = np.flatnonzero(is_mn)
    print(
        f"readout  MN9={len(mn9_i)}  Dandelion={len(dand_i)}  "
        f"L1={len(l1_i)}  VPN={len(vpn_i)}  motor={len(mn_i)}"
    )

    rng = np.random.default_rng(0)
    v = np.full(n, VREST, dtype=np.float32)
    gsyn = np.zeros(n, dtype=np.float32)
    refr = np.zeros(n, dtype=np.float32)
    pending = np.empty(0, dtype=np.int32)

    types = nodes["type"].astype(str).to_numpy()
    leak = np.float32(np.exp(-INNER_DT / TAU_S))
    inner_per_sec = PRINT_EVERY * INNER_STEPS
    vis_rate = np.zeros(len(pr_i), dtype=np.float32)

    print(
        f"\nrunning {TICK_HZ} Hz outer / {INNER_STEPS}×{INNER_DT*1000:.1f} ms LIF  "
        f"vision={'on' if eye else 'off'}  Ctrl+C to stop"
    )
    t0 = time.perf_counter()
    tick = 0
    n_ticks = None if MAX_SECONDS is None else int(MAX_SECONDS * TICK_HZ)

    grn_spikes_win = 0
    vis_spikes_win = 0
    net_spikes_win = 0
    mn9_spikes_win = 0
    dand_spikes_win = 0
    l1_spikes_win = 0
    vpn_spikes_win = 0
    mn_spikes_win = np.zeros(n, dtype=np.int32)
    mn_g_abs_sum = np.zeros(n, dtype=np.float64)
    v_sum = 0.0
    g_abs_sum = 0.0
    g_max = 0.0
    near_th_sum = 0
    vis_rate_mean = 0.0

    try:
        while n_ticks is None or tick < n_ticks:
            if eye is not None:
                vis_rate = eye.grab_rates()
                vis_rate_mean += float(vis_rate.mean())

            last_fired = None
            for _ in range(INNER_STEPS):
                src = pending

                if DRIVE_GRN and len(grn_i):
                    grn_fire = rng.random(len(grn_i)) < (GRN_HZ * INNER_DT)
                    if grn_fire.any():
                        gi = grn_i[grn_fire].astype(np.int32, copy=False)
                        src = gi if src.size == 0 else np.unique(np.concatenate([src, gi]))
                        grn_spikes_win += int(grn_fire.sum())

                if eye is not None and len(pr_i):
                    vis_fire = rng.random(len(pr_i)) < (vis_rate * INNER_DT)
                    if vis_fire.any():
                        vi = pr_i[vis_fire].astype(np.int32, copy=False)
                        src = vi if src.size == 0 else np.unique(np.concatenate([src, vi]))
                        vis_spikes_win += int(vis_fire.sum())

                if src.size:
                    dump_spikes(gsyn, src)
                gsyn *= leak

                active = refr <= 0
                v[active] += np.float32(INNER_DT) * ((VREST - v[active]) / TAU_M) + gsyn[active]
                v[~active] = VRESET
                refr = np.maximum(0.0, refr - INNER_DT)

                fired = active & (v >= VTH)
                last_fired = fired
                if fired.any():
                    v[fired] = VRESET
                    gsyn[fired] = 0.0
                    refr[fired] = TREF
                    pending = np.flatnonzero(fired).astype(np.int32)
                else:
                    pending = np.empty(0, dtype=np.int32)

                net_spikes_win += int(fired.sum())
                if len(mn9_i):
                    mn9_spikes_win += int(fired[mn9_i].sum())
                if len(dand_i):
                    dand_spikes_win += int(fired[dand_i].sum())
                if len(l1_i):
                    l1_spikes_win += int(fired[l1_i].sum())
                if len(vpn_i):
                    vpn_spikes_win += int(fired[vpn_i].sum())
                if len(mn_i):
                    mn_spikes_win[mn_i] += fired[mn_i].astype(np.int32)
                    mn_g_abs_sum[mn_i] += np.abs(gsyn[mn_i])

            fired = last_fired if last_fired is not None else np.zeros(n, dtype=bool)
            v_sum += float(v.mean())
            g_abs_sum += float(np.abs(gsyn).mean())
            g_max = max(g_max, float(np.abs(gsyn).max()))
            near_th_sum += int((v > (VTH - 2.0)).sum())

            tick += 1
            if tick % PRINT_EVERY == 0:
                wall = time.perf_counter() - t0
                bio = tick / TICK_HZ
                n_grn = max(len(grn_i), 1)
                n_pr = max(len(pr_i), 1)
                net_hz = net_spikes_win / n
                v_mean = v_sum / PRINT_EVERY
                g_mean = g_abs_sum / PRINT_EVERY
                near_pct = 100.0 * (near_th_sum / PRINT_EVERY) / n
                vr = vis_rate_mean / PRINT_EVERY
                top = ""
                if net_spikes_win:
                    names = pd.Series(types[fired]).value_counts().head(6)
                    top = "  " + ", ".join(f"{k}:{v}" for k, v in names.items())
                print(
                    f"t={bio:6.1f}s  wall={wall:5.1f}s  "
                    f"GRN {grn_spikes_win:4d}/s ({grn_spikes_win/n_grn:.1f} Hz/cell)  "
                    f"PR {vis_spikes_win:5d}/s ({vis_spikes_win/n_pr:.1f} Hz/cell, drive={vr:.1f} Hz)  "
                    f"L1 {l1_spikes_win:4d}/s  VPN {vpn_spikes_win:5d}/s  "
                    f"Dandelion {dand_spikes_win:3d}/s  MN9 {mn9_spikes_win:3d}/s  "
                    f"net {net_spikes_win:6d}/s ({net_hz*1000:.2f} mHz/cell)"
                )
                print(
                    f"         activity  Vmean={v_mean:7.2f} mV  "
                    f"|g|mean={g_mean:.4f} mV  |g|max={g_max:.2f} mV  "
                    f"near_th={near_pct:.4f}%  "
                    f"{'QUIET' if net_spikes_win == 0 else 'SPIKING'}"
                    f"{top}"
                )
                print_top_motor(
                    nodes, mn_i, mn_spikes_win, mn_g_abs_sum, v, inner_per_sec
                )
                grn_spikes_win = vis_spikes_win = net_spikes_win = 0
                mn9_spikes_win = dand_spikes_win = l1_spikes_win = vpn_spikes_win = 0
                mn_spikes_win[:] = 0
                mn_g_abs_sum[:] = 0
                v_sum = g_abs_sum = g_max = 0.0
                near_th_sum = 0
                vis_rate_mean = 0.0
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
