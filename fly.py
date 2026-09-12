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

from pathlib import Path
import sys
sys.path.append(r"C:\Dev\flywow\flytrain")
from flytrain.bake import apply_type_gains

sys.path.append(r"C:\Dev\flywow\flytrain")
from flytrain.monitor import FlyMonitor

DATA_DIR = Path(r"C:\Dev\flywow\data")
GUIDANCE = Path(r"C:\Dev\flywow\guidance.png")
#GUIDANCE = Path(r"C:\Dev\flywow\noguidance.png")
MIN_WEIGHT = 5

TICK_HZ = 100
INNER_STEPS = 2
INNER_DT = 0.01 / INNER_STEPS
DT = INNER_DT
PRINT_EVERY = TICK_HZ
MAX_SECONDS = 10
RECORD_KC = False
RECORD_HZ = 10  # 100 ms windows
OUT_DIR = Path(r"C:\Dev\flywow\leg_record")

GRN_HZ = 91.0
VISION_HZ_MAX = 80.0
FOV_PX = 256
FOV_SHIFT_X_FRAC = 0.18
FOV_SHIFT_Y_FRAC = 0.0

DRIVE_PR = False
DRIVE_VPN = True
DRIVE_GRN = False

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


def hex_visual_mask(nodes: pd.DataFrame) -> np.ndarray:
    """Cells that actually sit on the retinal lattice.

    assignedOlHex* is filled for photoreceptors and columnar optic-lobe
    neurons. Most VPNs do not have a hex, which is why VPNdrv went to 0.
    """
    h1 = pd.to_numeric(nodes.get("assignedOlHex1"), errors="coerce")
    has_hex = h1.notna()
    sc = nodes.superclass.astype(str)
    visual_sc = sc.isin(
        ["ol_sensory", "ol_intrinsic", "visual_projection", "visual_centrifugal"]
    )
    m = has_hex & visual_sc
    print(
        "hex visual drive  "
        + ", ".join(
            f"{name}={int((has_hex & sc.eq(name)).sum())}"
            for name in [
                "ol_sensory",
                "ol_intrinsic",
                "visual_projection",
                "visual_centrifugal",
            ]
        )
        + f"  total={int(m.sum())}"
    )
    return m.to_numpy()


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
    px = px + canvas * float(FOV_SHIFT_X_FRAC)
    py = py + canvas * float(FOV_SHIFT_Y_FRAC)
    px = np.where(np.isfinite(px), px, canvas / 2.0)
    py = np.where(np.isfinite(py), py, canvas / 2.0)
    px = np.clip(px, 0.0, canvas - 1e-3)
    py = np.clip(py, 0.0, canvas - 1e-3)
    return px, py


def fit_frame_to_fov_np(rgb, canvas: int):
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    ch = 1 if rgb.ndim < 3 else rgb.shape[2]
    scale = min(canvas / w, canvas / h)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    import cv2
    small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((canvas, canvas, ch), dtype=np.uint8)
    y0 = (canvas - nh) // 2
    x0 = (canvas - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out

class ScreenEye:
    def __init__(self, nodes: pd.DataFrame, pr_i: np.ndarray, vpn_i: np.ndarray) -> None:
        import dxcam
        from PIL import Image

        self.Image = Image
        self.cam = dxcam.create(output_color="RGB")
        if self.cam is None:
            raise RuntimeError("dxcam.create failed")
        
        self.overlay_fov = None
        if GUIDANCE.exists():
            raw = np.asarray(Image.open(GUIDANCE).convert("RGBA"))
            self.overlay_fov = fit_frame_to_fov_np(raw, FOV_PX).astype(np.float32) / 255.0
            print(f"vision overlay {GUIDANCE} → {FOV_PX}px  {self.overlay_fov.shape}")

        self.pr_i = pr_i
        self.vpn_i = vpn_i
        self.pr_rate = np.zeros(len(pr_i), dtype=np.float32)
        self.vpn_rate = np.zeros(len(vpn_i), dtype=np.float32)
        self.feat = {"left": 0.0, "right": 0.0, "mean": 0.0, "center": 0.0}

        if len(pr_i):
            sub = nodes.iloc[pr_i]
            self.kind = channel_kind(sub["type"].astype(str).to_numpy())
            h1 = pd.to_numeric(sub.get("assignedOlHex1"), errors="coerce").to_numpy()
            h2 = pd.to_numeric(sub.get("assignedOlHex2"), errors="coerce").to_numpy()
            self.px, self.py = hex_pixel_xy(h1, h2, FOV_PX)
            print(
                f"photoreceptors {len(pr_i)}  "
                f"luma={int((self.kind==0).sum())} UV={int((self.kind==1).sum())} "
                f"B={int((self.kind==2).sum())} G={int((self.kind==3).sum())}"
            )
        else:
            self.kind = np.zeros(0, dtype=np.int8)
            self.px = self.py = np.zeros(0)

        if len(vpn_i):
            vsub = nodes.iloc[vpn_i]
            self.vpn_type = vsub["type"].astype(str).str.upper().to_numpy()
            self.vpn_side = vsub["somaSide"].astype(str).to_numpy()
            vh1 = pd.to_numeric(vsub.get("assignedOlHex1"), errors="coerce").to_numpy()
            vh2 = pd.to_numeric(vsub.get("assignedOlHex2"), errors="coerce").to_numpy()
            self.vpn_px, self.vpn_py = hex_pixel_xy(vh1, vh2, FOV_PX)
            self.vpn_has_hex = np.isfinite(vh1) & np.isfinite(vh2)
            print(f"VPNs driven {len(vpn_i)}")
            print(vsub["type"].value_counts().head(12).to_string())
        else:
            self.vpn_type = self.vpn_side = np.array([])
            self.vpn_px = self.vpn_py = np.zeros(0)
            self.vpn_has_hex = np.zeros(0, dtype=bool)

    def grab(self) -> None:
        frame = self.cam.grab(new_frame_only=True)
        if frame is None:
            return
        fov_u8 = fit_frame_to_fov_np(frame, FOV_PX)
        fov = fov_u8.astype(np.float32) * np.float32(1.0 / 255.0)
        if self.overlay_fov is not None:
            a = self.overlay_fov[:, :, 3:4]
            fov = fov * (1.0 - a) + self.overlay_fov[:, :, :3] * a
        gray = fov.mean(axis=2)
        self.gray = gray
        h, w = gray.shape
        left = float(gray[:, : w // 2].mean())
        right = float(gray[:, w // 2 :].mean())
        mid = float(gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean())
        mean = float(gray.mean())
        self.feat = {"left": left, "right": right, "mean": mean, "center": mid}

        if len(self.pr_i):
            xi = np.clip(self.px.astype(int), 0, FOV_PX - 1)
            yi = np.clip(self.py.astype(int), 0, FOV_PX - 1)
            rgb = fov[yi, xi]
            uv, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
            luma = (uv + green + blue) / 3.0
            ch = np.empty(len(self.pr_i), dtype=np.float32)
            ch[self.kind == 0] = luma[self.kind == 0]
            ch[self.kind == 1] = uv[self.kind == 1]
            ch[self.kind == 2] = blue[self.kind == 2]
            ch[self.kind == 3] = green[self.kind == 3]
            self.pr_rate = ch * np.float32(VISION_HZ_MAX)

        if len(self.vpn_i):
            # Unmodified FOV: each VPN with a hex samples that pixel only.
            # No left/right/loom scalars — those made the legs move as a block.
            rate = np.zeros(len(self.vpn_i), dtype=np.float32)
            xi = np.clip(self.vpn_px.astype(int), 0, FOV_PX - 1)
            yi = np.clip(self.vpn_py.astype(int), 0, FOV_PX - 1)
            rate[:] = gray[yi, xi]
            rate[~self.vpn_has_hex] = 0.0
            self.vpn_rate = rate * np.float32(VISION_HZ_MAX)

        self.gray = gray


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

    _gains = Path(r"C:\Dev\flywow\flytrain\train_out\gains_type.csv")
    if _gains.exists():
        W = apply_type_gains(W, nodes, _gains).tocsc()
        print(f"loaded type-gains {_gains}")

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

    pr_i = np.flatnonzero(photoreceptor_mask(nodes)) if DRIVE_PR else np.array([], dtype=int)
    vpn_drv_i = np.flatnonzero(hex_visual_mask(nodes)) if DRIVE_VPN else np.array([], dtype=int)
    eye = ScreenEye(nodes, pr_i, vpn_drv_i) if (DRIVE_PR or DRIVE_VPN) else None

    mn9_i = np.flatnonzero(nodes["type"].astype(str).eq("MN9"))
    dand_i = np.flatnonzero(nodes["type"].astype(str).eq("AN13B002"))
    l1_i = np.flatnonzero(nodes["type"].astype(str).str.fullmatch(r"L1|L1_.*", case=False))
    vpn_i = np.flatnonzero(nodes.superclass.eq("visual_projection"))
    dn_i = np.flatnonzero(nodes.superclass.eq("descending_neuron"))
    is_mn = nodes.superclass.isin(["vnc_motor", "cb_motor"]).to_numpy()
    mn_i = np.flatnonzero(is_mn)

    def leg_mask(sub: str, side: str, nerve: str | None = None) -> np.ndarray:
        m = (
            nodes.superclass.eq("vnc_motor")
            & nodes.subclass.eq(sub)
            & nodes.somaSide.eq(side)
        )
        if nerve:
            m &= nodes.exitNerve.eq(nerve)
        return m.to_numpy()

    pools = {
        "LF": leg_mask("fl", "L"),
        "RF": leg_mask("fl", "R"),
        "LM": leg_mask("ml", "L"),
        "RM": leg_mask("ml", "R"),
        "LH": leg_mask("hl", "L", "MetaLN"),
        "RH": leg_mask("hl", "R", "MetaLN"),
    }
    for name, m in pools.items():
        print(f"  pool {name} {int(m.sum())}")
    kc_i = np.flatnonzero(nodes["class"].astype(str).eq("Kenyon_Cell"))
    if len(kc_i) == 0:
        kc_i = np.flatnonzero(nodes["type"].astype(str).str.startswith("KC"))
    print(
        f"readout  MN9={len(mn9_i)}  Dandelion={len(dand_i)}  "
        f"L1={len(l1_i)}  VPN={len(vpn_i)}  drivenVPN={len(vpn_drv_i)}  "
        f"DN={len(dn_i)}  motor={len(mn_i)}  KC={len(kc_i)}"
    )
    if len(kc_i):
        print(nodes.iloc[kc_i]["type"].value_counts().head(12).to_string())


    mon = FlyMonitor(icons_dir=Path(r"C:\Dev\flywow\labels\icons"))
    pool_ui = {k: 0 for k in pools}

    rng = np.random.default_rng(0)
    v = np.full(n, VREST, dtype=np.float32)
    gsyn = np.zeros(n, dtype=np.float32)
    refr = np.zeros(n, dtype=np.float32)
    pending = np.empty(0, dtype=np.int32)

    types = nodes["type"].astype(str).to_numpy()
    leak = np.float32(np.exp(-INNER_DT / TAU_S))
    inner_per_sec = PRINT_EVERY * INNER_STEPS
    print(
        f"\nrunning {TICK_HZ} Hz outer / {INNER_STEPS}×{INNER_DT*1000:.1f} ms LIF  "
        f"VPN={DRIVE_VPN} PR={DRIVE_PR} GRN={DRIVE_GRN}  Ctrl+C to stop"
    )
    t0 = time.perf_counter()
    acc = {"grab": 0.0, "dump": 0.0, "lif": 0.0, "print": 0.0}
    tick = 0
    n_ticks = None if MAX_SECONDS is None else int(MAX_SECONDS * TICK_HZ)

    grn_spikes_win = 0
    vis_spikes_win = 0
    net_spikes_win = 0
    mn9_spikes_win = 0
    dand_spikes_win = 0
    l1_spikes_win = 0
    vpn_spikes_win = 0
    vpn_drv_spikes_win = 0
    dn_spikes_win = 0
    mn_spikes_win = np.zeros(n, dtype=np.int32)
    pool_spikes_win = {k: 0 for k in pools}
    mn_g_abs_sum = np.zeros(n, dtype=np.float64)
    v_sum = 0.0
    g_abs_sum = 0.0
    g_max = 0.0
    near_th_sum = 0
    vis_rate_mean = 0.0

    rec_every = max(1, TICK_HZ // RECORD_HZ)
    rec_dt = rec_every / TICK_HZ
    kc_win = np.zeros(len(kc_i), dtype=np.int32)
    kc_g_win = np.zeros(len(kc_i), dtype=np.float64)
    kc_v_win = np.zeros(len(kc_i), dtype=np.float64)
    rec_spikes = []
    rec_g = []
    rec_v = []
    rec_t = []
    rec_vis = []
    rec_nspk = []
    rec_inner = 0

    try:
        while n_ticks is None or tick < n_ticks:
            if eye is not None:
                eye.grab()
                vis_rate_mean += float(eye.feat.get("mean", 0.0)) * VISION_HZ_MAX

            last_fired = None
            for _ in range(INNER_STEPS):
                src = pending

                if DRIVE_GRN and len(grn_i):
                    grn_fire = rng.random(len(grn_i)) < (GRN_HZ * INNER_DT)
                    if grn_fire.any():
                        gi = grn_i[grn_fire].astype(np.int32, copy=False)
                        src = gi if src.size == 0 else np.unique(np.concatenate([src, gi]))
                        grn_spikes_win += int(grn_fire.sum())

                if eye is not None and DRIVE_PR and len(pr_i):
                    vis_fire = rng.random(len(pr_i)) < (eye.pr_rate * INNER_DT)
                    if vis_fire.any():
                        vi = pr_i[vis_fire].astype(np.int32, copy=False)
                        src = vi if src.size == 0 else np.unique(np.concatenate([src, vi]))
                        vis_spikes_win += int(vis_fire.sum())

                if eye is not None and DRIVE_VPN and len(vpn_drv_i):
                    vpn_fire = rng.random(len(vpn_drv_i)) < (eye.vpn_rate * INNER_DT)
                    if vpn_fire.any():
                        vi = vpn_drv_i[vpn_fire].astype(np.int32, copy=False)
                        src = vi if src.size == 0 else np.unique(np.concatenate([src, vi]))
                        vpn_drv_spikes_win += int(vpn_fire.sum())


                t = time.perf_counter(); eye.grab(); acc["grab"] += time.perf_counter() - t
                t = time.perf_counter()

                if src.size:
                    dump_spikes(gsyn, src)
                    
                acc["dump"] += time.perf_counter() - t

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

                acc["lif"] += time.perf_counter() - t

                net_spikes_win += int(fired.sum())
                if len(mn9_i):
                    mn9_spikes_win += int(fired[mn9_i].sum())
                if len(dand_i):
                    dand_spikes_win += int(fired[dand_i].sum())
                if len(l1_i):
                    l1_spikes_win += int(fired[l1_i].sum())
                if len(vpn_i):
                    vpn_spikes_win += int(fired[vpn_i].sum())
                if len(dn_i):
                    dn_spikes_win += int(fired[dn_i].sum())
                for pname, pmask in pools.items():
                    pool_spikes_win[pname] += int(fired[pmask].sum())
                if len(mn_i):
                    mn_spikes_win[mn_i] += fired[mn_i].astype(np.int32)
                    mn_g_abs_sum[mn_i] += np.abs(gsyn[mn_i])
                if RECORD_KC and len(kc_i):
                    kc_win += fired[kc_i].astype(np.int32)
                    kc_g_win += np.abs(gsyn[kc_i])
                    kc_v_win += v[kc_i]
                    rec_inner += 1

                for pname, pmask in pools.items():
                    nfire = int(fired[pmask].sum())
                    pool_spikes_win[pname] += nfire
                    pool_ui[pname] += nfire

            fired = last_fired if last_fired is not None else np.zeros(n, dtype=bool)
            v_sum += float(v.mean())
            g_abs_sum += float(np.abs(gsyn).mean())
            g_max = max(g_max, float(np.abs(gsyn).max()))
            near_th_sum += int((v > (VTH - 2.0)).sum())

            tick += 1
            if mon is not None and tick % 3 == 0:
                dt = 3.0 / TICK_HZ
                hz = {k: pool_ui[k] / max(int(pools[k].sum()), 1) / dt for k in pools}
                mon.push(hz, fov=getattr(eye, "gray", None), px=getattr(eye, "vpn_px", None), py=getattr(eye, "vpn_py", None))
                for k in pool_ui:
                    pool_ui[k] = 0
                
            if RECORD_KC and len(kc_i) and tick % rec_every == 0:
                denom = max(rec_inner, 1)
                rec_spikes.append(kc_win.copy())
                rec_g.append((kc_g_win / denom).astype(np.float32))
                rec_v.append((kc_v_win / denom).astype(np.float32))
                rec_t.append(tick / TICK_HZ)
                rec_vis.append(float(eye.feat["mean"]) if eye is not None else 0.0)
                rec_nspk.append(int((kc_win > 0).sum()))
                kc_win[:] = 0
                kc_g_win[:] = 0
                kc_v_win[:] = 0
                rec_inner = 0
            if tick % PRINT_EVERY == 0:
                s = sum(acc.values()) or 1e-9
                wall = time.perf_counter() - t0
                bio = tick / TICK_HZ
                n_vpn = max(len(vpn_drv_i), 1)
                net_hz = net_spikes_win / n
                v_mean = v_sum / PRINT_EVERY
                g_mean = g_abs_sum / PRINT_EVERY
                near_pct = 100.0 * (near_th_sum / PRINT_EVERY) / n
                feat = eye.feat if eye is not None else {}
                top = ""
                if net_spikes_win:
                    names = pd.Series(types[fired]).value_counts().head(6)
                    top = "  " + ", ".join(f"{k}:{v}" for k, v in names.items())

                t = time.perf_counter()
                print(
                    f"t={bio:6.1f}s  wall={wall:5.1f}s  "
                    f"feat L={feat.get('left',0):.2f} R={feat.get('right',0):.2f} "
                    f"C={feat.get('center',0):.2f}  "
                    f"hex_rate={float(eye.vpn_rate.mean()) if eye is not None and len(eye.vpn_rate) else 0:.2f}  "
                    f"VPNdrv {vpn_drv_spikes_win:5d}/s ({vpn_drv_spikes_win/n_vpn:.1f} Hz/cell)  "
                    f"VPN {vpn_spikes_win:5d}/s  DN {dn_spikes_win:4d}/s  "
                    f"net {net_spikes_win:6d}/s ({net_hz*1000:.2f} mHz/cell)"
                )
                print(
                    f"         activity  Vmean={v_mean:7.2f} mV  "
                    f"|g|mean={g_mean:.4f} mV  |g|max={g_max:.2f} mV  "
                    f"near_th={near_pct:.4f}%  "
                    f"{'QUIET' if net_spikes_win == 0 else 'SPIKING'}"
                    f"{top}"
                )
                legs = "  ".join(
                    f"{k}={pool_spikes_win[k]/max(int(pools[k].sum()),1):.2f}Hz"
                    for k in ("LF", "RF", "LM", "RM", "LH", "RH")
                )
                print(f"         legs    {legs}")
                print_top_motor(
                    nodes, mn_i, mn_spikes_win, mn_g_abs_sum, v, inner_per_sec
                )
                print("         profile " + "  ".join(f"{k}={100*v/s:.0f}%" for k, v in acc.items()))
                for k in acc:
                    acc[k] = 0.0
                acc["print"] += time.perf_counter() - t
                grn_spikes_win = vis_spikes_win = net_spikes_win = 0
                vpn_drv_spikes_win = 0
                dn_spikes_win = 0
                for k in pool_spikes_win:
                    pool_spikes_win[k] = 0
                mn9_spikes_win = dand_spikes_win = l1_spikes_win = vpn_spikes_win = 0
                mn_spikes_win[:] = 0
                mn_g_abs_sum[:] = 0
                v_sum = g_abs_sum = g_max = 0.0
                near_th_sum = 0
                vis_rate_mean = 0.0
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if RECORD_KC and rec_spikes:
            save_kc_record(
                nodes, kc_i, rec_t, rec_spikes, rec_g, rec_v, rec_vis, rec_nspk, rec_dt
            )


def save_kc_record(
    nodes, kc_i, rec_t, rec_spikes, rec_g, rec_v, rec_vis, rec_nspk, rec_dt
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    spikes = np.stack(rec_spikes, axis=0).astype(np.int16)
    gmean = np.stack(rec_g, axis=0).astype(np.float32)
    vmean = np.stack(rec_v, axis=0).astype(np.float32)
    hz = spikes.astype(np.float32) / np.float32(rec_dt)
    t = np.asarray(rec_t, dtype=np.float32)
    vis = np.asarray(rec_vis, dtype=np.float32)
    nspk = np.asarray(rec_nspk, dtype=np.int32)
    bodies = nodes.iloc[kc_i]["body"].to_numpy()
    ktypes = nodes.iloc[kc_i]["type"].astype(str).to_numpy()

    npz = OUT_DIR / "kc_trace.npz"
    np.savez_compressed(
        npz,
        t=t,
        spikes=spikes,
        hz=hz,
        g_mean=gmean,
        v_mean=vmean,
        vis_drive_hz=vis,
        n_kc_spiking=nspk,
        body=bodies,
        type=ktypes,
        window_s=np.array(rec_dt, dtype=np.float32),
    )

    cells = pd.DataFrame({"col": np.arange(len(kc_i)), "body": bodies, "type": ktypes})
    cells.to_csv(OUT_DIR / "kc_cells.csv", index=False)

    win = pd.DataFrame(
        {
            "t": t,
            "vis_drive_hz": vis,
            "n_kc_spiking": nspk,
            "kc_spikes_total": spikes.sum(axis=1),
            "kc_mean_hz": hz.mean(axis=1),
            "kc_mean_abs_g": gmean.mean(axis=1),
            "kc_max_abs_g": gmean.max(axis=1),
            "kc_mean_v": vmean.mean(axis=1),
        }
    )
    win.to_csv(OUT_DIR / "kc_windows.csv", index=False)

    print(f"\nKC record  windows={len(t)}  cells={len(kc_i)}  dt={rec_dt:.3f}s")
    print(f"  {npz}")
    print(f"  {OUT_DIR / 'kc_windows.csv'}")
    print(
        win[
            [
                "t",
                "vis_drive_hz",
                "n_kc_spiking",
                "kc_mean_abs_g",
                "kc_max_abs_g",
                "kc_mean_v",
            ]
        ].to_string(index=False)
    )
    if nspk.max() == 0 and float(gmean.max()) < 0.05:
        print("  NOTE: no KC spikes and ~0 synaptic current. Nothing is hitting the calyx.")
    elif nspk.max() == 0:
        print("  NOTE: KCs get current but stay under threshold. Use g_mean as Stage A input.")


if __name__ == "__main__":
    main()
