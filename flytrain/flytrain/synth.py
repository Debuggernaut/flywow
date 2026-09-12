"""Tiny synthetic connectome + patch labels so the pipeline can run without feathers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse

from .connectome import Connectome, attach_hex_pixels, pool_masks
from .constants import BUTTONS, FOV_PX, MIN_WEIGHT
from .vision import make_patch_fov


def make_synthetic_connectome(seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    body = 1

    def add(n, **kw):
        nonlocal body
        if "class_" in kw:
            kw["class"] = kw.pop("class_")
        out = []
        for _ in range(n):
            rec = {"body": body}
            rec.update(kw)
            rows.append(rec)
            out.append(body)
            body += 1
        return out

    hex_bodies = []
    for side, h1_off in (("L", 0), ("R", 20)):
        for i in range(8):
            for j in range(8):
                typ = ["Tm1", "Tm2", "Dm3", "Pm1"][(i + j) % 4]
                b = add(
                    1,
                    type=typ,
                    superclass="ol_intrinsic",
                    class_="visual",
                    subclass="columnar",
                    somaSide=side,
                    exitNerve="",
                    assignedOlHex1=h1_off + i,
                    assignedOlHex2=j,
                    consensus_nt="acetylcholine",
                    predicted_nt="acetylcholine",
                    predicted_nt_confidence=1.0,
                    celltype_predicted_nt="acetylcholine",
                    sign=1,
                )[0]
                hex_bodies.append(b)

    vpn = []
    for side in ("L", "R"):
        for t in ("LPLC1", "LC10", "LC4", "LT1"):
            vpn += add(
                4,
                type=t,
                superclass="visual_projection",
                class_="visual",
                subclass="vpn",
                somaSide=side,
                exitNerve="",
                assignedOlHex1=np.nan,
                assignedOlHex2=np.nan,
                consensus_nt="acetylcholine",
                predicted_nt="acetylcholine",
                predicted_nt_confidence=1.0,
                celltype_predicted_nt="acetylcholine",
                sign=1,
            )

    dn = []
    for side in ("L", "R"):
        for t in ("DNa02", "DNg100", "DNp11"):
            dn += add(
                3,
                type=t,
                superclass="descending_neuron",
                class_="descending",
                subclass="dn",
                somaSide=side,
                exitNerve="",
                assignedOlHex1=np.nan,
                assignedOlHex2=np.nan,
                consensus_nt="acetylcholine",
                predicted_nt="acetylcholine",
                predicted_nt_confidence=1.0,
                celltype_predicted_nt="acetylcholine",
                sign=1,
            )

    vnc_int = add(
        12,
        type="VInt",
        superclass="vnc_intrinsic",
        class_="intrinsic",
        subclass="vint",
        somaSide="L",
        exitNerve="",
        assignedOlHex1=np.nan,
        assignedOlHex2=np.nan,
        consensus_nt="gaba",
        predicted_nt="gaba",
        predicted_nt_confidence=1.0,
        celltype_predicted_nt="gaba",
        sign=-1,
    )

    pool_spec = [
        ("LF", "fl", "L", ""),
        ("RF", "fl", "R", ""),
        ("LM", "ml", "L", ""),
        ("RM", "ml", "R", ""),
        ("LH", "hl", "L", "MetaLN"),
        ("RH", "hl", "R", "MetaLN"),
    ]
    motors = {name: [] for name, *_ in pool_spec}
    for name, sub, side, nerve in pool_spec:
        motors[name] = add(
            6,
            type=f"MN_{name}",
            superclass="vnc_motor",
            class_="motor",
            subclass=sub,
            somaSide=side,
            exitNerve=nerve if nerve else "ProLN",
            assignedOlHex1=np.nan,
            assignedOlHex2=np.nan,
            consensus_nt="acetylcholine",
            predicted_nt="acetylcholine",
            predicted_nt_confidence=1.0,
            celltype_predicted_nt="acetylcholine",
            sign=1,
        )

    add(
        8,
        type="KC",
        superclass="central_brain_intrinsic",
        class_="Kenyon_Cell",
        subclass="kc",
        somaSide="L",
        exitNerve="",
        assignedOlHex1=np.nan,
        assignedOlHex2=np.nan,
        consensus_nt="acetylcholine",
        predicted_nt="acetylcholine",
        predicted_nt_confidence=1.0,
        celltype_predicted_nt="acetylcholine",
        sign=1,
    )

    nodes = pd.DataFrame(rows)
    idx = {int(b): i for i, b in enumerate(nodes["body"])}

    def connect(pres, posts, w=8, p=0.4):
        rec = []
        for pre in pres:
            for post in posts:
                if pre == post:
                    continue
                if rng.random() < p:
                    rec.append({"pre": pre, "post": post, "weight": w + int(rng.integers(0, 6))})
        return rec

    edges = []
    nhex = len(hex_bodies)
    left_hex, right_hex = hex_bodies[: nhex // 2], hex_bodies[nhex // 2 :]
    left_vpn, right_vpn = vpn[: len(vpn) // 2], vpn[len(vpn) // 2 :]
    left_dn, right_dn = dn[: len(dn) // 2], dn[len(dn) // 2 :]
    edges += connect(left_hex, left_vpn, w=20, p=0.55)
    edges += connect(right_hex, right_vpn, w=20, p=0.55)
    edges += connect(left_vpn, left_dn, w=16, p=0.7)
    edges += connect(right_vpn, right_dn, w=16, p=0.7)
    edges += connect(left_dn, motors["LF"] + motors["LM"] + motors["LH"] + vnc_int, w=16, p=0.8)
    edges += connect(right_dn, motors["RF"] + motors["RM"] + motors["RH"] + vnc_int, w=16, p=0.8)
    edges += connect(left_dn, motors["RF"], p=0.05)
    edges += connect(right_dn, motors["LF"], p=0.05)

    edf = pd.DataFrame(edges)
    edf = edf[edf["weight"] >= MIN_WEIGHT]
    n = len(nodes)
    pre_i = edf["pre"].map(idx).to_numpy()
    post_i = edf["post"].map(idx).to_numpy()
    sign = nodes.set_index("body")["sign"].reindex(edf["pre"]).fillna(0).to_numpy()
    val = edf["weight"].to_numpy(dtype=np.float32) * sign.astype(np.float32)
    W = sparse.csr_matrix((val, (post_i, pre_i)), shape=(n, n), dtype=np.float32)

    hex_i = np.flatnonzero(nodes["superclass"].eq("ol_intrinsic"))
    net = Connectome(
        nodes=nodes,
        edges=edf,
        W=W,
        hex_i=hex_i,
        vpn_i=np.flatnonzero(nodes["superclass"].eq("visual_projection")),
        dn_i=np.flatnonzero(nodes["superclass"].eq("descending_neuron")),
        vnc_i=np.flatnonzero(nodes["superclass"].isin(["vnc_intrinsic", "vnc_motor"])),
        mn_i=np.flatnonzero(nodes["superclass"].isin(["vnc_motor", "cb_motor"])),
        kc_i=np.flatnonzero(nodes["class"].eq("Kenyon_Cell")),
        pools=pool_masks(nodes),
        types=nodes["type"].astype(str).to_numpy(),
        superclasses=nodes["superclass"].astype(str).to_numpy(),
    )
    net.pool_i = {k: np.flatnonzero(m) for k, m in net.pools.items()}
    attach_hex_pixels(net, FOV_PX)
    return net


PATCH_CENTERS = {
    "LF": (64, 80),
    "RF": (192, 80),
    "LM": (64, 128),
    "RM": (192, 128),
    "LH": (64, 176),
    "RH": (192, 176),
}


def _centers_from_net():
    net = make_synthetic_connectome(seed=0)
    px, py = net.hex_px, net.hex_py
    left = px < np.median(px)
    right = ~left
    top = py < np.quantile(py, 0.33)
    mid = (py >= np.quantile(py, 0.33)) & (py < np.quantile(py, 0.66))
    bot = py >= np.quantile(py, 0.66)
    def ctr(mask):
        if mask.sum() == 0:
            return (128.0, 128.0)
        return (float(px[mask].mean()), float(py[mask].mean()))
    return {
        "LF": ctr(left & top),
        "RF": ctr(right & top),
        "LM": ctr(left & mid),
        "RM": ctr(right & mid),
        "LH": ctr(left & bot),
        "RH": ctr(right & bot),
    }


def write_synthetic_labels(out_dir: Path, n_per_button: int = 4, seed: int = 0) -> Path:
    out_dir = Path(out_dir)
    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    k = 0
    centers = _centers_from_net()
    for name in BUTTONS:
        cx, cy = centers[name]
        for _ in range(n_per_button):
            jitter = rng.integers(-8, 9, size=2)
            fov = make_patch_fov(cx + int(jitter[0]), cy + int(jitter[1]), radius=16)
            fname = f"{k:04d}_{name}.png"
            Image.fromarray((fov * 255).astype(np.uint8)).save(frames / fname)
            rec = {b: 0 for b in BUTTONS}
            rec[name] = 1
            rec.update(
                {
                    "path": f"frames/{fname}",
                    "clip_id": f"{name}_{k}",
                    "t": 0.0,
                    "already_composited": 1,
                }
            )
            rows.append(rec)
            k += 1
    for i in range(max(2, n_per_button // 2)):
        fov = np.zeros((FOV_PX, FOV_PX, 3), dtype=np.float32)
        fname = f"{k:04d}_DARK.png"
        Image.fromarray((fov * 255).astype(np.uint8)).save(frames / fname)
        rec = {b: 0 for b in BUTTONS}
        rec.update(
            {
                "path": f"frames/{fname}",
                "clip_id": f"DARK_{k}",
                "t": 0.0,
                "already_composited": 1,
            }
        )
        rows.append(rec)
        k += 1
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "index.csv", index=False)
    return out_dir / "index.csv"
