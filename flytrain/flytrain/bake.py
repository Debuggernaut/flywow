"""Stage C: write θ back as a sidecar the live loop can multiply onto W.data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .connectome import Connectome
from .constants import (
    FOV_PX,
    GAIN,
    INNER_DT,
    INNER_STEPS,
    MIN_WEIGHT,
    TAU_M,
    TAU_S,
    TICK_HZ,
    TREF,
    VISION_HZ_MAX,
    VRESET,
    VREST,
    VTH,
    WINDOW_INNER_STEPS,
    WSYN,
)


README_TEMPLATE = """flytrain output
================

This folder holds type-gains for existing MaleCNS synapses.
Sparsity and signs are frozen. Global GAIN stays {gain}.

LIF constants (must match fly.py)
---------------------------------
TICK_HZ={tick}
INNER_STEPS={inner_steps}
INNER_DT={inner_dt}
WINDOW_INNER_STEPS={win}
TAU_M={tau_m}  TAU_S={tau_s}
VREST={vrest}  VTH={vth}  VRESET={vreset}  TREF={tref}
WSYN={wsyn}  GAIN={gain}  MIN_WEIGHT={min_w}
FOV_PX={fov}  VISION_HZ_MAX={vis}

How fly.py should apply gains
-----------------------------
After build_W(...).tocsc():

    from flytrain.bake import apply_type_gains
    apply_type_gains(W, nodes, "gains_type.csv")

apply_type_gains multiplies W.data on matching (type_pre, type_post) edges
in the three allowed cuts by exp(θ). Unlisted types keep gain 1.

Files
-----
gains_type.csv     type_pre, type_post, exp_theta, cut
metrics.json       val BCE, per-button AP, dark/white probes
cache/             Stage A LIF rate vectors (optional)

What this is not
----------------
Not a biophysical fly. Not a gait CPG. Not a pixel classifier.
Synapse `weight` is a count × guessed NT sign × learned scale.
Kenyon cells / mushroom body were not trained.
"""


def write_readme(out_dir: Path, extra: str = "") -> None:
    text = README_TEMPLATE.format(
        tick=TICK_HZ,
        inner_steps=INNER_STEPS,
        inner_dt=INNER_DT,
        win=WINDOW_INNER_STEPS,
        tau_m=TAU_M,
        tau_s=TAU_S,
        vrest=VREST,
        vth=VTH,
        vreset=VRESET,
        tref=TREF,
        wsyn=WSYN,
        gain=GAIN,
        min_w=MIN_WEIGHT,
        fov=FOV_PX,
        vis=VISION_HZ_MAX,
    )
    if extra:
        text += "\n" + extra + "\n"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "README.txt").write_text(text)


def apply_type_gains(
    W,
    nodes: pd.DataFrame,
    gains_csv,
    cuts_only: bool = True,
):
    """In-place multiply CSC/CSR W.data by exp(θ) for matching type pairs.

    `W` must be the matrix from build_W (post, pre). We iterate CSC columns
    so column index = pre neuron.
    """
    gains = pd.read_csv(gains_csv)
    pair = {}
    pre_only = {}
    for _, row in gains.iterrows():
        a = str(row["type_pre"])
        b = str(row["type_post"])
        g = float(row["exp_theta"])
        if b in {"*", "nan", "None"}:
            pre_only[a] = g
        else:
            pair[(a, b)] = g

    allowed_pre = {
        "ol_intrinsic",
        "visual_projection",
        "descending_neuron",
    }
    allowed_post = {
        "visual_projection",
        "descending_neuron",
        "vnc_intrinsic",
        "vnc_motor",
    }

    types = nodes["type"].astype(str).to_numpy()
    sc = nodes["superclass"].astype(str).to_numpy()
    Wc = W.tocsc()
    n_changed = 0
    for pre in range(Wc.shape[1]):
        a, b = Wc.indptr[pre], Wc.indptr[pre + 1]
        if a == b:
            continue
        if cuts_only and sc[pre] not in allowed_pre:
            continue
        tp = types[pre]
        dest = Wc.indices[a:b]
        scale = np.ones(b - a, dtype=np.float32)
        for k, post in enumerate(dest):
            if cuts_only and sc[post] not in allowed_post:
                continue
            tpost = types[post]
            if (tp, tpost) in pair:
                scale[k] = pair[(tp, tpost)]
            elif tp in pre_only:
                scale[k] = pre_only[tp]
        if np.any(scale != 1.0):
            Wc.data[a:b] *= scale
            n_changed += int((scale != 1.0).sum())
    print(f"applied type-gains to {n_changed} edges")
    return Wc.tocsr()


def bake_folder(out_dir: Path, net: Connectome | None = None) -> None:
    out_dir = Path(out_dir)
    write_readme(out_dir)
    gains = out_dir / "gains_type.csv"
    if gains.exists() and net is not None:
        # also write a compact npz keyed by pre body, post body for Tier B later
        df = pd.read_csv(gains)
        np.savez_compressed(
            out_dir / "gains_type.npz",
            type_pre=df["type_pre"].astype(str).to_numpy(),
            type_post=df["type_post"].astype(str).to_numpy(),
            exp_theta=df["exp_theta"].to_numpy(dtype=np.float32),
            cut=df["cut"].astype(str).to_numpy() if "cut" in df.columns else np.array([]),
        )
    meta = {
        "constants": {
            "TICK_HZ": TICK_HZ,
            "INNER_STEPS": INNER_STEPS,
            "INNER_DT": INNER_DT,
            "WINDOW_INNER_STEPS": WINDOW_INNER_STEPS,
            "TAU_M": TAU_M,
            "TAU_S": TAU_S,
            "VREST": VREST,
            "VTH": VTH,
            "VRESET": VRESET,
            "TREF": TREF,
            "WSYN": WSYN,
            "GAIN": GAIN,
            "MIN_WEIGHT": MIN_WEIGHT,
            "FOV_PX": FOV_PX,
            "VISION_HZ_MAX": VISION_HZ_MAX,
        }
    }
    existing = {}
    mpath = out_dir / "metrics.json"
    if mpath.exists():
        existing = json.loads(mpath.read_text())
    existing.update(meta)
    mpath.write_text(json.dumps(existing, indent=2, default=float))
