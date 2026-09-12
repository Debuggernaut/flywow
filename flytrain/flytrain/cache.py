"""Stage A: run the live LIF once per labeled frame and cache rate vectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .connectome import Connectome
from .constants import BUTTONS, GAIN, WINDOW_INNER_STEPS, WSYN
from .dataset import Example
from .lif import LifState, run_window
from .vision import hex_rates_from_fov, luma, make_dark_fov, make_white_fov


def _collect(net: Connectome) -> dict[str, np.ndarray]:
    return {
        "hex": net.hex_i,
        "vpn": net.vpn_i,
        "dn": net.dn_i,
        "mn": net.mn_i,
        "kc": net.kc_i,
        **{f"pool_{k}": v for k, v in net.pool_i.items()},
    }


def simulate_fov(
    net: Connectome,
    fov: np.ndarray,
    n_steps: int = WINDOW_INNER_STEPS,
    seed: int = 0,
    carry: LifState | None = None,
    W=None,
) -> dict:
    hex_hz_drive = hex_rates_from_fov(fov, net.hex_px, net.hex_py)
    rng = np.random.default_rng(seed)
    state, out = run_window(
        W if W is not None else net.W,
        net.hex_i,
        hex_hz_drive,
        n_steps=n_steps,
        state=carry,
        rng=rng,
        scale=float(WSYN * GAIN),
        collect_idx=_collect(net),
    )
    pool_hz = np.array(
        [
            float(out[f"pool_{k}_hz"].mean()) if len(net.pool_i[k]) else 0.0
            for k in BUTTONS
        ],
        dtype=np.float32,
    )
    other_mn = np.ones(net.n, dtype=bool)
    other_mn[net.mn_i] = True
    for k in BUTTONS:
        other_mn[net.pool_i[k]] = False
    other_mn &= np.isin(np.arange(net.n), net.mn_i)
    return {
        "state": state,
        "hex_drive_hz": hex_hz_drive,
        "hex_hz": out["hex_hz"],
        "vpn_hz": out["vpn_hz"],
        "vpn_g": out["vpn_g_mean"],
        "dn_hz": out["dn_hz"],
        "dn_g": out["dn_g_mean"],
        "mn_hz": out["mn_hz"],
        "kc_hz": out["kc_hz"] if len(net.kc_i) else np.zeros(0, dtype=np.float32),
        "pool_hz": pool_hz,
        "other_mn_hz": float(out["hz"][other_mn].mean()) if other_mn.any() else 0.0,
        "n_spikes": int(out["spikes"].sum()),
        "feat_mean": float(luma(fov).mean()),
    }


def cache_examples(
    net: Connectome,
    examples: list[Example],
    out_dir: Path,
    overlay_path: Path | None = None,
    composite_raw: bool = False,
    seed: int = 0,
) -> Path:
    from .dataset import rates_for_example

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = []
    carry: LifState | None = None
    prev_clip = None
    for i, ex in enumerate(examples):
        if ex.clip_id != prev_clip:
            carry = None
            prev_clip = ex.clip_id
        fov, hex_drive = rates_for_example(
            ex, net.hex_px, net.hex_py, overlay_path, composite_raw
        )
        sim = simulate_fov(net, fov, seed=seed + i, carry=carry)
        carry = sim["state"]
        rec = {
            "i": i,
            "path": str(ex.path),
            "clip_id": ex.clip_id,
            "t": ex.t,
            "split": ex.split,
            "y": ex.y,
            "hex_drive_hz": hex_drive,
            "vpn_hz": sim["vpn_hz"],
            "vpn_g": sim["vpn_g"],
            "dn_hz": sim["dn_hz"],
            "dn_g": sim["dn_g"],
            "pool_hz": sim["pool_hz"],
            "kc_mean_hz": float(sim["kc_hz"].mean()) if len(sim["kc_hz"]) else 0.0,
            "other_mn_hz": sim["other_mn_hz"],
            "n_spikes": sim["n_spikes"],
            "feat_mean": sim["feat_mean"],
        }
        recs.append(rec)
        print(
            f"[{i+1}/{len(examples)}] {ex.path.name}  "
            f"y={ex.y.astype(int).tolist()}  "
            f"LIF pools={np.round(sim['pool_hz'], 3).tolist()}  "
            f"DN_mean={float(sim['dn_hz'].mean()):.4f}Hz  "
            f"spikes={sim['n_spikes']}"
        )

    # regression probes
    dark = simulate_fov(net, make_dark_fov(), seed=1)
    white = simulate_fov(net, make_white_fov(), seed=2)
    probes = {
        "dark_pool_hz": dark["pool_hz"].tolist(),
        "dark_dn_mean": float(dark["dn_hz"].mean()),
        "dark_kc_mean": float(dark["kc_hz"].mean()) if len(dark["kc_hz"]) else 0.0,
        "dark_hex_drive_mean": float(dark["hex_drive_hz"].mean()),
        "dark_feat": dark["feat_mean"],
        "white_pool_hz": white["pool_hz"].tolist(),
        "white_dn_mean": float(white["dn_hz"].mean()),
        "white_n_spikes": white["n_spikes"],
    }
    print("dark probe", probes["dark_pool_hz"], "DN", probes["dark_dn_mean"])
    print("white probe pools", probes["white_pool_hz"], "DN", probes["white_dn_mean"])

    np.savez_compressed(
        out_dir / "cache.npz",
        y=np.stack([r["y"] for r in recs]),
        hex_drive_hz=np.stack([r["hex_drive_hz"] for r in recs]),
        vpn_hz=np.stack([r["vpn_hz"] for r in recs]),
        vpn_g=np.stack([r["vpn_g"] for r in recs]),
        dn_hz=np.stack([r["dn_hz"] for r in recs]),
        dn_g=np.stack([r["dn_g"] for r in recs]),
        pool_hz=np.stack([r["pool_hz"] for r in recs]),
        kc_mean_hz=np.array([r["kc_mean_hz"] for r in recs], dtype=np.float32),
        other_mn_hz=np.array([r["other_mn_hz"] for r in recs], dtype=np.float32),
        feat_mean=np.array([r["feat_mean"] for r in recs], dtype=np.float32),
        t=np.array([r["t"] for r in recs], dtype=np.float32),
    )
    meta = {
        "n": len(recs),
        "paths": [r["path"] for r in recs],
        "clip_id": [r["clip_id"] for r in recs],
        "split": [r["split"] for r in recs],
        "probes": probes,
        "n_hex": int(len(net.hex_i)),
        "n_vpn": int(len(net.vpn_i)),
        "n_dn": int(len(net.dn_i)),
    }
    (out_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir / "cache.npz"


def load_cache(cache_dir: Path) -> tuple[dict, dict]:
    cache_dir = Path(cache_dir)
    npz_path = cache_dir / "cache.npz" if cache_dir.is_dir() else cache_dir
    data = dict(np.load(npz_path, allow_pickle=False))
    meta = json.loads((npz_path.parent / "cache_meta.json").read_text())
    return data, meta
