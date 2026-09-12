"""Vectorized LIF matching fly.py event order.

Order per inner step (copied from fly.py):
  pending (previous firings) + driven Poisson sources
  → dump W @ spikes into g
  → leak g
  → Euler V
  → threshold, reset, set pending
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .constants import (
    GAIN,
    INNER_DT,
    TAU_M,
    TAU_S,
    TREF,
    VRESET,
    VREST,
    VTH,
    WINDOW_INNER_STEPS,
    WSYN,
)


@dataclass
class LifState:
    v: np.ndarray
    g: np.ndarray
    refr: np.ndarray
    pending: np.ndarray

    @classmethod
    def rest(cls, n: int) -> "LifState":
        return cls(
            v=np.full(n, VREST, dtype=np.float32),
            g=np.zeros(n, dtype=np.float32),
            refr=np.zeros(n, dtype=np.float32),
            pending=np.empty(0, dtype=np.int32),
        )


def run_window(
    W: sparse.csr_matrix,
    drive_idx: np.ndarray,
    drive_hz: np.ndarray,
    n_steps: int = WINDOW_INNER_STEPS,
    state: LifState | None = None,
    rng: np.random.Generator | None = None,
    scale: float | None = None,
    collect_idx: dict[str, np.ndarray] | None = None,
) -> tuple[LifState, dict]:
    """Run `n_steps` inner steps. Returns updated state and spike counts.

    `collect_idx` maps a name → neuron indices whose spikes are counted.
    Always also returns `spikes` (int32 length n) for the whole net.
    """
    n = W.shape[0]
    if state is None:
        state = LifState.rest(n)
    if rng is None:
        rng = np.random.default_rng(0)
    if scale is None:
        scale = float(WSYN * GAIN)

    v = state.v
    g = state.g
    refr = state.refr
    pending = state.pending
    leak = np.float32(np.exp(-INNER_DT / TAU_S))
    dt = np.float32(INNER_DT)
    tau_m = np.float32(TAU_M)
    vrest = np.float32(VREST)
    vth = np.float32(VTH)
    vreset = np.float32(VRESET)
    tref = np.float32(TREF)

    spikes = np.zeros(n, dtype=np.int32)
    g_abs_sum = np.zeros(n, dtype=np.float64)
    p = np.clip(drive_hz.astype(np.float64) * float(INNER_DT), 0.0, 1.0)

    for _ in range(n_steps):
        s = np.zeros(n, dtype=np.float32)
        if pending.size:
            s[pending] = 1.0
        if len(drive_idx):
            fire = rng.random(len(drive_idx)) < p
            if fire.any():
                s[drive_idx[fire]] = 1.0
        if s.any():
            g += np.float32(scale) * W.dot(s)
        g *= leak

        active = refr <= 0
        v[active] += dt * ((vrest - v[active]) / tau_m) + g[active]
        v[~active] = vreset
        refr = np.maximum(0.0, refr - dt)

        fired = active & (v >= vth)
        if fired.any():
            v[fired] = vreset
            g[fired] = 0.0
            refr[fired] = tref
            pending = np.flatnonzero(fired).astype(np.int32)
            spikes += fired.astype(np.int32)
        else:
            pending = np.empty(0, dtype=np.int32)
        g_abs_sum += np.abs(g)

    state.v = v
    state.g = g
    state.refr = refr
    state.pending = pending

    dur = n_steps * float(INNER_DT)
    out = {
        "spikes": spikes,
        "hz": spikes.astype(np.float32) / np.float32(dur),
        "g_mean": (g_abs_sum / max(n_steps, 1)).astype(np.float32),
        "duration_s": dur,
    }
    if collect_idx:
        for name, idx in collect_idx.items():
            out[f"{name}_spikes"] = spikes[idx]
            out[f"{name}_hz"] = out["hz"][idx]
            out[f"{name}_g_mean"] = out["g_mean"][idx]
    return state, out
