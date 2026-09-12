# flytrain — MaleCNS visual → 6-leg buttons

Train a **small set of gain parameters** on synapses that already exist in the
published male *Drosophila* CNS connectome (Berg et al., Cell 2026 / MaleCNS v1.0)
so that a labeled screen patch maps to six VNC motor pools.

This does **not** replace the fly with a pixel MLP. It does **not** train Kenyon
cells. Sparsity, synapse counts, and neurotransmitter signs stay frozen. The
only thing that moves is a per-pre-type (or per-type-pair) multiplier on three
cuts of the graph:

```
ol_intrinsic  →  visual_projection  →  descending_neuron  →  vnc_intrinsic | vnc_motor
```

Read the [disclaimer](DISCLAIMER.md) before you demo this.

## What you need on the Windows box (32 GB is plenty)

```
C:\Dev\flywow\
  fly.py
  guidance.png                  (optional overlay)
  data\
    body-annotations.feather
    body-neurotransmitters.feather
    connectome-weights.feather
  labels\
    index.csv
    frames\000123.png           (4K raw or already-composited)
  flytrain\                     (this folder)
```

RAM: the filtered graph (`weight >= 5`) is on the order of 6 million edges.
The CSR + LIF state is a few hundred MB. Caching 1k frames of hex/VPN/DN rates
is well under 1 GB. Do **not** BPTT 176k–211k LIFs inside Adam; Stage A caches
once, Stage B trains on the cached vectors.

Python 3.10+ :

```
pip install -r requirements.txt
```

`torch` is CPU-only fine. GPU is unnecessary for Tier A.

## Label file

`labels/index.csv`:

```
path,LF,RF,LM,RM,LH,RH,clip_id,t
frames/000123.png,1,0,0,0,0,0,clipA,0.00
frames/000124.png,1,0,0,0,0,0,clipA,0.10
frames/000125.png,0,1,0,0,0,0,clipB,0.00
```

- `path` — 4K PNG/JPG **or** a 256×256 `fov.npy`. Prefer frames that already
  include the same overlay the live grab uses. If they are raw captures, pass
  `--composite-raw` so `guidance.png` is composited the same way as `fly.py`.
- Six columns in order `LF,RF,LM,RM,LH,RH`. Bits (0/1) or soft floats in `[0,1]`.
- `clip_id` — LIF state carries across rows with the same clip, resets between
  clips. Split is **by clip**, not by random frames.
- Loose screenshots with no clip: each row gets its own clip (reset every
  100 ms). That mismatches live carry-over; prefer short clips.

Stub a CSV from a folder of screenshots:

```
python -m flytrain template --frames C:\Dev\flywow\labels\frames --labels C:\Dev\flywow\labels

# or click through them:
python -m flytrain label --frames C:\Dev\flywow\labels\frames --labels C:\Dev\flywow\labels --data C:\Dev\flywow\data --icons C:\Dev\flywow\labels\icons
```

Then fill in the six bits. Suggested on-screen convention after letterbox
(not a biological claim — just so you and the trainer agree):

| button | FOV region (256²) |
|--------|-------------------|
| LF     | upper-left        |
| RF     | upper-right       |
| LM     | mid-left          |
| RM     | mid-right         |
| LH     | lower-left        |
| RH     | lower-right       |

Labels should be **small bright patches on a dark field**, not full-white
slogans. The live net is already too hot on a white screen.

## Commands

From the folder that contains the `flytrain` package (or with `PYTHONPATH` set):

```
cd C:\Dev\flywow\flytrain

# 1. Did the three hops survive weight>=5?
python -m flytrain inspect --data C:\Dev\flywow\data

# 2. Cache 100 ms of the same LIF as fly.py
python -m flytrain cache --data C:\Dev\flywow\data --labels C:\Dev\flywow\labels --out train_out

# 3. Fit Tier A type-gains + a diagnostic DN→6 head
python -m flytrain train --data C:\Dev\flywow\data --out train_out --steps 3000

#    if hex→VPN→DN is empty / DNs stay 0 on patches:
python -m flytrain train --data C:\Dev\flywow\data --out train_out --tier-c --pair

# 4. Idle / white regression with baked gains
python -m flytrain check-dark --data C:\Dev\flywow\data --out train_out
```

Outputs:

```
train_out/gains_type.csv     type_pre, type_post, exp_theta, cut
train_out/metrics.json       val BCE, per-button AP, dark/white probes
train_out/README.txt         LIF constants + how to load gains in fly.py
train_out/cache/cache.npz    Stage A vectors
```

## Applying gains in `fly.py`

After `W = build_W(nodes, edges).tocsc()`:

```python
from pathlib import Path
import sys
sys.path.append(r"C:\Dev\flywow\flytrain")
from flytrain.bake import apply_type_gains

gains = Path(r"C:\Dev\flywow\flytrain\train_out\gains_type.csv")
if gains.exists():
    W = apply_type_gains(W, nodes, gains).tocsc()
```

Keep `GAIN = 0.2`. Do not raise it as a substitute for θ.

## Pipeline (matches the spec)

1. Letterbox the frame to 256², luma `(R+G+B)/3`.
2. Drive every `ol_sensory | ol_intrinsic | visual_projection | visual_centrifugal`
   cell that has `assignedOlHex1` (current dump: ~23,720 `ol_intrinsic`).
3. Poisson at `luma * 80 Hz` for 20 inner steps (100 ms).
4. Read mean spike rate of the six `vnc_motor` pools
   (`fl/ml` any nerve, `hl` + `MetaLN` only).
5. Update θ only on existing edges in the three cuts.
   `W_ij = W0_ij * GAIN * exp(θ[type_pre])`.

Stage B does **not** differentiate through 211k LIFs. It uses a 1-compartment
softplus hop through the three compact cut matrices plus a diagnostic linear
head from cached DN rates. If that diagnostic AP is chance **and** cached DNs
are ~0 on labeled patches, the hop is dead — print the missing cut and stop.
Do not train a pixel classifier to “fix” it.

## Smoke test without feathers

```
python -m flytrain synth-labels --labels _demo_labels -n 4
python -m flytrain inspect --synth
python -m flytrain cache  --synth --labels _demo_labels --out _demo_out
python -m flytrain train  --synth --out _demo_out --steps 200
python -m flytrain check-dark --synth --out _demo_out
```

Or `python scripts/smoke.py`.

## Milestone checklist

- [ ] `inspect` prints nonzero edges on hex-ol → VPN → DN → vnc_motor
- [ ] dark frame: all six pools ~0, KC ~0
- [ ] white frame: not used as a training target
- [ ] cached DNs move on the labeled patches
- [ ] val AP per button above chance on held-out *clips*
- [ ] baking θ does not make >100 MNs spike on dark
