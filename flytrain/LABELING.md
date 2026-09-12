# How to label 4K frames

The trainer never sees 4K pixels as a grid of features. It letterboxes the
frame to 256×256 the same way `fly.py` does, samples luma at each hex-driven
optic-lobe cell, and runs the connectome. Your job is to say which of the six
leg pools should be “on” for that frame.

## Files

```
C:\Dev\flywow\labels\
  index.csv
  frames\
    clipA_0000.png
    clipA_0001.png
    ...
```

`index.csv` columns:

| column   | meaning |
|----------|---------|
| path     | relative or absolute path to the PNG/JPG/NPY |
| LF RF LM RM LH RH | 0/1 or a soft score in `[0,1]` |
| clip_id  | same string for consecutive frames of one take |
| t        | seconds from the start of the clip (0.0, 0.1, 0.2, …) |

Generate a stub:

```
python -m flytrain template --frames C:\Dev\flywow\labels\frames --labels C:\Dev\flywow\labels
```


## Labeling applet

```
cd C:\Dev\flywow\flytrain
python -m flytrain label --frames C:\Dev\flywow\labels\frames --labels C:\Dev\flywow\labels --data C:\Dev\flywow\data --guidance C:\Dev\flywow\guidance.png --icons C:\Dev\flywow\labels\icons
```

`--data` is optional but recommended: it loads real `assignedOlHex*` so the
bottom-right panel is the actual optic-lobe sample lattice, not a placeholder.

The window is a 2×2:

* **Raw** — the screenshot as saved
* **Guidance + letterbox** — same overlay + 256² FOV `fly.py` builds
* **Luma** — `(R+G+B)/3`, the intensity every driven cell reads
* **Hex OL drive** — that luma painted only at hex-cell coordinates (left),
  and the sample sites on a dim FOV (right)

Footer toggles (also keys). Several legs can be on at once.

| key | pool | button |
|-----|------|--------|
| 1 | LF | Left front |
| 2 | LM | Left mid |
| 3 | LH | Left hind |
| 4 | RM | Right mid |
| 5 | RH | Right hind |
| E | RF | Right front |
| Enter / ✓ | — | write `index.csv` and load the next frame |
| Backspace | — | previous frame (reloads saved bits if any) |
| Esc | — | quit |

Icons: if `--icons` (or `labels/icons/`) contains `1.png`…`5.png` and `E.png`
(or `.gif` / `.ppm`), those are used on the footer buttons. Otherwise the
applet draws a letter tile. Checkmark always writes, then resets the toggles.

Already-labeled files are skipped on startup; ✓ overwrites a row if you go
back with Backspace and save again. `already_composited` is written as `0`
because these are raw screenshots — run cache with `--composite-raw`.

## Overlay

Live `fly.py` does:

1. grab desktop RGB
2. optional `guidance.png` RGBA, NEAREST-resized, `alpha_composite`
3. letterbox to 256² with black bars, bilinear

**Preferred:** save frames *after* that overlay, so the trainer and the live
loop see the same pixels. Set nothing extra.

**If you only have raw 4K captures:** keep a copy of `guidance.png` next to
`fly.py` and run cache with `--composite-raw`.

## What to put on screen

- Dark field, **one small bright patch** (or one UI widget) per example.
- Do not train on a full-white frame. The untrained net already sloshes
  current through mid/hind pools on white; that is a regression test, not a
  target.
- One-hot is fine for the first dataset (exactly one button = 1). Soft
  multi-label is allowed later.
- Short clips (0.3–1.0 s) of the same patch beat shuffled stills, because
  the LIF carries `V,g,refr` across 100 ms windows inside a clip.

## Pool definitions (must match fly.py)

| name | subclass | somaSide | exitNerve |
|------|----------|----------|-----------|
| LF   | fl       | L        | any       |
| RF   | fl       | R        | any       |
| LM   | ml       | L        | any       |
| RM   | ml       | R        | any       |
| LH   | hl       | L        | MetaLN only |
| RH   | hl       | R        | MetaLN only |

Hind-leg `hl` cells that leave through `AbN1` are dropped on purpose.

## Split

The trainer holds out **clips**, not random frames. If you put every frame of
a take in the same `clip_id`, adjacent 100 ms windows cannot leak into
validation.
