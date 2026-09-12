"""CLI: inspect / synth-labels / cache / train / bake / check-dark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bake import bake_folder, write_readme
from .cache import cache_examples, load_cache, simulate_fov
from .connectome import load_connectome, summarize
from .constants import BUTTONS, GAIN
from .cuts import extract_standard_cuts, inventory, missing_path_report
from .dataset import examples_from_index, load_index, split_by_clip
from .synth import make_synthetic_connectome, write_synthetic_labels
from .train import train_tier_a
from .vision import make_dark_fov, make_white_fov


DEFAULT_DATA = Path(r"C:\Dev\flywow\data")
DEFAULT_LABELS = Path(r"C:\Dev\flywow\labels")
DEFAULT_GUIDANCE = Path(r"C:\Dev\flywow\guidance.png")
DEFAULT_OUT = Path("train_out")


def _net(args):
    if getattr(args, "synth", False):
        print("using SYNTHETIC connectome (pipeline smoke test, not MaleCNS)")
        return make_synthetic_connectome(seed=getattr(args, "seed", 0))
    data = Path(args.data)
    if not (data / "body-annotations.feather").exists():
        raise SystemExit(
            f"No feathers in {data}. Pass --data C:\\Dev\\flywow\\data "
            f"or --synth for a smoke-test graph."
        )
    print(f"loading connectome from {data}")
    return load_connectome(data)


def cmd_inspect(args) -> None:
    net = _net(args)
    info = summarize(net)
    print(json.dumps(info, indent=2))
    rows = inventory(net)
    print("\ncuts at weight>=5")
    print(pd.DataFrame(rows).to_string(index=False))
    for a in missing_path_report(rows):
        print("ALERT:", a)
    cuts = extract_standard_cuts(net)
    print("\ntrainable compact cuts")
    for name, c in cuts.items():
        print(
            f"  {name:8s} edges={c.W_compact.nnz:7d}  "
            f"pre_types={len(set(c.pre_type.astype(str))):4d}  "
            f"type_pairs={len(set(zip(c.edge_pre_type.astype(str), c.edge_post_type.astype(str))))}"
        )


def cmd_synth_labels(args) -> None:
    path = write_synthetic_labels(Path(args.labels), n_per_button=args.n, seed=args.seed)
    print(f"wrote {path}")


def cmd_cache(args) -> None:
    net = _net(args)
    labels_dir = Path(args.labels)
    df, root = load_index(labels_dir)
    examples = split_by_clip(examples_from_index(df, root), val_frac=args.val_frac, seed=args.seed)
    print(f"{len(examples)} examples  "
          f"train={sum(e.split=='train' for e in examples)}  "
          f"val={sum(e.split=='val' for e in examples)}")
    out = Path(args.out) / "cache"
    cache_examples(
        net,
        examples,
        out,
        overlay_path=Path(args.guidance) if args.guidance else None,
        composite_raw=args.composite_raw,
        seed=args.seed,
    )
    print(f"cache → {out}")


def cmd_train(args) -> None:
    net = _net(args)
    cache, meta = load_cache(Path(args.out) / "cache")
    print(f"cache n={len(cache['y'])}  hex={cache['hex_drive_hz'].shape}  dn={cache['dn_hz'].shape}")
    # If Stage A DNs are ~0 on labeled (non-dark) frames, stop and print the cut.
    y = cache["y"]
    active = y.sum(axis=1) > 0
    dn_mean = float(cache["dn_hz"][active].mean()) if active.any() else float(cache["dn_hz"].mean())
    print(f"mean DN Hz on labeled-active frames: {dn_mean:.6f}")
    if dn_mean < 1e-4:
        print(
            "Stage A DNs are ~0 on labeled patches. Check the three cuts with "
            "`inspect`. If a cut is empty, enable --tier-c; do not raise GAIN."
        )
    res = train_tier_a(
        net,
        cache,
        meta,
        Path(args.out),
        steps=args.steps,
        lr=args.lr,
        l2=args.l2,
        pair=args.pair,
        use_tier_c=args.tier_c,
        device=args.device,
    )
    bake_folder(Path(args.out), net)
    print("gains →", Path(args.out) / "gains_type.csv")
    print("top |log gain| types:")
    g = res.gains.copy()
    g["abs_theta"] = g["theta"].abs()
    print(g.sort_values("abs_theta", ascending=False).head(16).to_string(index=False))


def cmd_bake(args) -> None:
    bake_folder(Path(args.out))
    print("wrote README.txt + gains sidecar in", args.out)


def cmd_check_dark(args) -> None:
    net = _net(args)
    from .bake import apply_type_gains

    W = net.W
    gains = Path(args.out) / "gains_type.csv"
    if gains.exists():
        W = apply_type_gains(W, net.nodes, gains)
    dark = simulate_fov(net, make_dark_fov(), seed=1, W=W)
    white = simulate_fov(net, make_white_fov(), seed=2, W=W)
    print("dark pools Hz", np.round(dark["pool_hz"], 4).tolist())
    print("dark DN mean", float(dark["dn_hz"].mean()), "KC mean",
          float(dark["kc_hz"].mean()) if len(dark["kc_hz"]) else 0.0)
    print("white pools Hz", np.round(white["pool_hz"], 4).tolist())
    print("white DN mean", float(white["dn_hz"].mean()), "net spikes", dark["n_spikes"], white["n_spikes"])
    n_mn_spk = int((dark["mn_hz"] > 0).sum()) if len(dark["mn_hz"]) else 0
    if n_mn_spk > 100:
        print("REJECT: >100 MNs spike on a dark frame")
    if len(dark["kc_hz"]) and float(dark["kc_hz"].mean()) > 0.05:
        print("WARN: Kenyon cells not idle on dark")


def cmd_label(args) -> None:
    from .labeler import launch

    frames = Path(args.frames) if args.frames else Path(args.labels) / "frames"
    launch(
        frames_dir=frames,
        labels_dir=Path(args.labels),
        guidance=Path(args.guidance) if args.guidance else None,
        data_dir=Path(args.data) if args.data else None,
        icons_dir=Path(args.icons) if args.icons else None,
        synth=bool(args.synth),
    )


def cmd_template(args) -> None:
    """Write a blank index.csv next to a folder of frames."""
    frames = Path(args.frames)
    labels = Path(args.labels)
    labels.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.npy"):
        paths += sorted(frames.glob(ext))
    if not paths:
        raise SystemExit(f"no images in {frames}")
    rows = []
    for i, p in enumerate(paths):
        rec = {b: 0 for b in BUTTONS}
        rec.update({"path": str(p), "clip_id": f"loose_{i}", "t": 0.0})
        rows.append(rec)
    df = pd.DataFrame(rows)
    dest = labels / "index.csv"
    df.to_csv(dest, index=False)
    print(f"wrote {dest} with {len(df)} rows. Fill in the six button columns.")
    print("Convention: 1 = that leg should be active for this frame.")
    print("Prefer clip_id shared across consecutive 100 ms frames of one take.")


def _common_flags() -> argparse.ArgumentParser:
    """Shared flags live on every subcommand so they work *after* the verb.

    argparse only accepts parent-parser options before `label`/`cache`/… unless
    those options are also registered on the subparser. Windows users (and
    everyone else) type them after the verb.
    """
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument("--data", type=Path, default=DEFAULT_DATA)
    c.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    c.add_argument("--guidance", type=Path, default=DEFAULT_GUIDANCE)
    c.add_argument("--out", type=Path, default=DEFAULT_OUT)
    c.add_argument("--synth", action="store_true", help="use a tiny fake connectome")
    c.add_argument("--seed", type=int, default=0)
    return c


def build_parser() -> argparse.ArgumentParser:
    common = _common_flags()
    p = argparse.ArgumentParser(
        prog="flytrain",
        description="MaleCNS visual→6-button trainer",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inspect", help="load feathers, print cut counts", parents=[common])
    sl = sub.add_parser("synth-labels", help="write demo patches + index.csv", parents=[common])
    sl.add_argument("-n", type=int, default=4)

    c = sub.add_parser("cache", help="Stage A: 100 ms LIF cache", parents=[common])
    c.add_argument("--val-frac", type=float, default=0.2)
    c.add_argument(
        "--composite-raw",
        action="store_true",
        help="alpha-composite guidance.png onto frames that are not already overlaid",
    )

    t = sub.add_parser("train", help="Stage B: Tier A type-gains", parents=[common])
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--lr", type=float, default=1e-2)
    t.add_argument("--l2", type=float, default=1e-4)
    t.add_argument("--pair", action="store_true", help="θ[type_pre, type_post] instead of type_pre")
    t.add_argument("--tier-c", action="store_true", help="also train ol_intrinsic → DN")
    t.add_argument("--device", default="auto", help="auto | cpu | cuda")

    sub.add_parser("bake", help="write README + sidecar", parents=[common])
    sub.add_parser("check-dark", help="Stage C idle/white regression", parents=[common])

    tm = sub.add_parser("template", help="index.csv stub from a folder of 4k frames", parents=[common])
    tm.add_argument("--frames", type=Path, required=True)

    lb = sub.add_parser("label", help="tkinter applet to label raw screenshots", parents=[common])
    lb.add_argument("--frames", type=Path, default=None, help="folder of raw screenshots")
    lb.add_argument(
        "--icons",
        type=Path,
        default=None,
        help="folder with 1.png .. 5.png and E.png (png/gif/ppm)",
    )
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    {
        "inspect": cmd_inspect,
        "synth-labels": cmd_synth_labels,
        "cache": cmd_cache,
        "train": cmd_train,
        "bake": cmd_bake,
        "check-dark": cmd_check_dark,
        "template": cmd_template,
        "label": cmd_label,
    }[args.cmd](args)
