"""Labeled frames: index.csv + images.

Expected columns (extras ignored):
  path, LF, RF, LM, RM, LH, RH [, clip_id, t, already_composited, already_fov]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import BUTTONS, FOV_PX
from .vision import frame_to_fov, hex_rates_from_fov, load_overlay


@dataclass
class Example:
    path: Path
    y: np.ndarray  # (6,) float in [0, 1]
    clip_id: str
    t: float
    already_composited: bool
    already_fov: bool
    split: str = "train"


def load_index(labels_dir: Path) -> pd.DataFrame:
    labels_dir = Path(labels_dir)
    csv = labels_dir / "index.csv"
    if not csv.exists():
        # accept a single csv passed as the "labels" path
        if labels_dir.suffix.lower() == ".csv":
            csv = labels_dir
            labels_dir = labels_dir.parent
        else:
            raise FileNotFoundError(
                f"No index.csv under {labels_dir}. Expected columns: "
                f"path, {', '.join(BUTTONS)}"
            )
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    # tolerate lowercase / alt names
    rename = {}
    for c in df.columns:
        key = c.strip()
        if key.lower() == "path":
            rename[c] = "path"
        elif key.upper() in BUTTONS:
            rename[c] = key.upper()
        elif key.lower() in {"clip_id", "clip", "seq"}:
            rename[c] = "clip_id"
        elif key.lower() in {"t", "time", "frame"}:
            rename[c] = "t"
    df = df.rename(columns=rename)
    missing = [b for b in BUTTONS if b not in df.columns]
    if missing:
        raise ValueError(f"index.csv missing button columns: {missing}")
    if "path" not in df.columns:
        raise ValueError("index.csv needs a 'path' column")
    if "clip_id" not in df.columns:
        df["clip_id"] = [f"frame_{i}" for i in range(len(df))]
    if "t" not in df.columns:
        df["t"] = np.arange(len(df), dtype=np.float32)
    df["path"] = df["path"].astype(str)
    for b in BUTTONS:
        df[b] = pd.to_numeric(df[b], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return df, labels_dir


def resolve_path(p: str, labels_dir: Path) -> Path:
    raw = Path(p)
    if raw.is_file():
        return raw
    for cand in (
        labels_dir / p,
        labels_dir / "frames" / Path(p).name,
        labels_dir.parent / p,
    ):
        if cand.is_file():
            return cand
    return labels_dir / p  # let the loader raise later


def examples_from_index(df: pd.DataFrame, labels_dir: Path) -> list[Example]:
    out = []
    for _, row in df.iterrows():
        out.append(
            Example(
                path=resolve_path(str(row["path"]), labels_dir),
                y=np.array([float(row[b]) for b in BUTTONS], dtype=np.float32),
                clip_id=str(row["clip_id"]),
                t=float(row["t"]),
                already_composited=bool(row.get("already_composited", True)),
                already_fov=str(row["path"]).lower().endswith(".npy")
                and "fov" in str(row["path"]).lower(),
            )
        )
    return out


def split_by_clip(
    examples: list[Example], val_frac: float = 0.2, seed: int = 0
) -> list[Example]:
    clips = sorted({e.clip_id for e in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(clips)
    n_val = max(1, int(round(len(clips) * val_frac))) if len(clips) > 1 else 0
    val = set(clips[:n_val])
    for e in examples:
        e.split = "val" if e.clip_id in val else "train"
    return examples


def rates_for_example(
    ex: Example,
    px: np.ndarray,
    py: np.ndarray,
    overlay_path: Path | None,
    composite_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (fov, hex_hz)."""
    overlay = None
    if composite_raw and not ex.already_composited:
        overlay = load_overlay(overlay_path)
    fov = frame_to_fov(
        ex.path,
        overlay=overlay,
        canvas=FOV_PX,
        already_fov=ex.already_fov,
    )
    return fov, hex_rates_from_fov(fov, px, py)
