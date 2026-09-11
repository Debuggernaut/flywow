#!/usr/bin/env python3
"""Hex-sample a framebuffer the way a fly eye roughly can.

Channels per ommatidium
  UV    <- source red   (your mapping)
  blue  <- source blue  (pale R8 / short-λ)
  green <- source green (yellow R8)

R1-6 in the real fly are broadband; here they can share (UV+G+B)/3
without forcing the preview to grayscale.

Writes:
  fly_eye_preview.png  hex mosaic reconstructed as RGB (R=UV, G=green, B=blue)
  fly_eye_hex.npz      q, r, uv, blue, green, luma  in 0..1
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

FRAME = Path("frame.png")
PREVIEW = Path("fly_eye_preview.png")
DUMP = Path("fly_eye_hex.npz")

# ~750-800 ommatidia per eye. 16 rings ~817 hexes.
HEX_RINGS = 16
BLUR_PX = 8  # cheap stand-in for facet acceptance angle


def hex_disk(rings: int) -> np.ndarray:
    """Axial (q, r) coordinates for a filled hex of given radius."""
    cells = []
    for q in range(-rings, rings + 1):
        rmin = max(-rings, -q - rings)
        rmax = min(rings, -q + rings)
        for r in range(rmin, rmax + 1):
            cells.append((q, r))
    return np.array(cells, dtype=np.int32)


def axial_to_pixel(q: np.ndarray, r: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray]:
    x = size * (3.0 / 2.0 * q)
    y = size * (np.sqrt(3) / 2.0 * q + np.sqrt(3) * r)
    return x, y


def hex_corners(cx: float, cy: float, size: float) -> list[tuple[float, float]]:
    pts = []
    for k in range(6):
        ang = np.radians(60 * k)
        pts.append((cx + size * np.cos(ang), cy + size * np.sin(ang)))
    return pts


def box_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray, rad: float) -> np.ndarray:
    """Mean RGB in a small square around each hex centre. img is H x W x 3 float 0..1."""
    h, w, _ = img.shape
    out = np.zeros((len(x), 3), dtype=np.float32)
    rad = max(int(rad), 1)
    xs = np.clip(x.astype(int), 0, w - 1)
    ys = np.clip(y.astype(int), 0, h - 1)
    for i, (cx, cy) in enumerate(zip(xs, ys)):
        x0, x1 = max(cx - rad, 0), min(cx + rad + 1, w)
        y0, y1 = max(cy - rad, 0), min(cy + rad + 1, h)
        out[i] = img[y0:y1, x0:x1].mean(axis=(0, 1))
    return out


def main() -> None:
    src = Image.open(FRAME).convert("RGB")
    if BLUR_PX:
        src = src.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(BLUR_PX))
    arr = np.asarray(src, dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    cells = hex_disk(HEX_RINGS)
    q, r = cells[:, 0], cells[:, 1]
    n = len(cells)

    # Fit the hex lattice inside the frame with a small margin.
    qx, qy = axial_to_pixel(q.astype(float), r.astype(float), 1.0)
    span = max(qx.max() - qx.min(), qy.max() - qy.min())
    size = 0.92 * min(w, h) / max(span, 1e-6)
    px, py = axial_to_pixel(q.astype(float), r.astype(float), size)
    px += w / 2.0
    py += h / 2.0

    rgb = box_sample(arr, px, py, size * 0.45)
    uv = rgb[:, 0]       # red -> UV
    green = rgb[:, 1]
    blue = rgb[:, 2]
    luma = (uv + green + blue) / 3.0  # stand-in for R1-6 broadband

    # Preview: put UV back on red so you can see what the fly-eye kept.
    preview = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(preview)
    for i in range(n):
        r8, g8, b8 = int(uv[i] * 255), int(green[i] * 255), int(blue[i] * 255)
        draw.polygon(hex_corners(float(px[i]), float(py[i]), size * 0.95), fill=(r8, g8, b8))

    preview.save(PREVIEW)
    np.savez(
        DUMP,
        q=q,
        r=r,
        x=px,
        y=py,
        uv=uv,
        blue=blue,
        green=green,
        luma=luma,
        hex_size=np.array(size),
    )
    print(f"{n} ommatidia  preview={PREVIEW}  dump={DUMP}")
    print(f"UV    mean={uv.mean():.3f}  max={uv.max():.3f}")
    print(f"green mean={green.mean():.3f}  max={green.max():.3f}")
    print(f"blue  mean={blue.mean():.3f}  max={blue.max():.3f}")


if __name__ == "__main__":
    main()
