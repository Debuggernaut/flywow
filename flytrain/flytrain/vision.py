"""Framebuffer → hex cell rates. Must match fly.py ScreenEye + hex_pixel_xy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .constants import FOV_PX, VISION_HZ_MAX


def hex_pixel_xy(
    h1: np.ndarray, h2: np.ndarray, canvas: int
) -> tuple[np.ndarray, np.ndarray]:
    """Map assignedOlHex1/2 into a square canvas. Missing hex → center."""
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
    px = np.where(np.isfinite(px), px, canvas / 2.0)
    py = np.where(np.isfinite(py), py, canvas / 2.0)
    return px, py


def fit_frame_to_fov(im: Image.Image, canvas: int = FOV_PX) -> Image.Image:
    scale = min(canvas / im.width, canvas / im.height)
    nw = max(1, int(im.width * scale))
    nh = max(1, int(im.height * scale))
    small = im.resize((nw, nh), Image.Resampling.BILINEAR)
    out = Image.new("RGB", (canvas, canvas), (0, 0, 0))
    out.paste(small, ((canvas - nw) // 2, (canvas - nh) // 2))
    return out


def maybe_composite(im: Image.Image, overlay: Image.Image | None) -> Image.Image:
    if overlay is None:
        return im.convert("RGB")
    base = im.convert("RGBA")
    ov = overlay.convert("RGBA")
    if ov.size != base.size:
        ov = ov.resize(base.size, Image.Resampling.NEAREST)
    return Image.alpha_composite(base, ov).convert("RGB")


def load_overlay(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return Image.open(path).convert("RGBA")


def frame_to_fov(
    path_or_arr,
    overlay: Image.Image | None = None,
    canvas: int = FOV_PX,
    already_fov: bool = False,
) -> np.ndarray:
    """Return float32 RGB in [0, 1] with shape (canvas, canvas, 3)."""
    if isinstance(path_or_arr, np.ndarray):
        arr = path_or_arr
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.float32 and arr.max() > 1.5:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        if already_fov and arr.shape[0] == canvas and arr.shape[1] == canvas:
            return np.clip(arr[..., :3], 0.0, 1.0)
        im = Image.fromarray(
            np.clip(arr[..., :3] * (255.0 if arr.max() <= 1.5 else 1.0), 0, 255).astype(
                np.uint8
            )
        )
    else:
        p = Path(path_or_arr)
        if p.suffix.lower() == ".npy":
            arr = np.load(p)
            return frame_to_fov(arr, overlay, canvas, already_fov)
        im = Image.open(p)

    im = maybe_composite(im, overlay)
    if already_fov and im.size == (canvas, canvas):
        fov = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        return fov
    fov = np.asarray(fit_frame_to_fov(im.convert("RGB"), canvas), dtype=np.float32)
    return fov / 255.0


def luma(fov: np.ndarray) -> np.ndarray:
    return fov[..., :3].mean(axis=2)


def hex_rates_from_fov(
    fov: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    vision_hz_max: float = VISION_HZ_MAX,
) -> np.ndarray:
    gray = luma(fov)
    h, w = gray.shape
    xi = np.clip(px.astype(int), 0, w - 1)
    yi = np.clip(py.astype(int), 0, h - 1)
    return (gray[yi, xi] * np.float32(vision_hz_max)).astype(np.float32)


def make_dark_fov(canvas: int = FOV_PX) -> np.ndarray:
    return np.zeros((canvas, canvas, 3), dtype=np.float32)


def make_white_fov(canvas: int = FOV_PX) -> np.ndarray:
    return np.ones((canvas, canvas, 3), dtype=np.float32)


def make_patch_fov(
    cx: float,
    cy: float,
    radius: int = 18,
    canvas: int = FOV_PX,
    value: float = 1.0,
) -> np.ndarray:
    """Small bright disk on a dark field (the intended label regime)."""
    fov = np.zeros((canvas, canvas, 3), dtype=np.float32)
    yy, xx = np.ogrid[:canvas, :canvas]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    fov[mask] = value
    return fov



def hex_retina_image(
    fov: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    radius: int = 1,
) -> np.ndarray:
    """Paint each hex sample onto a black canvas. This is the OL drive, spatially.

    Returns uint8 RGB, same H×W as the FOV.
    """
    gray = luma(fov)
    h, w = gray.shape
    canvas = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    xi = np.clip(np.rint(px).astype(int), 0, w - 1)
    yi = np.clip(np.rint(py).astype(int), 0, h - 1)
    if radius <= 0:
        np.add.at(canvas, (yi, xi), gray[yi, xi])
        np.add.at(count, (yi, xi), 1.0)
    else:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                yy = np.clip(yi + dy, 0, h - 1)
                xx = np.clip(xi + dx, 0, w - 1)
                np.add.at(canvas, (yy, xx), gray[yi, xi])
                np.add.at(count, (yy, xx), 1.0)
    out = np.divide(canvas, np.maximum(count, 1.0))
    rgb = np.stack([out, out, out], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def fov_with_sample_dots(
    fov: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    max_dots: int = 4000,
) -> np.ndarray:
    """Letterboxed RGB dimmed, with sample sites drawn in amber."""
    img = np.clip(fov[..., :3] * 0.45, 0.0, 1.0).copy()
    h, w = img.shape[:2]
    xi = np.clip(np.rint(px).astype(int), 0, w - 1)
    yi = np.clip(np.rint(py).astype(int), 0, h - 1)
    n = len(xi)
    if n > max_dots:
        rng = np.random.default_rng(0)
        take = rng.choice(n, size=max_dots, replace=False)
        xi, yi = xi[take], yi[take]
    img[yi, xi] = (1.0, 0.75, 0.15)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def to_uint8_rgb(fov: np.ndarray) -> np.ndarray:
    arr = fov[..., :3]
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def luma_uint8(fov: np.ndarray) -> np.ndarray:
    g = np.clip(luma(fov) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)
