#!/usr/bin/env python3
"""Live dxcam preview in tkinter.

Shows the composited grab (framebuffer + optional guidance.png).
Optional second pane: hex fly-eye (R->UV, G->green, B->blue).

Requires: dxcam, pillow, numpy, tkinter
"""

from __future__ import annotations

from pathlib import Path
import tkinter as tk

import dxcam
import numpy as np
from PIL import Image, ImageTk, ImageDraw

GUIDANCE = Path("guidance.png")
SHOW_FLY_EYE = True
HEX_RINGS = 14
MAX_VIEW_W = 960
TARGET_MS = 30  # ~33 Hz UI tick; dxcam is newer-frame when it can


def hex_disk(rings: int) -> np.ndarray:
    cells = []
    for q in range(-rings, rings + 1):
        rmin = max(-rings, -q - rings)
        rmax = min(rings, -q + rings)
        for r in range(rmin, rmax + 1):
            cells.append((q, r))
    return np.array(cells, dtype=np.int32)


def axial_to_pixel(q, r, size):
    x = size * (1.5 * q)
    y = size * (np.sqrt(3) / 2.0 * q + np.sqrt(3) * r)
    return x, y


def build_hex_lookup(w: int, h: int, rings: int):
    cells = hex_disk(rings)
    q, r = cells[:, 0].astype(float), cells[:, 1].astype(float)
    qx, qy = axial_to_pixel(q, r, 1.0)
    span = max(float(qx.max() - qx.min()), float(qy.max() - qy.min()), 1e-6)
    size = 0.92 * min(w, h) / span
    px, py = axial_to_pixel(q, r, size)
    px += w / 2.0
    py += h / 2.0
    return cells, px, py, size


def paint_hex(rgb: np.ndarray, px, py, size) -> Image.Image:
    h, w, _ = rgb.shape
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    rad = max(int(size * 0.45), 1)
    for i, (cx, cy) in enumerate(zip(px, py)):
        x0 = int(np.clip(cx - rad, 0, w - 1))
        x1 = int(np.clip(cx + rad + 1, 1, w))
        y0 = int(np.clip(cy - rad, 0, h - 1))
        y1 = int(np.clip(cy + rad + 1, 1, h))
        r, g, b = rgb[y0:y1, x0:x1].mean(axis=(0, 1))
        s = size * 0.95
        pts = [
            (cx + s * np.cos(np.radians(60 * k)), cy + s * np.sin(np.radians(60 * k)))
            for k in range(6)
        ]
        draw.polygon(pts, fill=(int(r), int(g), int(b)))
    return img


def fit(im: Image.Image, max_w: int) -> Image.Image:
    if im.width <= max_w:
        return im
    h = int(im.height * max_w / im.width)
    return im.resize((max_w, h), Image.Resampling.BILINEAR)


class Viewer:
    def __init__(self) -> None:
        self.cam = dxcam.create(output_color="RGB")
        if self.cam is None:
            raise RuntimeError("dxcam.create failed")

        self.overlay = None
        if GUIDANCE.exists():
            self.overlay = Image.open(GUIDANCE).convert("RGBA")

        self.root = tk.Tk()
        self.root.title("fly cam")
        self.root.configure(bg="#111")

        self.left = tk.Label(self.root, bg="#111")
        self.left.grid(row=0, column=0, padx=4, pady=4)
        self.right = tk.Label(self.root, bg="#111")
        if SHOW_FLY_EYE:
            self.right.grid(row=0, column=1, padx=4, pady=4)

        self.status = tk.Label(self.root, text="", fg="#ccc", bg="#111", font=("Consolas", 10))
        self.status.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=4)

        self._photo_l = None
        self._photo_r = None
        self._hex = None
        self.n = 0
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(0, self.tick)

    def grab(self) -> Image.Image | None:
        frame = self.cam.grab(new_frame_only=False)
        if frame is None:
            return None
        base = Image.fromarray(frame).convert("RGBA")
        if self.overlay is not None:
            ov = self.overlay
            if ov.size != base.size:
                ov = ov.resize(base.size, Image.Resampling.NEAREST)
                self.overlay = ov
            base = Image.alpha_composite(base, ov)
        return base.convert("RGB")

    def tick(self) -> None:
        im = self.grab()
        if im is not None:
            view = fit(im, MAX_VIEW_W)
            self._photo_l = ImageTk.PhotoImage(view)
            self.left.configure(image=self._photo_l)

            if SHOW_FLY_EYE:
                small = fit(im, 480)
                arr = np.asarray(small)
                if self._hex is None or self._hex[1] != small.size:
                    cells, px, py, size = build_hex_lookup(small.width, small.height, HEX_RINGS)
                    self._hex = (cells, px, py, size, small.size)
                _, px, py, size, _ = self._hex
                eye = paint_hex(arr, px, py, size)
                self._photo_r = ImageTk.PhotoImage(eye)
                self.right.configure(image=self._photo_r)

            self.n += 1
            extra = f"  hex={len(self._hex[0])}" if SHOW_FLY_EYE and self._hex else ""
            self.status.configure(text=f"frames={self.n}  {im.width}x{im.height}{extra}  Q quit")

        self.root.after(TARGET_MS, self.tick)

    def close(self) -> None:
        try:
            self.cam.stop()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.bind("q", lambda e: self.close())
        self.root.mainloop()


if __name__ == "__main__":
    Viewer().run()
