"""Live tkinter monitor: optic-lobe FOV + six leg lamps.

Does not run its own mainloop. Call ``push`` from the LIF thread and it
will ``root.update()``. Target ~30 Hz from fly.py (every 3rd 100 Hz tick).

Hook (after ScreenEye and pools exist)::

    import sys
    from pathlib import Path
    sys.path.append(r"C:\\Dev\\flywow\\flytrain")
    from flytrain.monitor import FlyMonitor
    mon = FlyMonitor(icons_dir=Path(r"C:\\Dev\\flywow\\labels\\icons"))

    # once per outer tick, after grab + LIF:
    if mon is not None and tick % 3 == 0:
        dt = 3.0 / TICK_HZ
        hz = {
            k: pool_ui[k] / max(int(pools[k].sum()), 1) / dt
            for k in pools
        }
        mon.push(hz, fov=eye.gray, px=eye.vpn_px, py=eye.vpn_py)
        for k in pool_ui:
            pool_ui[k] = 0
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .constants import BUTTONS, FOV_PX
from .labeler import (
    FOOTER_ORDER,
    POOL_LABEL,
    POOL_TO_KEY,
    fallback_icon,
    find_icon,
    pil_to_tk,
)
from .vision import hex_retina_image, luma, to_uint8_rgb

ON_HZ = 0.20
VIEW_W = 546  # 420 * 1.30
VIEW_H = 420
ICON_PX = 45  # 56 * 0.80


def _need_tk():
    import tkinter as tk
    from tkinter import font as tkfont

    return tk, tkfont


def _luma2d(fov) -> np.ndarray | None:
    if fov is None:
        return None
    arr = np.asarray(fov)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    arr = arr.astype(np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _retina_pil(fov, px, py) -> Image.Image:
    gray = _luma2d(fov)
    if gray is None:
        gray = np.zeros((FOV_PX, FOV_PX), dtype=np.float32)
    rgb = np.stack([gray, gray, gray], axis=-1)
    if px is not None and py is not None and len(px) and len(py):
        try:
            img = hex_retina_image(rgb, np.asarray(px), np.asarray(py), radius=1)
            return Image.fromarray(img)
        except Exception:
            pass
    return Image.fromarray(to_uint8_rgb(rgb))


class FlyMonitor:
    def __init__(
        self,
        icons_dir: Path | None = None,
        on_hz: float = ON_HZ,
        view_w: int = VIEW_W,
        view_h: int = VIEW_H,
        icon_px: int = ICON_PX,
        title: str = "flywow monitor",
    ) -> None:
        tk, tkfont = _need_tk()
        self.tk = tk
        self.on_hz = float(on_hz)
        self.view_w = int(view_w)
        self.view_h = int(view_h)
        self.icon_px = int(icon_px)
        self.icons_dir = Path(icons_dir) if icons_dir else None
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="#12141a")
        self.root.resizable(False, False)
        self._photos: list = []
        self._icon_photos: dict = {}
        self._last_on: dict[str, bool] = {b: False for b in BUTTONS}

        hdr = tk.Label(
            self.root,
            text="optic lobe  (hex drive)",
            bg="#12141a",
            fg="#c8cdd8",
            font=tkfont.Font(size=11),
        )
        hdr.pack(anchor="w", padx=10, pady=(8, 2))

        self.panel = tk.Label(self.root, bg="#000000")
        self.panel.pack(padx=10, pady=(0, 8))

        row = tk.Frame(self.root, bg="#12141a")
        row.pack(fill="x", padx=8, pady=(0, 10))
        self.leg_btn: dict = {}
        self.leg_hz: dict = {}
        for pool in FOOTER_ORDER:
            cell = tk.Frame(row, bg="#12141a")
            cell.pack(side="left", expand=True, padx=1)
            btn = tk.Button(
                cell,
                text=f"  {pool}",
                compound="left",
                font=tkfont.Font(size=10, weight="bold"),
                fg="#ffffff",
                bg="#2a2c33",
                activebackground="#2a2c33",
                relief="flat",
                bd=0,
                padx=5,
                pady=4,
            )
            btn.pack()
            hz = tk.Label(
                cell,
                text=f"{POOL_LABEL[pool]}   0.00 Hz",
                bg="#12141a",
                fg="#8b90a0",
                font=tkfont.Font(size=8),
            )
            hz.pack()
            self.leg_btn[pool] = btn
            self.leg_hz[pool] = hz
            self._paint_leg(pool, 0.0)

        self.root.update_idletasks()
        self.root.update()

    def _paint_leg(self, pool: str, hz: float) -> None:
        on = hz >= self.on_hz
        key = POOL_TO_KEY[pool]
        photo = self._icon(key, on)
        bg = "#1f6f43" if on else "#2a2c33"
        self.leg_btn[pool].configure(
            image=photo,
            bg=bg,
            activebackground="#2e8b57" if on else "#3a3d46",
            highlightthickness=2 if on else 0,
            highlightbackground="#7CFF9A",
        )
        self.leg_hz[pool].configure(
            text=f"{POOL_LABEL[pool]}   {hz:.2f} Hz",
            fg="#b6f5c8" if on else "#8b90a0",
        )
        self._last_on[pool] = on

    def _icon(self, key: str, on: bool):
        cache = (key, on)
        if cache in self._icon_photos:
            return self._icon_photos[cache]
        path = find_icon(self.icons_dir, key) if self.icons_dir else None
        if path is not None:
            im = Image.open(path).convert("RGBA")
            im.thumbnail((self.icon_px, self.icon_px), Image.Resampling.LANCZOS)
            if on:
                ring = Image.new("RGBA", im.size, (0, 0, 0, 0))
                d = ImageDraw.Draw(ring)
                d.rounded_rectangle(
                    (1, 1, im.size[0] - 2, im.size[1] - 2),
                    radius=8,
                    outline=(80, 220, 120, 255),
                    width=3,
                )
                im = Image.alpha_composite(im, ring)
        else:
            im = fallback_icon(key, on, self.icon_px)
        photo = pil_to_tk(self.root, im)
        self._icon_photos[cache] = photo
        return photo

    def push(
        self,
        pool_hz: dict,
        fov=None,
        px=None,
        py=None,
    ) -> None:
        """Redraw hex panel + lamps. Safe to call if the window was closed."""
        try:
            if not self.root.winfo_exists():
                return
        except Exception:
            return
        if fov is not None:
            im = _retina_pil(fov, px, py)
            im = im.resize((self.view_w, self.view_h), Image.Resampling.NEAREST)
            photo = pil_to_tk(self.root, im)
            self._photos = [photo]
            self.panel.configure(image=photo)
        for pool in BUTTONS:
            self._paint_leg(pool, float(pool_hz.get(pool, 0.0)))
        try:
            self.root.update_idletasks()
            self.root.update()
        except Exception:
            pass
