"""Tkinter applet: raw screenshot → fly.py FOV / hex drive → six-leg labels.

Keys
----
1  Left front   (LF)
2  Left mid     (LM)
3  Left hind    (LH)
4  Right mid    (RM)
5  Right hind   (RH)
E  Right front  (RF)
Enter / ✓      write row and advance
Esc            quit
BackSpace      previous frame
"""

from __future__ import annotations

import csv
from pathlib import Path

tk = None  # set by _need_tk()
messagebox = None

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .constants import BUTTONS, FOV_PX, VISION_HZ_MAX
from .vision import (
    fit_frame_to_fov,
    fov_with_sample_dots,
    frame_to_fov,
    hex_rates_from_fov,
    hex_retina_image,
    load_overlay,
    luma,
    luma_uint8,
    maybe_composite,
    to_uint8_rgb,
)

# Key → pool. E is right front as requested; 1–5 are the other five.
KEY_TO_POOL = {
    "1": "LF",
    "2": "LM",
    "3": "LH",
    "4": "RM",
    "5": "RH",
    "e": "RF",
    "E": "RF",
}

POOL_TO_KEY = {
    "LF": "1",
    "LM": "2",
    "LH": "3",
    "RM": "4",
    "RH": "5",
    "RF": "E",
}

POOL_LABEL = {
    "LF": "Left front",
    "RF": "Right front",
    "LM": "Left mid",
    "RM": "Right mid",
    "LH": "Left hind",
    "RH": "Right hind",
}

# Footer order: left column front→hind, then right column front→hind.
FOOTER_ORDER = ("LF", "LM", "LH", "RF", "RM", "RH")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
ICON_EXTS = (".png", ".gif", ".ppm", ".pgm", ".gif")


def list_frames(folder: Path) -> list[Path]:
    folder = Path(folder)
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def find_icon(icons_dir: Path | None, key: str) -> Path | None:
    if icons_dir is None:
        return None
    icons_dir = Path(icons_dir)
    if not icons_dir.is_dir():
        return None
    for name in (key, key.upper(), key.lower()):
        for ext in ICON_EXTS:
            cand = icons_dir / f"{name}{ext}"
            if cand.is_file():
                return cand
    return None


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fallback_icon(key: str, selected: bool, size: int = 56) -> Image.Image:
    bg = (40, 130, 70) if selected else (46, 48, 56)
    fg = (255, 255, 255)
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((1, 1, size - 2, size - 2), radius=10, fill=bg, outline=(200, 200, 200))
    font = _font(max(18, size // 2))
    text = key.upper()
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - tw) / 2, (size - th) / 2 - 2), text, fill=fg, font=font)
    return im


def pil_to_tk(root: tk.Misc, im: Image.Image) -> tk.PhotoImage:
    try:
        from PIL import ImageTk

        return ImageTk.PhotoImage(im, master=root)
    except Exception:
        import io

        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PPM")
        return tk.PhotoImage(master=root, data=buf.getvalue())


def fit_to_box(im: Image.Image, box_w: int, box_h: int) -> Image.Image:
    if im.width == 0 or im.height == 0:
        return im
    scale = min(box_w / im.width, box_h / im.height, 1.0)
    # Allow upscale for the 256 FOV so the pixels are visible.
    if im.width <= 256 and im.height <= 256:
        scale = min(box_w / im.width, box_h / im.height)
    nw = max(1, int(im.width * scale))
    nh = max(1, int(im.height * scale))
    resample = (
        Image.Resampling.NEAREST
        if (im.width <= 256 and scale >= 2)
        else Image.Resampling.BILINEAR
    )
    return im.resize((nw, nh), resample)


def _need_tk():
    global tk, messagebox
    if tk is not None:
        return
    try:
        import tkinter as _tk
        from tkinter import messagebox as _mb
    except ModuleNotFoundError as e:
        raise SystemExit(
            "tkinter is not installed. On Windows use the official python.org "
            "installer (tcl/tk is included). On Linux: sudo apt install python3-tk"
        ) from e
    tk = _tk
    messagebox = _mb


class LabelApp:
    def __init__(
        self,
        frames: list[Path],
        labels_dir: Path,
        guidance: Path | None,
        hex_xy: tuple[np.ndarray, np.ndarray] | None,
        icons_dir: Path | None,
        start_at: int = 0,
    ) -> None:
        _need_tk()
        self.frames = frames
        self.labels_dir = Path(labels_dir)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.labels_dir / "index.csv"
        self.overlay = load_overlay(guidance) if guidance else None
        self.hex_px, self.hex_py = hex_xy if hex_xy is not None else (None, None)
        self.icons_dir = icons_dir
        self.idx = max(0, min(start_at, len(frames) - 1)) if frames else 0
        self.existing = self._load_csv()
        self.selected = {b: False for b in BUTTONS}
        self._photos: list[tk.PhotoImage] = []
        self._icon_photos: dict[tuple[str, bool], tk.PhotoImage] = {}
        self._view_cache: dict[str, tuple] = {}

        self.root = tk.Tk()
        self.root.title("flytrain labeler")
        self.root.configure(bg="#1b1c20")
        self.root.geometry("1280x820")
        self.root.minsize(960, 640)

        self._build()
        self._bind()
        self._resize_job = None
        self.root.bind("<Configure>", self._on_resize)
        if self.frames:
            self.root.after(80, self._show_current)
        else:
            messagebox.showerror("No frames", "That folder has no screenshots.")

    def _load_csv(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not self.csv_path.exists():
            return out
        with self.csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get("path", "")
                out[p] = row
                try:
                    out[str(Path(p).resolve())] = row
                except OSError:
                    pass
                out[Path(p).name] = row
        return out

    def _row_for(self, path: Path) -> dict | None:
        keys = [str(path), str(path.resolve()), path.name]
        try:
            rel = str(path.resolve().relative_to(self.labels_dir.resolve()))
            keys.append(rel)
        except ValueError:
            pass
        for k in keys:
            if k in self.existing:
                return self.existing[k]
        return None

    def _build(self) -> None:
        bg = "#1b1c20"
        fg = "#e8e8ea"
        muted = "#9aa0aa"

        top = tk.Frame(self.root, bg=bg)
        top.pack(fill="x", padx=12, pady=(10, 4))
        self.status = tk.Label(
            top, text="", bg=bg, fg=fg, font=("Segoe UI", 12, "bold"), anchor="w"
        )
        self.status.pack(side="left", fill="x", expand=True)
        self.stats = tk.Label(top, text="", bg=bg, fg=muted, font=("Segoe UI", 10))
        self.stats.pack(side="right")

        hint = tk.Label(
            self.root,
            text="1 LF   2 LM   3 LH   4 RM   5 RH   E RF    Enter save+next    Backspace previous",
            bg=bg,
            fg=muted,
            font=("Segoe UI", 9),
            anchor="w",
        )
        hint.pack(fill="x", padx=12)

        grid = tk.Frame(self.root, bg=bg)
        grid.pack(fill="both", expand=True, padx=10, pady=6)
        for r in range(2):
            grid.rowconfigure(r, weight=1)
        for c in range(2):
            grid.columnconfigure(c, weight=1)

        self.panels = {}
        titles = [
            (0, 0, "raw", "Raw screenshot"),
            (0, 1, "guided", "Guidance overlay + letterbox (256² FOV)"),
            (1, 0, "luma", "Luma (R+G+B)/3  — this is the intensity"),
            (1, 1, "hex", "Hex-sampled OL drive (one pixel per driven cell)"),
        ]
        for r, c, key, title in titles:
            cell = tk.Frame(grid, bg="#101114", highlightbackground="#3a3d46", highlightthickness=1)
            cell.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            tk.Label(cell, text=title, bg="#101114", fg=muted, font=("Segoe UI", 9)).pack(
                anchor="w", padx=6, pady=(4, 0)
            )
            lbl = tk.Label(cell, bg="#000000")
            lbl.pack(fill="both", expand=True, padx=4, pady=4)
            cap = tk.Label(cell, text="", bg="#101114", fg=muted, font=("Segoe UI", 8), anchor="w")
            cap.pack(fill="x", padx=6, pady=(0, 4))
            self.panels[key] = (lbl, cap)

        footer = tk.Frame(self.root, bg=bg)
        footer.pack(fill="x", padx=10, pady=(0, 10))

        self.leg_btns: dict[str, tk.Button] = {}
        legs = tk.Frame(footer, bg=bg)
        legs.pack(side="left", fill="x", expand=True)
        for pool in FOOTER_ORDER:
            b = tk.Button(
                legs,
                text=f"{POOL_TO_KEY[pool]}  {POOL_LABEL[pool]}",
                command=lambda p=pool: self._toggle(p),
                bd=0,
                padx=8,
                pady=6,
                font=("Segoe UI", 10, "bold"),
                cursor="hand2",
            )
            b.pack(side="left", padx=4, pady=4)
            self.leg_btns[pool] = b

        right = tk.Frame(footer, bg=bg)
        right.pack(side="right")
        self.back_btn = tk.Button(
            right,
            text="←",
            command=self._prev,
            font=("Segoe UI", 16),
            bd=0,
            padx=12,
            pady=6,
            bg="#2a2c33",
            fg=fg,
            cursor="hand2",
        )
        self.back_btn.pack(side="left", padx=4)
        self.commit_btn = tk.Button(
            right,
            text="  ✓  ",
            command=self._commit,
            font=("Segoe UI", 20, "bold"),
            bd=0,
            padx=16,
            pady=4,
            bg="#2e8b57",
            fg="white",
            cursor="hand2",
            activebackground="#3cb371",
        )
        self.commit_btn.pack(side="left", padx=4)

        self._refresh_leg_style()

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget is not self.root:
            return
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(180, self._paint_panels)

    def _bind(self) -> None:
        self.root.bind("<Key>", self._on_key)
        self.root.bind("<Return>", lambda e: self._commit())
        self.root.bind("<KP_Enter>", lambda e: self._commit())
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("<BackSpace>", lambda e: self._prev())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

    def _on_key(self, event: tk.Event) -> None:
        if event.keysym in ("Return", "KP_Enter", "BackSpace", "Escape"):
            return
        ch = event.char or ""
        pool = KEY_TO_POOL.get(ch)
        if pool:
            self._toggle(pool)

    def _toggle(self, pool: str) -> None:
        self.selected[pool] = not self.selected[pool]
        self._refresh_leg_style()

    def _reset_legs(self) -> None:
        for b in BUTTONS:
            self.selected[b] = False
        self._refresh_leg_style()

    def _load_icon_photo(self, key: str, selected: bool) -> tk.PhotoImage | None:
        cache_key = (key, selected)
        if cache_key in self._icon_photos:
            return self._icon_photos[cache_key]
        path = find_icon(self.icons_dir, key)
        if path is not None:
            im = Image.open(path).convert("RGBA")
            im.thumbnail((56, 56), Image.Resampling.LANCZOS)
            if selected:
                # green ring so the custom art still reads as "on"
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
            im = fallback_icon(key, selected, 56)
        photo = pil_to_tk(self.root, im)
        self._icon_photos[cache_key] = photo
        return photo

    def _refresh_leg_style(self) -> None:
        for pool, btn in self.leg_btns.items():
            on = self.selected[pool]
            key = POOL_TO_KEY[pool]
            photo = self._load_icon_photo(key, on)
            btn.configure(
                image=photo,
                compound="left",
                bg="#1f6f43" if on else "#2a2c33",
                fg="#ffffff",
                activebackground="#2e8b57" if on else "#3a3d46",
                highlightthickness=2 if on else 0,
                highlightbackground="#7CFF9A",
            )

    def _rel_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.labels_dir.resolve()))
        except ValueError:
            return str(path)

    def _commit(self) -> None:
        if not self.frames:
            return
        path = self.frames[self.idx]
        rel = self._rel_path(path)
        row = {
            "path": rel,
            **{b: int(self.selected[b]) for b in BUTTONS},
            "clip_id": path.stem,
            "t": 0.0,
            "already_composited": 0,
        }
        self._upsert_csv(row)
        self.existing[rel] = {k: str(v) for k, v in row.items()}
        self.existing[path.name] = self.existing[rel]
        if self.idx + 1 >= len(self.frames):
            messagebox.showinfo("Done", f"Labeled all {len(self.frames)} frames.\n{self.csv_path}")
            return
        self.idx += 1
        self._reset_legs()
        self._show_current()

    def _upsert_csv(self, row: dict) -> None:
        fieldnames = ["path", *BUTTONS, "clip_id", "t", "already_composited"]
        rows = []
        if self.csv_path.exists():
            with self.csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
        replaced = False
        target = row["path"]
        target_name = Path(target).name
        out_rows = []
        for old in rows:
            if old.get("path") == target or Path(old.get("path", "")).name == target_name:
                out_rows.append(row)
                replaced = True
            else:
                out_rows.append(old)
        if not replaced:
            out_rows.append(row)
        with self.csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in out_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})

    def _prev(self) -> None:
        if self.idx <= 0:
            return
        self.idx -= 1
        self._reset_legs()
        self._show_current()

    def _apply_saved_bits(self, path: Path) -> None:
        saved = self._row_for(path)
        if not saved:
            return
        for b in BUTTONS:
            val = saved.get(b, "0")
            try:
                self.selected[b] = float(val) > 0.5
            except (TypeError, ValueError):
                self.selected[b] = str(val) in {"1", "true", "True"}
        self._refresh_leg_style()

    def _paint_panels(self) -> None:
        if not self._view_cache:
            return
        for key, (im, caption) in self._view_cache.items():
            lbl, cap = self.panels[key]
            lbl.update_idletasks()
            box_w = max(lbl.winfo_width(), 280)
            box_h = max(lbl.winfo_height(), 160)
            shown = fit_to_box(im, box_w - 8, box_h - 8)
            photo = pil_to_tk(self.root, shown)
            self._photos.append(photo)
            lbl.configure(image=photo)
            cap.configure(text=caption)

    def _show_current(self) -> None:
        path = self.frames[self.idx]
        n_done = sum(
            1
            for p in self.frames
            if self._row_for(p) is not None
        )
        self.status.configure(text=f"{self.idx + 1} / {len(self.frames)}   {path.name}")
        self.stats.configure(text=f"{n_done} saved in {self.csv_path.name}")
        
        raw = Image.open(path).convert("RGBA")
        fov = frame_to_fov(path, overlay=self.overlay, canvas=FOV_PX, already_fov=False)
        fov_im = Image.fromarray(to_uint8_rgb(fov), mode="RGB")
        luma_im = Image.fromarray(luma_uint8(fov), mode="RGB")

        mean_luma = float(luma(fov).mean())
        raw_cap = f"{raw.width}×{raw.height}"
        g_cap = (
            f"overlay {'on' if self.overlay is not None else 'off'}  "
            f"FOV {FOV_PX}²  mean luma {mean_luma:.3f}"
        )
        self._view_cache = {
            "raw": (raw.convert("RGB"), raw_cap),
            "guided": (fov_im, g_cap),
            "luma": (
                luma_im,
                "Every driven OL cell reads one pixel of this after hex mapping.",
            ),
        }

        if self.hex_px is not None:
            retina = hex_retina_image(fov, self.hex_px, self.hex_py, radius=1)
            dots = fov_with_sample_dots(fov, self.hex_px, self.hex_py)
            # stack retina | sample sites so both readings are visible
            left = Image.fromarray(retina, mode="RGB")
            right = Image.fromarray(dots, mode="RGB")
            pair = Image.new("RGB", (left.width * 2 + 4, left.height), (16, 16, 18))
            pair.paste(left, (0, 0))
            pair.paste(right, (left.width + 4, 0))
            rates = hex_rates_from_fov(fov, self.hex_px, self.hex_py)
            self._view_cache["hex"] = (
                pair,
                f"{len(self.hex_px):,} hex cells   "
                f"mean drive {float(rates.mean()):.2f} Hz   "
                f"max {float(rates.max()):.1f} / {VISION_HZ_MAX:.0f}   "
                f"left = painted samples, right = sites on dim FOV",
            )
        else:
            note = Image.new("RGB", (FOV_PX, FOV_PX), (12, 12, 14))
            d = ImageDraw.Draw(note)
            d.text(
                (16, 110),
                "No hex coordinates loaded.\n"
                "Pass --data C:\\Dev\\flywow\\data\n"
                "to paint the real OL lattice.",
                fill=(180, 180, 190),
                font=_font(14),
            )
            self._view_cache["hex"] = (note, "hex sampler missing — luma panel is still exact")

        self._paint_panels()
        saved = self._row_for(path)
        if saved:
            self._apply_saved_bits(path)
        else:
            self._reset_legs()

    def run(self) -> None:
        _need_tk()
        self.root.mainloop()


def resolve_icons(explicit: Path | None, labels_dir: Path, frames_dir: Path) -> Path | None:
    cands = []
    if explicit:
        cands.append(Path(explicit))
    cands.extend(
        [
            labels_dir / "icons",
            frames_dir / "icons",
            frames_dir.parent / "icons",
            Path(__file__).resolve().parent.parent / "icons",
        ]
    )
    for c in cands:
        if c.is_dir() and any(c.glob("*.*")):
            return c
    return explicit


def launch(
    frames_dir: Path,
    labels_dir: Path,
    guidance: Path | None = None,
    data_dir: Path | None = None,
    icons_dir: Path | None = None,
    synth: bool = False,
) -> None:
    frames = list_frames(frames_dir)
    if not frames:
        raise SystemExit(f"No screenshots in {frames_dir}")

    hex_xy = None
    if synth:
        from .synth import make_synthetic_connectome

        net = make_synthetic_connectome()
        hex_xy = (net.hex_px, net.hex_py)
        print(f"synth hex cells {len(net.hex_px)}")
    elif data_dir is not None:
        from .connectome import load_hex_sampler

        sampler = load_hex_sampler(Path(data_dir), FOV_PX)
        if sampler:
            hex_xy = (sampler["px"], sampler["py"])
            print(f"hex-driven cells {sampler['n']:,} from {data_dir}")
        else:
            print(f"no hex coordinates in {data_dir}; hex panel will be empty")

    icons = resolve_icons(icons_dir, Path(labels_dir), Path(frames_dir))
    if icons:
        print(f"icons {icons}")

    # Skip straight to the first unlabeled frame when possible.
    start = 0
    app_probe_csv = Path(labels_dir) / "index.csv"
    if app_probe_csv.exists():
        labeled_names = set()
        with app_probe_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                labeled_names.add(Path(row.get("path", "")).name)
        for i, p in enumerate(frames):
            if p.name not in labeled_names:
                start = i
                break
        else:
            start = 0

    app = LabelApp(
        frames=frames,
        labels_dir=Path(labels_dir),
        guidance=Path(guidance) if guidance else None,
        hex_xy=hex_xy,
        icons_dir=icons,
        start_at=start,
    )
    app.run()
