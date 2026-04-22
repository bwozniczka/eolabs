#!/usr/bin/env python3
"""
Hyperspectral BSQ Viewer
------------------------
Browse ENVI/BSQ hyperspectral data cubes.
Click any pixel in the RGB preview to display its full spectral signature.
Export the spectrum to CSV. Optional spectral library: folder + class + auto-save on click.

Usage:
    python viewer.py [path/to/file.hdr]

If no path is given the tool searches known data folders for .hdr files.
Requires Python 3.9+.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

try:
    import spectral.io.envi as envi
except ImportError:
    sys.exit(
        "The 'spectral' library is missing.\n"
        "Install it with:  pip install spectral"
    )

# ── Configuration ─────────────────────────────────────────────────────────────

# Search roots for auto-load (relative to this script)
DATA_DIRS = [
    Path(__file__).parent / "data" / "images",
    Path(__file__).parent / "Obrazy lotnicze",
]

FALLBACK_RGB = (30, 20, 10)

LIBRARY_DEFAULT_CLASSES = (
    "woda",
    "woda-zanieczyszczenia",
    "las",
    "pola"
)
MANIFEST_NAME = "manifest.csv"

# Modern matplotlib defaults (overridden after Tk root exists for DPI)
MPL_STYLE = {
    "figure.facecolor": "#f4f6fb",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#d8dee9",
    "axes.labelcolor": "#2e3440",
    "axes.titlecolor": "#2e3440",
    "text.color": "#2e3440",
    "xtick.color": "#4c566a",
    "ytick.color": "#4c566a",
    "grid.color": "#e5e9f0",
    "grid.linestyle": "-",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.sans-serif": ["SF Pro Display", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.grid": True,
    "axes.axisbelow": True,
}

# ── ENVI header helpers ───────────────────────────────────────────────────────


def find_hdr_files() -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for d in DATA_DIRS:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.hdr")):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


def parse_wavelengths(meta: dict) -> Optional[np.ndarray]:
    wl = meta.get("wavelength")
    if wl:
        return np.array([float(w) for w in wl])
    return None


def get_rgb_bands(meta: dict) -> tuple[int, int, int]:
    db = meta.get("default bands")
    if db and len(db) >= 3:
        return tuple(int(float(v)) - 1 for v in db[:3])
    return FALLBACK_RGB


def get_ignore_value(meta: dict) -> Optional[float]:
    raw = meta.get("data ignore value")
    if raw:
        try:
            return float(str(raw).strip())
        except ValueError:
            pass
    return None


def get_reflectance_scale_factor(meta: dict) -> Optional[float]:
    raw = meta.get("reflectance scale factor")
    if raw:
        try:
            v = float(str(raw).strip())
            return v if v > 0 else None
        except ValueError:
            pass
    return None


# ── Image I/O ─────────────────────────────────────────────────────────────────


def load_image(hdr_path: Path):
    return envi.open(str(hdr_path))


def read_rgb(img, r: int, g: int, b: int, ignore_value: Optional[float]) -> np.ndarray:
    rgb = img.read_bands([r, g, b]).astype(np.float32)
    if ignore_value is not None:
        rgb[rgb >= ignore_value] = np.nan
    rgb[rgb < 0] = np.nan
    for c in range(3):
        ch = rgb[:, :, c]
        p2, p98 = np.nanpercentile(ch, [2, 98])
        rgb[:, :, c] = np.clip((ch - p2) / max(p98 - p2, 1e-6), 0, 1)
    return np.nan_to_num(rgb, nan=0.0)


def read_spectrum(img, row: int, col: int, ignore_value: Optional[float]) -> np.ndarray:
    spec = img.read_pixel(row, col).astype(np.float64)
    if ignore_value is not None:
        spec[spec >= ignore_value] = np.nan
    spec[spec < 0] = np.nan
    return spec


# ── Application ───────────────────────────────────────────────────────────────


class HyperspectralViewer:
    ACCENT = "#2563eb"
    ACCENT_HOVER = "#1d4ed8"
    BG = "#eef1f8"
    CARD = "#ffffff"
    TEXT = "#1e293b"
    MUTED = "#64748b"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hyperspectral Cube Explorer")
        self.root.geometry("1360x780")
        self.root.configure(bg=self.BG)
        self.root.minsize(1024, 640)

        self.img = None
        self.hdr_path: Optional[Path] = None
        self.wavelengths: Optional[np.ndarray] = None
        self.ignore_value: Optional[float] = None
        self.reflectance_scale: Optional[float] = None
        self.rgb_bands: tuple[int, int, int] = FALLBACK_RGB
        self.rgb_display: Optional[np.ndarray] = None
        self.spectrum_raw: Optional[np.ndarray] = None
        self.pixel_pos: Optional[Tuple[int, int]] = None
        # None = pełny widok RGB; inaczej (xlim, ylim) jak po set_xlim/set_ylim
        self._map_view: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        # LMB: krótki klik = piksel, przeciągnięcie = przesuwanie mapy
        self._map_drag_btn: Optional[int] = None
        self._map_press_xy: Optional[Tuple[float, float]] = None
        self._map_last_canvas: Optional[Tuple[float, float]] = None
        self._map_was_panning = False

        self.library_dir: Optional[Path] = None
        self.library_autosave = tk.BooleanVar(value=True)
        self.library_class = tk.StringVar(value=LIBRARY_DEFAULT_CLASSES[0])
        self.library_path_var = tk.StringVar(value="Biblioteka: —")
        self._library_seq: Dict[str, int] = {}

        matplotlib.rcParams.update(MPL_STYLE)

        self._build_styles()
        self._build_ui()
        self._bind_shortcuts()
        self._auto_load()

    def _build_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("App.TFrame", background=self.BG)
        style.configure("Card.TFrame", background=self.CARD)
        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("SF Pro Display", 15, "bold"),
        )
        style.configure(
            "Muted.TLabel",
            background=self.CARD,
            foreground=self.MUTED,
            font=("SF Pro Display", 10),
        )
        style.configure(
            "Status.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("SF Pro Display", 10),
        )
        style.configure(
            "Accent.TButton",
            padding=(10, 6),
            background=self.ACCENT,
            foreground="#ffffff",
            font=("SF Pro Display", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.ACCENT_HOVER), ("pressed", "#1e40af")],
            foreground=[("disabled", "#94a3b8")],
        )

    def _build_ui(self):
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(14, 12))
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="Hyperspectral Cube Explorer", style="Title.TLabel").pack(
            side=tk.LEFT
        )

        bar = ttk.Frame(outer, style="Card.TFrame", padding=(10, 8))
        bar.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(bar, text="Open…", command=self._open_file, style="Accent.TButton").pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(bar, text="Export CSV…", command=self._export_csv).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(bar, text="Save plot PNG…", command=self._export_plot_png).pack(
            side=tk.LEFT, padx=4
        )

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=12, pady=4
        )

        ttk.Label(bar, text="RGB bands (0-based):", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        self.spin_r = tk.Spinbox(bar, from_=0, to=999, width=5, command=self._apply_rgb_bands)
        self.spin_g = tk.Spinbox(bar, from_=0, to=999, width=5, command=self._apply_rgb_bands)
        self.spin_b = tk.Spinbox(bar, from_=0, to=999, width=5, command=self._apply_rgb_bands)
        self.spin_r.pack(side=tk.LEFT, padx=2)
        self.spin_g.pack(side=tk.LEFT, padx=2)
        self.spin_b.pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Apply RGB", command=self._apply_rgb_bands).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ttk.Button(bar, text="Reset RGB", command=self._reset_rgb_bands).pack(
            side=tk.LEFT, padx=(0, 0)
        )

        ttk.Separator(bar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=12, pady=4
        )
        ttk.Label(bar, text="Mapa RGB:", style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(0, 4)
        )
        ttk.Button(bar, text="−", width=3, command=self._map_zoom_out).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bar, text="+", width=3, command=self._map_zoom_in).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bar, text="Pełny widok", command=self._map_zoom_reset).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        lib = ttk.Frame(outer, style="Card.TFrame", padding=(10, 6))
        lib.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            lib,
            text="Nowy folder biblioteki…",
            command=self._library_create_folder,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(lib, text="Otwórz folder…", command=self._library_open_folder).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Label(lib, textvariable=self.library_path_var, style="Muted.TLabel").pack(
            side=tk.LEFT, padx=(8, 8), fill=tk.X, expand=True
        )
        ttk.Checkbutton(
            lib,
            text="Zapis przy kliku",
            variable=self.library_autosave,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(lib, text="Klasa:", style="Muted.TLabel").pack(side=tk.LEFT, padx=(4, 4))
        self.library_combo = ttk.Combobox(
            lib,
            textvariable=self.library_class,
            values=LIBRARY_DEFAULT_CLASSES,
            width=22,
        )
        self.library_combo.pack(side=tk.LEFT, padx=(0, 0))

        self.status_var = tk.StringVar(value="No file loaded.")
        ttk.Label(outer, textvariable=self.status_var, style="Status.TLabel").pack(
            anchor=tk.W, pady=(0, 6)
        )

        plot_card = ttk.Frame(outer, style="Card.TFrame", padding=8)
        plot_card.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(14, 6.8), dpi=100, facecolor=MPL_STYLE["figure.facecolor"])
        self.ax_rgb = self.fig.add_subplot(1, 2, 1)
        self.ax_spec = self.fig.add_subplot(1, 2, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_card)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav_frame = ttk.Frame(plot_card, style="Card.TFrame")
        nav_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, nav_frame)

        self.canvas.mpl_connect("button_press_event", self._on_map_button_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_map_motion)
        self.canvas.mpl_connect("button_release_event", self._on_map_button_release)
        self.canvas.mpl_connect("scroll_event", self._on_map_scroll)

    def _bind_shortcuts(self):
        self.root.bind("<Command-o>", lambda e: self._open_file())
        self.root.bind("<Control-o>", lambda e: self._open_file())
        self.root.bind("<Command-e>", lambda e: self._export_csv())
        self.root.bind("<Control-e>", lambda e: self._export_csv())
        self.root.bind("<Command-s>", lambda e: self._export_plot_png())
        self.root.bind("<Control-s>", lambda e: self._export_plot_png())

    def _set_spin_limits(self):
        if self.img is None:
            return
        nb = self.img.nbands - 1
        for sp in (self.spin_r, self.spin_g, self.spin_b):
            sp.config(to=nb)
        r, g, b = self.rgb_bands
        self.spin_r.delete(0, tk.END)
        self.spin_r.insert(0, str(min(r, nb)))
        self.spin_g.delete(0, tk.END)
        self.spin_g.insert(0, str(min(g, nb)))
        self.spin_b.delete(0, tk.END)
        self.spin_b.insert(0, str(min(b, nb)))

    def _apply_rgb_bands(self):
        if self.img is None:
            return
        try:
            r = int(self.spin_r.get())
            g = int(self.spin_g.get())
            b = int(self.spin_b.get())
        except ValueError:
            messagebox.showwarning("RGB bands", "Enter integer band indices.")
            return
        nb = self.img.nbands
        for name, v in (("R", r), ("G", g), ("B", b)):
            if not (0 <= v < nb):
                messagebox.showwarning("RGB bands", f"{name} band must be in [0, {nb - 1}].")
                return
        self.rgb_bands = (r, g, b)
        self.rgb_display = read_rgb(self.img, r, g, b, self.ignore_value)
        self._map_view = None
        self._refresh_plots()
        if self.hdr_path is not None:
            self.status_var.set(
                f"{self.hdr_path.name}  ·  RGB bands R{r} G{g} B{b}  ·  "
                f"{self.img.nrows}×{self.img.ncols} × {self.img.nbands}"
            )

    def _reset_rgb_bands(self):
        if self.img is None:
            messagebox.showinfo("Reset RGB", "Load an image first.")
            return
        self.rgb_bands = get_rgb_bands(self.img.metadata)
        self._set_spin_limits()
        r, g, b = self.rgb_bands
        self.rgb_display = read_rgb(self.img, r, g, b, self.ignore_value)
        self._map_view = None
        self._refresh_plots()
        if self.hdr_path is not None:
            self.status_var.set(
                f"{self.hdr_path.name}  ·  RGB reset → R{r} G{g} B{b} (header default)  ·  "
                f"{self.img.nrows}×{self.img.ncols} × {self.img.nbands}"
            )

    def _auto_load(self):
        if len(sys.argv) > 1:
            self._load(Path(sys.argv[1]))
            return

        hdrs = find_hdr_files()
        if not hdrs:
            roots = ", ".join(str(d) for d in DATA_DIRS)
            self.status_var.set(f"No .hdr files found. Searched: {roots}")
            return
        if len(hdrs) == 1:
            self._load(hdrs[0])
        else:
            self._pick_file(hdrs)

    def _pick_file(self, hdrs: list[Path]):
        dlg = tk.Toplevel(self.root)
        dlg.title("Wybór datasetu")
        dlg.configure(bg=self.BG)
        dlg.transient(self.root)
        dlg.geometry("640x400")
        dlg.minsize(480, 280)

        def selected_index() -> int:
            sel = lb.curselection()
            if sel:
                return int(sel[0])
            try:
                active = lb.index(tk.ACTIVE)
                if 0 <= active < lb.size():
                    return int(active)
            except tk.TclError:
                pass
            return 0

        def on_ok(_event=None):
            i = selected_index()
            if not (0 <= i < len(hdrs)):
                i = 0
            dlg.grab_release()
            dlg.destroy()
            self._load(hdrs[i])

        def on_cancel(_event=None):
            dlg.grab_release()
            dlg.destroy()
            self.status_var.set("Anulowano wybór datasetu — użyj „Open…” aby wczytać plik.")

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        ttk.Label(
            dlg,
            text="Znaleziono kilka plików .hdr — wybierz jeden i naciśnij „Otwórz wybrany” (lub Enter):",
            background=self.BG,
            foreground=self.TEXT,
            wraplength=600,
        ).pack(padx=14, pady=(12, 6), anchor=tk.W)

        btn_bar = ttk.Frame(dlg, style="App.TFrame", padding=(14, 8))
        btn_bar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(
            btn_bar,
            text="Anuluj",
            command=on_cancel,
        ).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(
            btn_bar,
            text="Otwórz wybrany",
            command=on_ok,
            style="Accent.TButton",
        ).pack(side=tk.RIGHT)

        list_frame = ttk.Frame(dlg, style="App.TFrame")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)

        scroll = ttk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        lb = tk.Listbox(
            list_frame,
            height=10,
            font=("SF Mono", 10),
            selectmode=tk.SINGLE,
            exportselection=False,
            yscrollcommand=scroll.set,
        )
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=lb.yview)

        for h in hdrs:
            lb.insert(tk.END, str(h))
        lb.selection_set(0)
        lb.activate(0)
        lb.see(0)

        dlg.grab_set()
        dlg.bind("<Return>", on_ok)
        dlg.bind("<Escape>", on_cancel)
        lb.bind("<Return>", on_ok)
        lb.bind("<Double-Button-1>", on_ok)

        lb.focus_set()
        dlg.lift()
        self.root.update_idletasks()
        dlg.wait_window()

    def _open_file(self):
        initial = DATA_DIRS[0] if DATA_DIRS[0].exists() else Path.home()
        path = filedialog.askopenfilename(
            title="Open ENVI header (.hdr)",
            initialdir=initial,
            filetypes=[("ENVI header", "*.hdr"), ("All files", "*.*")],
        )
        if path:
            self._load(Path(path))

    def _load(self, hdr_path: Path):
        self.status_var.set(f"Loading {hdr_path.name} …")
        self.root.update_idletasks()
        try:
            self.img = load_image(hdr_path)
            self.hdr_path = hdr_path
            meta = self.img.metadata
            self.wavelengths = parse_wavelengths(meta)
            self.ignore_value = get_ignore_value(meta)
            self.reflectance_scale = get_reflectance_scale_factor(meta)
            self.rgb_bands = get_rgb_bands(meta)
            self.rgb_display = read_rgb(
                self.img, *self.rgb_bands, self.ignore_value
            )
            self.spectrum_raw = None
            self.pixel_pos = None
            self._map_view = None
            self._set_spin_limits()
            self._refresh_plots()
            scale_note = (
                f"  |  scale factor {self.reflectance_scale:g}"
                if self.reflectance_scale
                else ""
            )
            self.status_var.set(
                f"{hdr_path.name}  ·  {self.img.nrows}×{self.img.ncols}  ·  "
                f"{self.img.nbands} bands{scale_note}  ·  Click a pixel for spectrum."
            )
        except Exception as exc:
            messagebox.showerror("Error loading file", str(exc))
            self.status_var.set("Load failed.")

    def _get_rgb_full_limits(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Zwraca (xlim), (y_hi, y_lo) jak w imshow origin=upper."""
        if self.rgb_display is None:
            return None
        nrows, ncols = self.rgb_display.shape[0], self.rgb_display.shape[1]
        return ((-0.5, ncols - 0.5), (nrows - 0.5, -0.5))

    def _clamp_map_limits(
        self, x0: float, x1: float, y_hi: float, y_lo: float
    ) -> tuple[float, float, float, float]:
        fl = self._get_rgb_full_limits()
        if fl is None:
            return x0, x1, y_hi, y_lo
        (fx0, fx1), (fy_hi, fy_lo) = fl
        x0, x1 = sorted((float(x0), float(x1)))
        y_hi, y_lo = float(y_hi), float(y_lo)
        if y_hi < y_lo:
            y_hi, y_lo = y_lo, y_hi

        full_w = fx1 - fx0
        full_h = fy_hi - fy_lo
        if x1 - x0 >= full_w - 1e-9:
            x0, x1 = fx0, fx1
        else:
            if x0 < fx0:
                x1 += fx0 - x0
                x0 = fx0
            if x1 > fx1:
                x0 -= x1 - fx1
                x1 = fx1
            if x1 <= x0:
                x0, x1 = fx0, fx1

        if y_hi - y_lo >= full_h - 1e-9:
            y_hi, y_lo = fy_hi, fy_lo
        else:
            if y_hi > fy_hi:
                y_lo -= y_hi - fy_hi
                y_hi = fy_hi
            if y_lo < fy_lo:
                y_hi += fy_lo - y_lo
                y_lo = fy_lo
            if y_hi <= y_lo:
                y_hi, y_lo = fy_hi, fy_lo

        return x0, x1, y_hi, y_lo

    def _rgb_zoom_at(self, cx: float, cy: float, scale: float) -> None:
        """scale > 1: przybliżenie (węższy zakres). scale < 1: oddalenie."""
        if self.rgb_display is None:
            return
        fl = self._get_rgb_full_limits()
        if fl is None:
            return
        ax = self.ax_rgb
        x0, x1 = ax.get_xlim()
        if x0 > x1:
            x0, x1 = x1, x0
        y_hi, y_lo = ax.get_ylim()
        if y_hi < y_lo:
            y_hi, y_lo = y_lo, y_hi

        new_xw = (x1 - x0) / scale
        new_yh = (y_hi - y_lo) / scale
        if x1 != x0:
            relx = (cx - x0) / (x1 - x0)
        else:
            relx = 0.5
        relx = min(max(relx, 0.0), 1.0)
        if y_hi != y_lo:
            rely = (y_hi - cy) / (y_hi - y_lo)
        else:
            rely = 0.5
        rely = min(max(rely, 0.0), 1.0)

        nx0 = cx - relx * new_xw
        nx1 = cx + (1.0 - relx) * new_xw
        ny_hi = cy + rely * new_yh
        ny_lo = cy - (1.0 - rely) * new_yh

        nx0, nx1, ny_hi, ny_lo = self._clamp_map_limits(nx0, nx1, ny_hi, ny_lo)
        ax.set_xlim(nx0, nx1)
        ax.set_ylim(ny_hi, ny_lo)
        self._map_view = ((nx0, nx1), (ny_hi, ny_lo))
        self.canvas.draw_idle()

    def _rgb_zoom_centered(self, scale: float) -> None:
        if self.rgb_display is None:
            return
        ax = self.ax_rgb
        x0, x1 = ax.get_xlim()
        if x0 > x1:
            x0, x1 = x1, x0
        y_hi, y_lo = ax.get_ylim()
        if y_hi < y_lo:
            y_hi, y_lo = y_lo, y_hi
        cx = (x0 + x1) / 2.0
        cy = (y_hi + y_lo) / 2.0
        self._rgb_zoom_at(cx, cy, scale)

    def _map_zoom_in(self) -> None:
        self._rgb_zoom_centered(1.25)

    def _map_zoom_out(self) -> None:
        self._rgb_zoom_centered(1.0 / 1.25)

    def _map_zoom_reset(self) -> None:
        self._map_view = None
        self._refresh_plots()

    def _on_map_scroll(self, event) -> None:
        if event.inaxes is not self.ax_rgb or self.rgb_display is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            return
        base = 1.15
        scale = base if step > 0 else 1.0 / base
        self._rgb_zoom_at(float(event.xdata), float(event.ydata), scale)

    def _class_slug(self, label: str) -> str:
        s = label.strip().lower().replace(" ", "_")
        s = re.sub(r"[^a-z0-9._-]+", "_", s, flags=re.I)
        s = re.sub(r"_+", "_", s).strip("._-")
        return s or "inne"

    def _update_library_path_label(self) -> None:
        if self.library_dir:
            s = str(self.library_dir.resolve())
            if len(s) > 52:
                s = "…" + s[-49:]
            self.library_path_var.set(f"Folder: {s}")
        else:
            self.library_path_var.set("Biblioteka: —")

    def _scan_library_sequences(self) -> None:
        self._library_seq = {}
        if not self.library_dir or not self.library_dir.is_dir():
            return
        pat = re.compile(r"^(.+)_(\d{3})_r(\d+)_c(\d+)\.csv$")
        for p in self.library_dir.glob("*.csv"):
            if p.name == MANIFEST_NAME:
                continue
            m = pat.match(p.name)
            if m:
                slug, n = m.group(1), int(m.group(2))
                self._library_seq[slug] = max(self._library_seq.get(slug, 0), n)

    def _write_spectrum_csv(self, path: Path, spectrum: np.ndarray) -> None:
        x = (
            self.wavelengths
            if self.wavelengths is not None
            else np.arange(len(spectrum))
        )
        xcol = "wavelength_nm" if self.wavelengths is not None else "band"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([xcol, "value"])
            for xi, raw in zip(x, spectrum):
                writer.writerow([float(xi), "" if np.isnan(raw) else float(raw)])

    def _append_manifest(
        self, class_label: str, row: int, col: int, filename: str
    ) -> None:
        if not self.library_dir:
            return
        mp = self.library_dir / MANIFEST_NAME
        hdr_name = self.hdr_path.name if self.hdr_path else ""
        new_file = not mp.exists()
        with open(mp, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["class", "row", "col", "filename", "source_hdr"])
            w.writerow([class_label, row, col, filename, hdr_name])

    def _save_to_library(self, row: int, col: int) -> Optional[Path]:
        if not self.library_dir or self.spectrum_raw is None:
            return None
        slug = self._class_slug(self.library_class.get())
        n = self._library_seq.get(slug, 0) + 1
        self._library_seq[slug] = n
        fname = f"{slug}_{n:03d}_r{row}_c{col}.csv"
        path = self.library_dir / fname
        self._write_spectrum_csv(path, self.spectrum_raw)
        self._append_manifest(self.library_class.get().strip(), row, col, fname)
        return path

    def _library_create_folder(self) -> None:
        parent = filedialog.askdirectory(title="Katalog nadrzędny dla biblioteki spektralnej")
        if not parent:
            return
        name = simpledialog.askstring(
            "Nowy folder",
            "Nazwa folderu biblioteki:",
            initialvalue="biblioteka_spektralna",
            parent=self.root,
        )
        if not name or not name.strip():
            return
        folder = Path(parent) / name.strip()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Biblioteka", str(exc))
            return
        self.library_dir = folder
        self._scan_library_sequences()
        self._update_library_path_label()
        self.status_var.set(f"Biblioteka: {folder}")

    def _library_open_folder(self) -> None:
        path = filedialog.askdirectory(title="Folder biblioteki spektralnej")
        if not path:
            return
        folder = Path(path)
        if not folder.is_dir():
            messagebox.showerror("Biblioteka", "Nieprawidłowy folder.")
            return
        self.library_dir = folder
        self._scan_library_sequences()
        self._update_library_path_label()
        self.status_var.set(f"Biblioteka: {folder}")

    def _apply_pixel_pick(self, row: int, col: int) -> None:
        self.pixel_pos = (row, col)
        self.spectrum_raw = read_spectrum(self.img, row, col, self.ignore_value)
        self._refresh_plots()
        parts = [f"Piksel ({row}, {col})", "⌘E CSV", "⌘S PNG"]
        if (
            self.library_dir
            and self.library_autosave.get()
            and self.spectrum_raw is not None
        ):
            try:
                saved = self._save_to_library(row, col)
                if saved is not None:
                    parts.append(f"zapis: {saved.name}")
            except OSError as exc:
                messagebox.showerror("Zapis biblioteki", str(exc))
        self.status_var.set("  ·  ".join(parts))

    def _on_map_button_press(self, event) -> None:
        if event.button != 1:
            return
        if event.inaxes is self.ax_rgb and self.img is not None:
            self._map_drag_btn = 1
            self._map_press_xy = (float(event.x), float(event.y))
            self._map_last_canvas = (float(event.x), float(event.y))
            self._map_was_panning = False
        else:
            self._map_drag_btn = None

    def _on_map_motion(self, event) -> None:
        if self._map_drag_btn != 1 or self.rgb_display is None:
            return
        if self._map_press_xy is None:
            return
        px0, py0 = self._map_press_xy
        dist = math.hypot(float(event.x) - px0, float(event.y) - py0)
        if dist <= 5.0:
            return
        if not self._map_was_panning:
            self._map_was_panning = True
            self._map_last_canvas = (float(event.x), float(event.y))
            return

        ax = self.ax_rgb
        if self._map_last_canvas is None:
            return
        inv = ax.transData.inverted()
        try:
            xd0, yd0 = inv.transform(self._map_last_canvas)
            xd1, yd1 = inv.transform((event.x, event.y))
        except (ValueError, OverflowError):
            return
        dx = float(xd1 - xd0)
        dy = float(yd1 - yd0)
        self._map_last_canvas = (float(event.x), float(event.y))

        x0, x1 = ax.get_xlim()
        if x0 > x1:
            x0, x1 = x1, x0
        nx0, nx1 = x0 - dx, x1 - dx
        yt, yb = ax.get_ylim()
        if yt < yb:
            yt, yb = yb, yt
        ny_t, ny_b = yt - dy, yb - dy
        nx0, nx1, ny_t, ny_b = self._clamp_map_limits(nx0, nx1, ny_t, ny_b)
        ax.set_xlim(nx0, nx1)
        ax.set_ylim(ny_t, ny_b)
        self._map_view = ((nx0, nx1), (ny_t, ny_b))
        self.canvas.draw_idle()

    def _on_map_button_release(self, event) -> None:
        if event.button != 1:
            return
        if self._map_drag_btn != 1:
            return
        self._map_drag_btn = None
        was_pan = self._map_was_panning
        self._map_was_panning = False
        self._map_last_canvas = None
        self._map_press_xy = None

        if was_pan or self.img is None:
            return
        if event.inaxes is not self.ax_rgb:
            return
        if event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if not (0 <= row < self.img.nrows and 0 <= col < self.img.ncols):
            return
        self._apply_pixel_pick(row, col)

    def _refresh_plots(self):
        accent = self.ACCENT
        fill = "#93c5fd"

        self.ax_rgb.clear()
        if self.rgb_display is not None:
            self.ax_rgb.imshow(
                self.rgb_display,
                interpolation="bilinear",
                aspect="auto",
            )
            if self.pixel_pos:
                row, col = self.pixel_pos
                self.ax_rgb.scatter(
                    [col],
                    [row],
                    s=120,
                    facecolors="none",
                    edgecolors=accent,
                    linewidths=2.5,
                )
                self.ax_rgb.plot(col, row, "+", color="#f97316", markersize=10, markeredgewidth=2)
        self.ax_rgb.set_title(
            "RGB — klik: piksel  ·  przeciągnij: przesuń mapę",
            color=self.TEXT,
            fontsize=11,
            pad=8,
        )
        self.ax_rgb.set_facecolor("#fafbfc")
        self.ax_rgb.axis("off")
        if self.rgb_display is not None and self._map_view is not None:
            (xl, xr), (y_hi, y_lo) = self._map_view
            xl, xr, y_hi, y_lo = self._clamp_map_limits(xl, xr, y_hi, y_lo)
            self.ax_rgb.set_xlim(xl, xr)
            self.ax_rgb.set_ylim(y_hi, y_lo)
            self._map_view = ((xl, xr), (y_hi, y_lo))

        self.ax_spec.clear()
        if self.spectrum_raw is not None:
            row, col = self.pixel_pos  # type: ignore
            x = (
                self.wavelengths
                if self.wavelengths is not None
                else np.arange(len(self.spectrum_raw))
            )
            xlabel = "Wavelength (nm)" if self.wavelengths is not None else "Band index"
            y_plot = self.spectrum_raw
            ylab = "Value (DN / scaled)"
            (line,) = self.ax_spec.plot(
                x,
                y_plot,
                linewidth=2.0,
                color=accent,
                solid_capstyle="round",
            )
            yv = np.asarray(y_plot, dtype=float)
            xv = np.asarray(x, dtype=float)
            mask = np.isfinite(yv)
            if np.any(mask):
                yv_m = yv[mask]
                xv_m = xv[mask]
                lo = float(np.nanmin(yv_m))
                y0 = np.full_like(yv_m, min(lo, 0.0))
                self.ax_spec.fill_between(
                    xv_m,
                    y0,
                    yv_m,
                    alpha=0.2,
                    color=fill,
                    linewidth=0,
                )
            self.ax_spec.set_title(
                f"Spectrum  ·  row {row}, col {col}",
                color=self.TEXT,
                fontsize=11,
                pad=8,
            )
            self.ax_spec.set_xlabel(xlabel, fontsize=10)
            self.ax_spec.set_ylabel(ylab, fontsize=10)
            for spine in self.ax_spec.spines.values():
                spine.set_color("#cbd5e1")
            self.ax_spec.grid(True, alpha=0.85, linestyle="-", linewidth=0.6)

            valid = np.isfinite(self.spectrum_raw)
            if np.any(valid):
                mn = float(np.nanmin(self.spectrum_raw))
                mx = float(np.nanmax(self.spectrum_raw))
                self.ax_spec.text(
                    0.02,
                    0.98,
                    f"Range: [{mn:.4g}, {mx:.4g}]",
                    transform=self.ax_spec.transAxes,
                    fontsize=8,
                    color=self.MUTED,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#e2e8f0"),
                )
        else:
            self.ax_spec.set_title("Spectral signature", color=self.TEXT, fontsize=11, pad=8)
            self.ax_spec.text(
                0.5,
                0.5,
                "Click a pixel in the RGB image",
                ha="center",
                va="center",
                transform=self.ax_spec.transAxes,
                color=self.MUTED,
                fontsize=12,
            )
            self.ax_spec.set_facecolor("#fafbfc")
            for spine in self.ax_spec.spines.values():
                spine.set_color("#cbd5e1")

        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, wspace=0.22)
        self.canvas.draw()

    def _export_csv(self):
        if self.spectrum_raw is None:
            messagebox.showinfo("Nothing to export", "Click on a pixel first.")
            return
        row, col = self.pixel_pos  # type: ignore
        default_name = f"spectrum_r{row}_c{col}.csv"
        path = filedialog.asksaveasfilename(
            title="Save spectrum as CSV",
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self._write_spectrum_csv(Path(path), self.spectrum_raw)
        except OSError as exc:
            messagebox.showerror("Export CSV", str(exc))
            return

        self.status_var.set(f"Saved → {path}")
        messagebox.showinfo("Saved", f"Spectrum exported to:\n{path}")

    def _export_plot_png(self):
        if self.spectrum_raw is None:
            messagebox.showinfo("Nothing to save", "Click on a pixel first.")
            return
        row, col = self.pixel_pos  # type: ignore
        default_name = f"spectrum_r{row}_c{col}.png"
        path = filedialog.asksaveasfilename(
            title="Save spectrum plot",
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
        )
        if not path:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=self.fig.get_facecolor())
        self.status_var.set(f"Plot saved → {path}")


if __name__ == "__main__":
    root = tk.Tk()
    HyperspectralViewer(root)
    root.mainloop() 
