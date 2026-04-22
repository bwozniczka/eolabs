#!/usr/bin/env python3
"""
Ładuje spektra z biblioteki_spektralna (wg manifest.csv) i zapisuje wykresy
dla każdego rodzaju pokrycia: pełne pasma (krzywa λ vs reflektancja) oraz
uśrednienie w szerszych przedziałach widma (wykres słupkowy).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Katalog skryptu → biblioteka i wyjście
LAB5 = Path(__file__).resolve().parent
LIB = LAB5 / "biblioteka_spektralna"
MANIFEST = LIB / "manifest.csv"
OUT = LIB / "wykresy"

# Obcięcie końca widma (artefakty czujnika powyżej ~2460 nm)
WL_MAX_PLOT = 2450.0

# Szersze „pasma” do wykresów słupkowych: (etykieta, λ_min, λ_max) [nm]
BROAD_BANDS: list[tuple[str, float, float]] = [
    ("400–500 nm", 400, 500),
    ("500–600 nm", 500, 600),
    ("600–700 nm", 600, 700),
    ("700–900 nm", 700, 900),
    ("900–1300 nm", 900, 1300),
    ("1300–1700 nm", 1300, 1700),
    ("1700–2100 nm", 1700, 2100),
    ("2100–2450 nm", 2100, 2450),
]

CLASS_LABELS = {
    "las": "Las",
    "pola": "Pola",
    "woda": "Woda",
    "woda-zanieczyszczenia": "Woda (zan.)",
}


def discover_csv_paths() -> dict[str, Path]:
    by_name: dict[str, Path] = {}
    for p in LIB.rglob("*.csv"):
        if p.name == "manifest.csv":
            continue
        by_name[p.name] = p
    return by_name


def load_manifest_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with MANIFEST.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["class"].strip(), row["filename"].strip()))
    return rows


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    wl: list[float] = []
    val: list[float] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or "wavelength" not in header[0].lower():
            f.seek(0)
            reader = csv.reader(f)
        for parts in reader:
            if len(parts) < 2:
                continue
            try:
                wl.append(float(parts[0]))
                val.append(float(parts[1]))
            except ValueError:
                continue
    return np.asarray(wl, dtype=np.float64), np.asarray(val, dtype=np.float64)


def mean_std_spectra(
    paths: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    arrays: list[np.ndarray] = []
    ref_wl: np.ndarray | None = None
    for path in paths:
        wl, v = read_spectrum(path)
        m = wl <= WL_MAX_PLOT
        wl, v = wl[m], v[m]
        if ref_wl is None:
            ref_wl = wl
        elif wl.shape != ref_wl.shape or not np.allclose(wl, ref_wl, rtol=0, atol=0.05):
            return None
        arrays.append(v)
    if not arrays or ref_wl is None:
        return None
    stack = np.stack(arrays, axis=0)
    return ref_wl, stack.mean(axis=0), stack.std(axis=0, ddof=1) if len(arrays) > 1 else np.zeros_like(ref_wl)


def band_means(wl: np.ndarray, values: np.ndarray) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    means: list[float] = []
    for name, lo, hi in BROAD_BANDS:
        m = (wl >= lo) & (wl < hi)
        if not np.any(m):
            labels.append(name)
            means.append(np.nan)
            continue
        labels.append(name)
        means.append(float(np.mean(values[m])))
    return labels, np.asarray(means, dtype=np.float64)


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"Brak manifestu: {MANIFEST}")

    by_name = discover_csv_paths()
    class_to_paths: dict[str, list[Path]] = defaultdict(list)
    missing = 0
    for cls, fname in load_manifest_rows():
        p = by_name.get(fname)
        if p is None:
            missing += 1
            continue
        class_to_paths[cls].append(p)

    if missing:
        print(f"Uwaga: {missing} plików z manifestu nie znaleziono na dysku.")

    OUT.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 150,
            "font.size": 10,
        }
    )

    # --- Krzywe per klasa ---
    for cls, paths in sorted(class_to_paths.items()):
        if not paths:
            continue
        stats = mean_std_spectra(paths)
        if stats is None:
            print(f"Pomijam {cls}: niespójna siatka długości fali.")
            continue
        wl, mean, std = stats
        label = CLASS_LABELS.get(cls, cls)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.fill_between(wl, mean - std, mean + std, alpha=0.25, color="C0", label="±1σ")
        ax.plot(wl, mean, color="C0", lw=1.2, label="Średnia")
        ax.set_xlabel("Długość fali (nm)")
        ax.set_ylabel("Reflektancja")
        ax.set_title(f"Biblioteka spektralna — {label} (n = {len(paths)})")
        ax.set_xlim(wl.min(), wl.max())
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = cls.replace("-", "_")
        fig.savefig(OUT / f"spektrum_srednie_{safe}.png")
        plt.close(fig)

        # --- Słupki: uśrednione szersze pasma ---
        blabels, bmeans = band_means(wl, mean)
        fig2, ax2 = plt.subplots(figsize=(9, 4.5))
        x = np.arange(len(blabels))
        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(blabels)))
        ax2.bar(x, bmeans, color=colors, edgecolor="black", linewidth=0.4)
        ax2.set_xticks(x)
        ax2.set_xticklabels(blabels, rotation=35, ha="right")
        ax2.set_ylabel("Średnia reflektancja w paśmie")
        ax2.set_title(f"Uśrednione pasma — {label}")
        ax2.grid(True, axis="y", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(OUT / f"pasma_srednie_{safe}.png")
        plt.close(fig2)

    # --- Porównanie wszystkich klas na jednym wykresie ---
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for i, (cls, paths) in enumerate(sorted(class_to_paths.items())):
        if not paths:
            continue
        stats = mean_std_spectra(paths)
        if stats is None:
            continue
        wl, mean, _ = stats
        label = CLASS_LABELS.get(cls, cls)
        ax3.plot(wl, mean, lw=1.1, label=f"{label} (n={len(paths)})")
    ax3.set_xlabel("Długość fali (nm)")
    ax3.set_ylabel("Średnia reflektancja")
    ax3.set_title("Porównanie średnich spektrów — wszystkie rodzaje")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)
    if ax3.lines:
        xs = np.concatenate([ln.get_xdata() for ln in ax3.lines])
        ax3.set_xlim(float(xs.min()), float(xs.max()))
    fig3.tight_layout()
    fig3.savefig(OUT / "spektrum_porownanie_wszystkie.png")
    plt.close(fig3)

    print(f"Zapisano wykresy w: {OUT}")


if __name__ == "__main__":
    main()
