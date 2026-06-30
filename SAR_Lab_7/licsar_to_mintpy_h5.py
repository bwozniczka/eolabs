#!/usr/bin/env python3
"""
Convert downloaded COMET-LiCSAR GeoTIFF interferograms into MintPy input HDF5 files.

Input layout expected from the downloader:

licsar_062D_pre20170624/
├── interferograms/
│   ├── 20161227_20170120/
│   │   ├── 20161227_20170120.geo.unw.tif
│   │   └── 20161227_20170120.geo.cc.tif
│   ├── ...
└── metadata/
    ├── 062D_05831_131313.geo.hgt.tif
    ├── 062D_05831_131313.geo.E.tif
    ├── 062D_05831_131313.geo.N.tif
    ├── 062D_05831_131313.geo.U.tif
    └── baselines

Output:

mintpy_project/
└── inputs/
    ├── ifgramStack.h5
    └── geometryGeo.h5

Notes:
    - LiCSAR unwrapped phase is expected to be radians.
    - MintPy expects unwrapPhase in radians.
    - Coherence is normalized to 0-1 if it appears to be stored as 0-255.
    - bperp is estimated from metadata/baselines if possible; otherwise set to zero.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window


PAIR_RE = re.compile(r"^(\d{8})_(\d{8})$")
S1_WAVELENGTH_M = 0.05546576  # Sentinel-1 C-band wavelength, approximate


def find_pairs(licsar_dir: Path) -> list[tuple[str, Path, Path]]:
    ifg_dir = licsar_dir / "interferograms"
    if not ifg_dir.exists():
        raise FileNotFoundError(f"Missing interferograms directory: {ifg_dir}")

    pairs = []

    for pair_dir in sorted(ifg_dir.iterdir()):
        if not pair_dir.is_dir():
            continue

        pair = pair_dir.name
        if not PAIR_RE.match(pair):
            continue

        unw = pair_dir / f"{pair}.geo.unw.tif"
        cc = pair_dir / f"{pair}.geo.cc.tif"

        if not unw.exists():
            print(f"Skipping {pair}: missing {unw.name}")
            continue

        if not cc.exists():
            print(f"Skipping {pair}: missing {cc.name}")
            continue

        pairs.append((pair, unw, cc))

    if not pairs:
        raise RuntimeError("No valid interferogram pairs found.")

    return pairs


def read_baselines(metadata_dir: Path) -> dict[str, float]:
    """
    Parse LiCSAR metadata/baselines file as flexibly as possible.

    Expected idea:
        one date per line, with a perpendicular baseline value somewhere nearby.

    If parsing fails, return an empty dict.
    """
    baseline_file = metadata_dir / "baselines"
    if not baseline_file.exists():
        print("No metadata/baselines file found. bperp will be zero.")
        return {}

    out: dict[str, float] = {}

    with open(baseline_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue

            date_match = re.search(r"\b(20\d{6}|19\d{6})\b", line)
            if not date_match:
                continue

            date = date_match.group(1)

            # All numeric values after removing the date token.
            line_no_date = line.replace(date, " ")
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line_no_date)

            if not nums:
                continue

            # Usually the first floating value is adequate for relative baseline.
            try:
                out[date] = float(nums[0])
            except ValueError:
                continue

    if out:
        print(f"Read baselines for {len(out)} acquisitions.")
    else:
        print("Could not parse metadata/baselines. bperp will be zero.")

    return out


def get_orbit_direction_from_name(path: Path) -> str:
    """
    Frame names usually look like 062D_05831_131313 or 137A_...
    """
    m = re.search(r"\d{3}([AD])_", str(path))
    if not m:
        return "unknown"
    return "descending" if m.group(1) == "D" else "ascending"


def get_window(ds, bbox):
    """
    bbox: west south east north in dataset coordinates.
    """
    if bbox is None:
        return None

    west, south, east, north = bbox
    raw = from_bounds(west, south, east, north, ds.transform)

    # Round to integer pixel offsets/lengths.
    win = raw.round_offsets().round_lengths()

    # Clip to raster bounds.
    col_off = max(0, int(win.col_off))
    row_off = max(0, int(win.row_off))
    col_end = min(ds.width, int(win.col_off + win.width))
    row_end = min(ds.height, int(win.row_off + win.height))

    width = col_end - col_off
    height = row_end - row_off

    if width <= 0 or height <= 0:
        raise ValueError("Requested --bbox does not overlap the raster.")

    return Window(col_off, row_off, width, height)


def read_raster(path: Path, window=None, dtype=np.float32) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(1, window=window).astype(dtype)
        profile = ds.profile.copy()
        transform = ds.window_transform(window) if window is not None else ds.transform
        nodata = ds.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    arr[~np.isfinite(arr)] = 0.0

    profile["transform"] = transform
    profile["height"], profile["width"] = arr.shape

    return arr, profile


def profile_to_mintpy_attrs(profile: dict, project_name: str, orbit_direction: str) -> dict[str, str]:
    transform = profile["transform"]
    length = int(profile["height"])
    width = int(profile["width"])

    crs = profile.get("crs")
    if crs is not None and crs.is_geographic:
        x_unit = "degrees"
        y_unit = "degrees"
    else:
        x_unit = "meters"
        y_unit = "meters"

    # Rasterio transform:
    #   x = c + a * col + b * row
    #   y = f + d * col + e * row
    attrs = {
        "PROJECT_NAME": project_name,
        "PROCESSOR": "licsar",
        "PLATFORM": "Sen",
        "SENSOR": "Sen",
        "WAVELENGTH": f"{S1_WAVELENGTH_M:.8f}",
        "ORBIT_DIRECTION": orbit_direction,
        "LENGTH": str(length),
        "WIDTH": str(width),
        "FILE_LENGTH": str(length),
        "X_FIRST": str(transform.c),
        "Y_FIRST": str(transform.f),
        "X_STEP": str(transform.a),
        "Y_STEP": str(transform.e),
        "X_UNIT": x_unit,
        "Y_UNIT": y_unit,
        "UNIT": "radian",
        "ALOOKS": "1",
        "RLOOKS": "1",
        "CENTER_LINE_UTC": "43200.0",
        "ANTENNA_SIDE": "-1",
        "RANGE_PIXEL_SIZE": "1.0",
        "AZIMUTH_PIXEL_SIZE": "1.0",
        "EARTH_RADIUS": "6371000.0",
        "HEIGHT": "693000.0",
    }

    return attrs


def write_attrs(h5: h5py.File, attrs: dict[str, str]) -> None:
    for k, v in attrs.items():
        h5.attrs[k] = str(v)


def normalize_coherence(cc: np.ndarray) -> np.ndarray:
    finite = cc[np.isfinite(cc)]
    if finite.size == 0:
        return np.zeros_like(cc, dtype=np.float32)

    p99 = np.nanpercentile(finite, 99)

    # LiCSAR coherence is often byte-like 0-255.
    if p99 > 1.5:
        cc = cc / 255.0

    cc = np.clip(cc, 0.0, 1.0)
    cc[~np.isfinite(cc)] = 0.0
    return cc.astype(np.float32)


def create_lat_lon_arrays(profile: dict) -> tuple[np.ndarray, np.ndarray]:
    transform = profile["transform"]
    length = int(profile["height"])
    width = int(profile["width"])

    rows = np.arange(length, dtype=np.float32)
    cols = np.arange(width, dtype=np.float32)

    # Pixel centers.
    xs = transform.c + transform.a * (cols + 0.5)
    ys = transform.f + transform.e * (rows + 0.5)

    lon = np.repeat(xs[np.newaxis, :], length, axis=0).astype(np.float32)
    lat = np.repeat(ys[:, np.newaxis], width, axis=1).astype(np.float32)

    return lat, lon


def find_metadata_file(metadata_dir: Path, suffix: str) -> Path | None:
    matches = sorted(metadata_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def make_geometry(
    licsar_dir: Path,
    out_dir: Path,
    profile: dict,
    attrs: dict[str, str],
    window=None,
    write_latlon: bool = True,
    compression: str | None = "lzf",
) -> None:
    metadata_dir = licsar_dir / "metadata"
    out_file = out_dir / "inputs" / "geometryGeo.h5"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    hgt_file = find_metadata_file(metadata_dir, ".geo.hgt.tif")
    e_file = find_metadata_file(metadata_dir, ".geo.E.tif")
    n_file = find_metadata_file(metadata_dir, ".geo.N.tif")
    u_file = find_metadata_file(metadata_dir, ".geo.U.tif")

    length = int(profile["height"])
    width = int(profile["width"])

    with h5py.File(out_file, "w") as h5:
        geom_attrs = attrs.copy()
        geom_attrs["FILE_TYPE"] = "geometry"
        geom_attrs["UNIT"] = "meter"
        write_attrs(h5, geom_attrs)

        if hgt_file is not None:
            height, _ = read_raster(hgt_file, window=window)
        else:
            print("No height/DEM file found. Writing zero height.")
            height = np.zeros((length, width), dtype=np.float32)

        h5.create_dataset("height", data=height.astype(np.float32), compression=compression)

        if e_file is not None and n_file is not None and u_file is not None:
            E, _ = read_raster(e_file, window=window)
            N, _ = read_raster(n_file, window=window)
            U, _ = read_raster(u_file, window=window)

            # Incidence from vertical. abs(U) avoids convention ambiguity in LOS direction.
            incidence = np.degrees(np.arccos(np.clip(np.abs(U), 0.0, 1.0))).astype(np.float32)

            # MintPy docs define azimuth angle from north, anti-clockwise positive.
            # atan2(E, N) is clockwise-positive from north, so use negative sign.
            azimuth = (-np.degrees(np.arctan2(E, N))).astype(np.float32)

            h5.create_dataset("incidenceAngle", data=incidence, compression=compression)
            h5.create_dataset("azimuthAngle", data=azimuth, compression=compression)
        else:
            print("LOS E/N/U files not found. Writing approximate incidence=35 deg, azimuth=0 deg.")
            h5.create_dataset(
                "incidenceAngle",
                data=np.full((length, width), 35.0, dtype=np.float32),
                compression=compression,
            )
            h5.create_dataset(
                "azimuthAngle",
                data=np.zeros((length, width), dtype=np.float32),
                compression=compression,
            )

        if write_latlon:
            lat, lon = create_lat_lon_arrays(profile)
            h5.create_dataset("latitude", data=lat, compression=compression)
            h5.create_dataset("longitude", data=lon, compression=compression)

    print(f"Wrote {out_file}")


def make_ifgram_stack(
    licsar_dir: Path,
    out_dir: Path,
    pairs: list[tuple[str, Path, Path]],
    profile: dict,
    attrs: dict[str, str],
    window=None,
    compression: str | None = "lzf",
) -> None:
    metadata_dir = licsar_dir / "metadata"
    baseline_map = read_baselines(metadata_dir)

    out_file = out_dir / "inputs" / "ifgramStack.h5"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    length = int(profile["height"])
    width = int(profile["width"])
    num_ifg = len(pairs)

    date_pairs = []
    bperp = np.zeros(num_ifg, dtype=np.float32)

    for i, (pair, _, _) in enumerate(pairs):
        d1, d2 = pair.split("_")
        date_pairs.append([d1.encode("utf-8"), d2.encode("utf-8")])

        if d1 in baseline_map and d2 in baseline_map:
            bperp[i] = baseline_map[d2] - baseline_map[d1]
        else:
            bperp[i] = 0.0

    stack_attrs = attrs.copy()
    stack_attrs["FILE_TYPE"] = "ifgramStack"
    stack_attrs["UNIT"] = "radian"
    stack_attrs["START_DATE"] = min(d[0].decode("utf-8") for d in date_pairs)
    stack_attrs["END_DATE"] = max(d[1].decode("utf-8") for d in date_pairs)
    stack_attrs["DATE12"] = f"{date_pairs[0][0].decode('utf-8')[2:]}-{date_pairs[0][1].decode('utf-8')[2:]}"

    chunks = (1, min(length, 512), min(width, 512))

    with h5py.File(out_file, "w") as h5:
        write_attrs(h5, stack_attrs)

        h5.create_dataset("date", data=np.array(date_pairs, dtype="S8"))
        h5.create_dataset("bperp", data=bperp)
        h5.create_dataset("dropIfgram", data=np.ones(num_ifg, dtype=bool))

        unw_ds = h5.create_dataset(
            "unwrapPhase",
            shape=(num_ifg, length, width),
            dtype=np.float32,
            chunks=chunks,
            compression=compression,
        )

        cc_ds = h5.create_dataset(
            "coherence",
            shape=(num_ifg, length, width),
            dtype=np.float32,
            chunks=chunks,
            compression=compression,
        )

        for i, (pair, unw_path, cc_path) in enumerate(pairs):
            print(f"[{i + 1:03d}/{num_ifg:03d}] {pair}")

            unw, unw_profile = read_raster(unw_path, window=window)
            cc, _ = read_raster(cc_path, window=window)
            cc = normalize_coherence(cc)

            if unw.shape != (length, width):
                raise ValueError(f"Shape mismatch for {unw_path}: {unw.shape} != {(length, width)}")

            if cc.shape != (length, width):
                raise ValueError(f"Shape mismatch for {cc_path}: {cc.shape} != {(length, width)}")

            unw_ds[i, :, :] = unw.astype(np.float32)
            cc_ds[i, :, :] = cc.astype(np.float32)

    print(f"Wrote {out_file}")


def write_basic_config(out_dir: Path) -> None:
    cfg = out_dir / "smallbaselineApp_licsar.cfg"

    text = """# Minimal MintPy config for a pre-built LiCSAR-derived HDF5 stack.
# The HDF5 files are already in ./inputs/, so start MintPy after load_data:
#     smallbaselineApp.py smallbaselineApp_licsar.cfg --start modify_network

mintpy.compute.maxMemory = auto

# Keep this first run simple.
mintpy.unwrapError.method = no
mintpy.troposphericDelay.method = no
mintpy.topographicResidual = no
mintpy.deramp = no

# Useful for a first landslide test; tighten later.
mintpy.networkInversion.weightFunc = coherence
mintpy.networkInversion.minNormVelocity = yes

# Let MintPy choose a reference point automatically at first.

mintpy.reference.date = auto
mintpy.reference.yx = auto
mintpy.reference.maskFile = no
mintpy.reference.coherenceFile = auto
mintpy.reference.minCoherence = 0.3

# Do not geocode; data are already geographic.
mintpy.geocode = no

# Optional outputs can be added later.
mintpy.save.kmz = no
mintpy.save.hdfEos5 = no
"""

    cfg.write_text(text, encoding="utf-8")
    print(f"Wrote {cfg}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--licsar-dir",
        required=True,
        help="Directory containing interferograms/ and metadata/.",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output MintPy project directory.",
    )

    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional geographic subset in lon/lat or projection coordinates.",
    )

    parser.add_argument(
        "--no-latlon",
        action="store_true",
        help="Do not write latitude/longitude datasets to geometryGeo.h5.",
    )

    parser.add_argument(
        "--compression",
        default="lzf",
        choices=["lzf", "gzip", "none"],
        help="HDF5 compression. Default: lzf.",
    )

    args = parser.parse_args()

    licsar_dir = Path(args.licsar_dir).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    compression = None if args.compression == "none" else args.compression

    pairs = find_pairs(licsar_dir)
    print(f"Found {len(pairs)} interferograms.")

    first_unw = pairs[0][1]
    with rasterio.open(first_unw) as ds:
        window = get_window(ds, args.bbox)
        if window is not None:
            _, profile = read_raster(first_unw, window=window)
            print(f"Using subset window: {window}")
            print(f"Subset shape: {profile['height']} x {profile['width']}")
        else:
            _, profile = read_raster(first_unw, window=None)
            print(f"Full shape: {profile['height']} x {profile['width']}")

    project_name = out_dir.name
    orbit_direction = get_orbit_direction_from_name(licsar_dir)
    attrs = profile_to_mintpy_attrs(profile, project_name, orbit_direction)

    make_geometry(
        licsar_dir=licsar_dir,
        out_dir=out_dir,
        profile=profile,
        attrs=attrs,
        window=window,
        write_latlon=not args.no_latlon,
        compression=compression,
    )

    make_ifgram_stack(
        licsar_dir=licsar_dir,
        out_dir=out_dir,
        pairs=pairs,
        profile=profile,
        attrs=attrs,
        window=window,
        compression=compression,
    )

    write_basic_config(out_dir)

    print("\nDone.")
    print(f"MintPy project directory: {out_dir}")
    print("\nNext checks:")
    print(f"  cd {out_dir}")
    print("  info.py inputs/ifgramStack.h5")
    print("  info.py inputs/geometryGeo.h5")
    print("  view.py inputs/geometryGeo.h5 height")
    print("  plot_network.py inputs/ifgramStack.h5")
    print("\nThen run:")
    print("  smallbaselineApp.py smallbaselineApp_licsar.cfg --start modify_network")


if __name__ == "__main__":
    main()