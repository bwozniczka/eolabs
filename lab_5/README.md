# Hyperspectral BSQ Viewer

A minimal desktop tool for exploring ENVI/BSQ hyperspectral data cubes.

## Features

- Loads any ENVI/BSQ image from `data/images/` (reads the `.hdr` sidecar for metadata)
- Displays an RGB preview using the bands declared in `default_bands` of the header
- Click any pixel to see its full spectral signature as a chart
- Export the selected spectrum to a two-column CSV (`wavelength_nm`, `value`)

## Requirements

Python 3.10+ and the packages listed in `requirements.txt`.

## Installation

```bash
pip install -r requirements.txt
```

If you are using conda:

```bash
conda install -c conda-forge numpy matplotlib
pip install spectral
```

## Running

```bash
# auto-detect dataset in data/images/
python viewer.py

# open a specific file directly
python viewer.py data/images/221000_Odra_HS_Blok_A_008_VS_join_atm.hdr
```

On startup the tool will:

1. Scan `data/images/` for `.hdr` files.
2. If only one is found it opens immediately; if multiple are found a small picker dialog appears.
3. The RGB preview loads (reads 3 bands — fast even for large cubes).
4. Click any pixel → spectral signature appears on the right.
5. Click **Export spectrum to CSV…** to save the current spectrum.

## Data format assumptions

The tool reads standard ENVI headers (`.hdr` + `.bsq` pairs). The following header keys are used when present:

| Key | Purpose |
|---|---|
| `samples`, `lines`, `bands` | Image dimensions |
| `data type` | Pixel data type (e.g. `2` = int16) |
| `byte order` | Endianness |
| `interleave` | Must be `BSQ` |
| `default bands` | 1-based RGB band indices for preview |
| `wavelength` | Per-band wavelengths (nm) for the X axis |
| `data ignore value` | No-data threshold; masked in both RGB and spectrum |
| `reflectance scale factor` | Informational only (not applied automatically) |

To adapt the tool to a different sensor, edit `get_rgb_bands()` and `FALLBACK_RGB` at the top of `viewer.py`.

## Notes

- The `.bsq` files are excluded from the repository via `.gitignore` due to their size (7–18 GB each). Download the data from OneDrive and place it in `data/images/`.
- The tool never loads the full cube into RAM. Band reads are memory-mapped by the `spectral` library; reading one pixel's spectrum requires only 456 × 2 bytes of I/O.
