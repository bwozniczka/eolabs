# EO Labs

Practical notebooks for exploring Earth Observation data with Python.

---

## Contents

| File | Description |
|---|---|
| `visualise_imagery.ipynb` | Quick side-by-side visualisation of VHR, Sentinel-2 and Landsat over Istanbul |
| `eo_lab_1.ipynb` | Guided lab: multi-resolution comparison, spectral indices (NDVI, NDWI), GeoTIFF export |
| `pyproject.toml` | Dependencies for uv (recommended) |
| `environment.yml` | Conda environment file (alternative) |
| `data/` | Local raster data (see structure below) |

### Data layout

```
data/
  vhr/
    istanbul_vhr.tiff          # Very high resolution RGB (~1 m), EPSG:32635
  stac_downloads/
    sentinel2/
      s2_B02.tif               # Blue  (10 m)
      s2_B03.tif               # Green (10 m)
      s2_B04.tif               # Red   (10 m)
      s2_B08.tif               # NIR   (10 m)
    landsat/
      landsat_blue.tif
      landsat_green.tif
      landsat_red.tif
      landsat_nir08.tif        # all bands 30 m
```

All rasters are in **EPSG:32635 (UTM zone 35N)**, covering Istanbul, Turkey.

---

## Setup

### 1. Install Git LFS and clone

Git LFS is required to download the raster data files.

```bash
# Install Git LFS from https://git-lfs.com, then:
git clone https://github.com/mloopa/eolabs.git
cd eolabs/lab_1
```

If you cloned without Git LFS installed the raster files will be small
text stubs (~200 bytes) instead of actual data, causing rasterio to fail
with *"not recognized as being in a supported file format"*.

Check whether the files downloaded correctly:

```bash
# Each .tif should be tens or hundreds of MB -- not a few hundred bytes
ls -lh data/vhr/istanbul_vhr.tiff
```

If the file is small, fetch the real data:

```bash
git lfs pull
```

---

### Option A — uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that
handles the virtual environment and all dependencies in one step.

**Install uv:**

Windows (PowerShell):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Create environment and install dependencies:**

```bash
uv sync
```

**Launch Jupyter:**

```bash
uv run jupyter lab
```

> Make sure you run these commands from inside the `lab_1/` directory
> so that relative paths to `data/` resolve correctly.

---

### Option B — Conda (fallback for Windows if uv does not work)

Install [Miniforge](https://github.com/conda-forge/miniforge/releases/latest), then:

```bash
conda env create -f environment.yml
conda activate eolabs
jupyter lab
```

---

## Lab 1 overview

The notebook walks through:

1. Loading raster files with `rasterio` and inspecting metadata
2. Clipping all sensors to a common geographic extent
3. Visualising true colour composites at native resolution
4. Grayscale conversion and false colour (NIR/R/G)
5. Computing NDVI and comparing it across sensors
6. **Final challenge:** implement NDWI and export results to GeoTIFF for QGIS

Output files are written to `data/outputs/`.
