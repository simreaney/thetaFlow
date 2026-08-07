#!/usr/bin/env python3
"""Spatial utility functions for the thetaFlow 2-D module.

Provides:
* GeoTIFF I/O (DEM, land-cover, soil-type maps) via *rasterio*
* FD8 multiple-flow-direction routing weights (Freeman 1991)
* Darcy–Weisbach overland-flow velocity and flux
* Forcing helpers (uniform CSV or gridded GeoTIFF directory)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional imports – rasterio required for GeoTIFF work
# ---------------------------------------------------------------------------
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import calculate_default_transform, reproject
    _RASTERIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RASTERIO_AVAILABLE = False

_G = 9.81  # gravitational acceleration (m s⁻²)

# 8-neighbour offsets (row_delta, col_delta)
_NEIGHBOURS: list[tuple[int, int]] = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]
# diagonal distance multiplier for FD8 (cardinal = 1.0, diagonal = √2)
_DIST_MULT = np.array([np.sqrt(2), 1.0, np.sqrt(2),
                        1.0,             1.0,
                        np.sqrt(2), 1.0, np.sqrt(2)], dtype=float)


# ---------------------------------------------------------------------------
# Raster I/O helpers
# ---------------------------------------------------------------------------

def _require_rasterio() -> None:
    if not _RASTERIO_AVAILABLE:
        raise ImportError(
            "rasterio is required for spatial GeoTIFF operations. "
            "Install it with: pip install rasterio"
        )


@dataclass
class RasterMeta:
    """Minimal raster metadata needed for spatial calculations."""
    nrows: int
    ncols: int
    cell_size_m: float          # assumes square cells after reprojection
    nodata: Optional[float]
    crs: object                 # rasterio CRS object (or None for synthetic data)
    transform: object           # rasterio Affine transform (or None)


def read_dem(
    dem_path: Path,
    target_cell_size_m: float = 50.0,
) -> tuple[np.ndarray, RasterMeta]:
    """Read a DEM GeoTIFF and resample to *target_cell_size_m* resolution.

    Returns
    -------
    elevation : np.ndarray, shape (nrows, ncols)
        Elevation values in metres (nodata cells set to NaN).
    meta : RasterMeta
    """
    _require_rasterio()
    with rasterio.open(dem_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, src.crs,
            src.width, src.height,
            *src.bounds,
            resolution=target_cell_size_m,
        )
        data = np.full((height, width), np.nan, dtype=float)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=src.crs,
            resampling=Resampling.bilinear,
        )
        nodata = src.nodata
        if nodata is not None:
            data[np.isclose(data, nodata)] = np.nan
        crs = src.crs

    meta = RasterMeta(
        nrows=height,
        ncols=width,
        cell_size_m=target_cell_size_m,
        nodata=nodata,
        crs=crs,
        transform=transform,
    )
    return data, meta


def read_raster_int(
    raster_path: Path,
    reference_meta: RasterMeta,
) -> np.ndarray:
    """Read an integer-coded GeoTIFF (land-cover or soil-type map) and
    resample to match *reference_meta* using nearest-neighbour resampling.

    Returns
    -------
    codes : np.ndarray of int, shape (nrows, ncols)
        Integer class codes; nodata cells → -1.
    """
    _require_rasterio()
    nrows, ncols = reference_meta.nrows, reference_meta.ncols
    data = np.full((nrows, ncols), -1, dtype=np.int32)
    with rasterio.open(raster_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_meta.transform,
            dst_crs=reference_meta.crs,
            resampling=Resampling.nearest,
        )
        nodata = src.nodata
        if nodata is not None:
            data[np.isclose(data.astype(float), float(nodata))] = -1
    return data


# ---------------------------------------------------------------------------
# Vegetation / soil type mapping
# ---------------------------------------------------------------------------

def load_vegetation_map(
    landcover_codes: np.ndarray,
    veg_json_path: Path,
    code_to_name: dict[int, str],
) -> list[list[Optional[object]]]:
    """Map integer land-cover codes to VegetationType objects.

    Parameters
    ----------
    landcover_codes : (nrows, ncols) int array
    veg_json_path : path to vegetation_types.json
    code_to_name : mapping from integer code → vegetation name string
        e.g. {1: "grass", 2: "broadleaf_woodland", ...}

    Returns
    -------
    grid of VegetationType (or None for nodata cells)
    """
    # Import here to avoid circular dependency
    from simulate_soil_column import VegetationType, VEGETATION_LIBRARY

    with veg_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Rebuild library with darcy_weisbach_f
    veg_lib: dict[str, VegetationType] = {}
    for entry in raw["vegetation_types"]:
        vt = VEGETATION_LIBRARY.get(entry["name"])
        if vt is None:
            vt = VegetationType(
                name=entry["name"],
                rooting_depth_m=entry["rooting_depth_m"],
                pet_scale=entry.get("pet_scale", 1.0),
                description=entry.get("description", ""),
            )
        veg_lib[entry["name"]] = vt

    nrows, ncols = landcover_codes.shape
    grid: list[list[Optional[VegetationType]]] = []
    for r in range(nrows):
        row_list: list[Optional[VegetationType]] = []
        for c in range(ncols):
            code = int(landcover_codes[r, c])
            name = code_to_name.get(code)
            row_list.append(veg_lib.get(name) if name else None)
        grid.append(row_list)
    return grid


def load_soil_map(
    soil_codes: np.ndarray,
    soil_types_json_path: Path,
) -> list[list[Optional[object]]]:
    """Map integer soil codes to SoilProperties objects.

    The JSON must have the structure::

        {"soil_types": {"1": {...}, "2": {...}, ...}}

    where each value matches the SoilProperties dataclass fields.

    Returns
    -------
    grid of SoilProperties (or None for nodata cells)
    """
    from simulate_soil_column import SoilProperties

    with soil_types_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    soil_lookup: dict[int, SoilProperties] = {}
    for code_str, props in raw["soil_types"].items():
        sp = SoilProperties(
            theta_r=props["theta_r"],
            theta_s=props["theta_s"],
            alpha_per_m=props["alpha_per_m"],
            n=props["n"],
            ks_m_per_s=props["ks_m_per_s"],
            pore_connectivity=props.get("pore_connectivity", 0.5),
        )
        soil_lookup[int(code_str)] = sp

    nrows, ncols = soil_codes.shape
    grid: list[list[Optional[object]]] = []
    for r in range(nrows):
        row_list: list[Optional[object]] = []
        for c in range(ncols):
            code = int(soil_codes[r, c])
            row_list.append(soil_lookup.get(code))
        grid.append(row_list)
    return grid


# ---------------------------------------------------------------------------
# Darcy–Weisbach friction factor grid
# ---------------------------------------------------------------------------

def build_friction_factor_grid(
    landcover_codes: np.ndarray,
    veg_json_path: Path,
    code_to_name: dict[int, str],
    overrides: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Build a (nrows, ncols) grid of Darcy–Weisbach friction factors.

    Default values come from the ``darcy_weisbach_f`` field in
    *vegetation_types.json*.  Per-name overrides in *overrides* take
    precedence.

    Parameters
    ----------
    overrides : dict mapping vegetation *name* → f value
    """
    with veg_json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Build name → f lookup
    f_lookup: dict[str, float] = {}
    for entry in raw["vegetation_types"]:
        name = entry["name"]
        f_val = float(entry.get("darcy_weisbach_f", 0.5))
        if overrides and name in overrides:
            f_val = float(overrides[name])
        f_lookup[name] = f_val

    default_f = 0.5
    nrows, ncols = landcover_codes.shape
    f_grid = np.full((nrows, ncols), default_f, dtype=float)
    for r in range(nrows):
        for c in range(ncols):
            code = int(landcover_codes[r, c])
            name = code_to_name.get(code)
            if name and name in f_lookup:
                f_grid[r, c] = f_lookup[name]
    return f_grid


# ---------------------------------------------------------------------------
# DEM-derived slope and FD8 routing
# ---------------------------------------------------------------------------

def compute_slope_grid(
    dem: np.ndarray,
    cell_size_m: float,
) -> np.ndarray:
    """Compute per-cell slope magnitude (rise/run) from the DEM using central
    differences (edges use one-sided differences).

    Returns
    -------
    slope : (nrows, ncols) float array — dimensionless rise/run (tan θ)
    """
    dz_dx = np.gradient(dem, cell_size_m, axis=1)
    dz_dy = np.gradient(dem, cell_size_m, axis=0)
    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    return slope


def compute_fd8_weights(
    dem: np.ndarray,
    cell_size_m: float,
    exponent: float = 1.1,
) -> np.ndarray:
    """Compute FD8 multiple-flow-direction weights (Freeman 1991).

    For each cell the slope to each of its 8 neighbours is computed.  Only
    downslope directions (positive slope, i.e. lower neighbours) receive
    weight.  Weights are normalised to sum to 1.0.

    Parameters
    ----------
    dem : (nrows, ncols) elevation array
    cell_size_m : grid spacing in metres (square cells assumed)
    exponent : FD8 exponent *p* (Freeman 1991 recommends 1.1)

    Returns
    -------
    weights : (nrows, ncols, 8) float array
        weights[r, c, k] is the fraction of flow from cell (r, c) directed
        to neighbour k (using the _NEIGHBOURS ordering).
    """
    nrows, ncols = dem.shape
    weights = np.zeros((nrows, ncols, 8), dtype=float)

    for k, (dr, dc) in enumerate(_NEIGHBOURS):
        dist = cell_size_m * _DIST_MULT[k]
        # Vectorised slope from each cell to neighbour k (positive = downhill)
        rr = np.arange(nrows)
        cc = np.arange(ncols)
        rr_n = np.clip(rr + dr, 0, nrows - 1)
        cc_n = np.clip(cc + dc, 0, ncols - 1)

        elev = dem
        elev_n = dem[np.ix_(rr_n, cc_n)]
        slope_k = (elev - elev_n) / dist  # positive = downhill
        # Boundary cells: zero weight toward virtual copies of themselves
        on_edge = np.zeros((nrows, ncols), dtype=bool)
        if dr == -1:
            on_edge[0, :] = True
        elif dr == 1:
            on_edge[-1, :] = True
        if dc == -1:
            on_edge[:, 0] = True
        elif dc == 1:
            on_edge[:, -1] = True
        slope_k = np.where(on_edge, 0.0, slope_k)

        # FD8: only downslope neighbours
        w_k = np.maximum(slope_k, 0.0) ** exponent
        weights[:, :, k] = w_k

    # Normalise
    total = weights.sum(axis=2, keepdims=True)
    total = np.where(total <= 0.0, 1.0, total)  # avoid /0 for flat areas
    weights /= total
    return weights


def fd8_route_flux(
    flux: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Distribute a 2-D flux array to downslope neighbours using FD8 weights.

    Parameters
    ----------
    flux : (nrows, ncols) array — outflow from each cell (m/s or m³/s)
    weights : (nrows, ncols, 8) FD8 weight array from :func:`compute_fd8_weights`

    Returns
    -------
    inflow : (nrows, ncols) array — total inflow received by each cell
    """
    nrows, ncols = flux.shape
    inflow = np.zeros_like(flux)

    for k, (dr, dc) in enumerate(_NEIGHBOURS):
        # Outflow from each cell to neighbour k
        partial = flux * weights[:, :, k]
        # Accumulate at destination cell using np.roll (wrapping handled below)
        dest = np.roll(np.roll(partial, dr, axis=0), dc, axis=1)
        # Zero out contributions that rolled across domain boundaries
        if dr == -1:
            dest[-1, :] = 0.0
        elif dr == 1:
            dest[0, :] = 0.0
        if dc == -1:
            dest[:, -1] = 0.0
        elif dc == 1:
            dest[:, 0] = 0.0
        inflow += dest

    return inflow


def compute_flow_vectors(
    flux: np.ndarray,
    weights: np.ndarray,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (vx, vy) flow-vector components from FD8 weights × flux for
    visualisation purposes.

    Returns
    -------
    vx, vy : (nrows, ncols) arrays — x (east) and y (north) components
    """
    # Column/row offsets for each of the 8 neighbours
    _DC = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=float)
    _DR = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=float)

    vx = np.zeros_like(flux)
    vy = np.zeros_like(flux)
    for k in range(8):
        w = weights[:, :, k] * flux
        vx += w * _DC[k]
        vy += w * (-_DR[k])  # row increases downward → flip for north-up

    return vx, vy


# ---------------------------------------------------------------------------
# Darcy–Weisbach overland flow
# ---------------------------------------------------------------------------

def darcy_weisbach_velocity(
    flow_depth_m: np.ndarray,
    slope: np.ndarray,
    f_grid: np.ndarray,
) -> np.ndarray:
    """Compute overland flow velocity using Darcy–Weisbach.

    v = sqrt(8 g R S / f)

    where hydraulic radius R = flow depth h (thin sheet-flow assumption),
    S = slope (dimensionless), f = friction factor.

    Returns
    -------
    velocity : (nrows, ncols) array in m/s (zero where depth or slope ≤ 0)
    """
    h = np.maximum(flow_depth_m, 0.0)
    S = np.maximum(slope, 1.0e-6)
    f = np.maximum(f_grid, 1.0e-6)
    v = np.sqrt(8.0 * _G * h * S / f)
    v = np.where(h <= 0.0, 0.0, v)
    return v


def update_flow_depth(
    flow_depth_m: np.ndarray,
    runoff_flux: np.ndarray,
    velocity: np.ndarray,
    fd8_weights: np.ndarray,
    cell_size_m: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance overland flow depth by one timestep using a simple explicit
    continuity scheme with FD8 redistribution.

    Outflow from each cell = v * h * cell_width * dt
    Inflow = FD8-weighted contributions from upslope cells.

    Parameters
    ----------
    flow_depth_m : current flow depth (m)
    runoff_flux : surface runoff produced this timestep (m/s)
    velocity : Darcy–Weisbach velocity (m/s)
    fd8_weights : (nrows, ncols, 8)
    cell_size_m : grid spacing (m)
    dt_s : timestep (s)

    Returns
    -------
    new_depth, vx, vy : updated depth array and flow vectors
    """
    h = np.maximum(flow_depth_m, 0.0)

    # Volume of water leaving each cell per second per unit area (m/s)
    outflow_rate = velocity * h / cell_size_m  # m/s normalised by cell width

    # Inflow from upslope cells
    inflow_rate = fd8_route_flux(outflow_rate, fd8_weights)

    # Continuity: dh/dt = runoff_flux + inflow - outflow
    h_new = h + dt_s * (runoff_flux + inflow_rate - outflow_rate)
    h_new = np.maximum(h_new, 0.0)

    vx, vy = compute_flow_vectors(outflow_rate, fd8_weights, cell_size_m)
    return h_new, vx, vy


# ---------------------------------------------------------------------------
# Forcing helpers
# ---------------------------------------------------------------------------

def read_uniform_forcing(
    csv_path: Path,
    default_dt_hours: float = 1.0,
) -> pd.DataFrame:
    """Read a uniform (single-column) forcing CSV identical to the 1-D format.

    The DataFrame is processed with the same logic as
    ``simulate_soil_column._process_forcing_df``.
    """
    from simulate_soil_column import _process_forcing_df

    df = pd.read_csv(csv_path)
    return _process_forcing_df(df, default_dt_hours, source_name=csv_path.name)


def _parse_forcing_tif_name(filename: str) -> tuple[Optional[str], Optional[int]]:
    """Parse a gridded forcing filename.

    Expected patterns::

        rainfall_00001.tif   → ("rainfall", 1)
        pet_00001.tif        → ("pet", 1)
    """
    m = re.match(r"^(rainfall|pet)_(\d+)\.tif$", filename, re.IGNORECASE)
    if m:
        return m.group(1).lower(), int(m.group(2))
    return None, None


def read_gridded_forcing(
    forcing_dir: Path,
    reference_meta: RasterMeta,
    default_dt_hours: float = 1.0,
) -> list[dict[str, object]]:
    """Read spatially gridded rainfall and PET from a directory of GeoTIFFs.

    Files must be named ``rainfall_NNNNN.tif`` and ``pet_NNNNN.tif`` where
    NNNNN is a zero-padded step index (same index = same timestep).

    Returns
    -------
    list of dicts, one per timestep, each with keys:
        ``step``, ``dt_seconds``, ``rainfall_grid`` (ndarray, mm/h),
        ``pet_grid`` (ndarray, mm/h)
    """
    _require_rasterio()
    forcing_dir = Path(forcing_dir)
    rainfall_files: dict[int, Path] = {}
    pet_files: dict[int, Path] = {}

    for p in sorted(forcing_dir.glob("*.tif")):
        kind, step = _parse_forcing_tif_name(p.name)
        if kind == "rainfall" and step is not None:
            rainfall_files[step] = p
        elif kind == "pet" and step is not None:
            pet_files[step] = p

    steps = sorted(set(rainfall_files) & set(pet_files))
    if not steps:
        raise ValueError(
            f"No matching rainfall_NNNNN.tif / pet_NNNNN.tif pairs found in {forcing_dir}"
        )

    dt_s = default_dt_hours * 3600.0
    records: list[dict] = []
    for step in steps:
        rain_grid = _read_forcing_grid(rainfall_files[step], reference_meta)
        pet_grid = _read_forcing_grid(pet_files[step], reference_meta)
        records.append(
            {
                "step": step,
                "dt_seconds": dt_s,
                "rainfall_grid": rain_grid,
                "pet_grid": pet_grid,
            }
        )
    return records


def _read_forcing_grid(tif_path: Path, reference_meta: RasterMeta) -> np.ndarray:
    _require_rasterio()
    nrows, ncols = reference_meta.nrows, reference_meta.ncols
    data = np.zeros((nrows, ncols), dtype=float)
    with rasterio.open(tif_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_meta.transform,
            dst_crs=reference_meta.crs,
            resampling=Resampling.bilinear,
        )
        nodata = src.nodata
        if nodata is not None:
            data[np.isclose(data, nodata)] = 0.0
    return np.maximum(data, 0.0)
