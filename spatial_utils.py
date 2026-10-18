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
import warnings
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

def _vegetation_entries(veg_types: "Path | dict") -> list[dict]:
    """Return the ``vegetation_types`` entry list from a JSON path or an in-memory dict."""
    if isinstance(veg_types, dict):
        raw = veg_types
    else:
        with Path(veg_types).open("r", encoding="utf-8") as f:
            raw = json.load(f)
    return list(raw["vegetation_types"])


def load_vegetation_types(veg_types: "Path | dict") -> dict[str, object]:
    """Build a ``{name: VegetationType}`` lookup from a JSON path or dict.

    Entries in the JSON take precedence; the built-in ``VEGETATION_LIBRARY``
    only fills in fields that an entry leaves out.
    """
    from simulate_soil_column import VegetationType, VEGETATION_LIBRARY

    veg_lib: dict[str, VegetationType] = {}
    for entry in _vegetation_entries(veg_types):
        base = VEGETATION_LIBRARY.get(entry["name"])
        veg_lib[entry["name"]] = VegetationType(
            name=entry["name"],
            rooting_depth_m=float(
                entry.get("rooting_depth_m", base.rooting_depth_m if base else 0.0)
            ),
            pet_scale=float(entry.get("pet_scale", base.pet_scale if base else 1.0)),
            description=entry.get("description", base.description if base else ""),
        )
    return veg_lib


def load_code_map(path: Path) -> dict[int, str]:
    """Read a land-cover ``{"code": "vegetation name"}`` JSON map.

    Keys starting with ``_`` (e.g. ``"_comment"``) are ignored.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items() if not str(k).startswith("_")}


def load_vegetation_map(
    landcover_codes: np.ndarray,
    veg_json_path: "Path | dict",
    code_to_name: dict[int, str],
    default_name: Optional[str] = None,
) -> list[list[Optional[object]]]:
    """Map integer land-cover codes to VegetationType objects.

    Parameters
    ----------
    landcover_codes : (nrows, ncols) int array
    veg_json_path : path to vegetation_types.json, or an equivalent dict
        ``{"vegetation_types": [...]}``
    code_to_name : mapping from integer code → vegetation name string
        e.g. {1: "grass", 2: "broadleaf_woodland", ...}
    default_name : vegetation name used for nodata cells and codes missing
        from *code_to_name* (``None`` → no vegetation for those cells)

    Returns
    -------
    grid of VegetationType (or None for unmapped cells without a default)
    """
    veg_lib = load_vegetation_types(veg_json_path)
    default_veg = veg_lib.get(default_name) if default_name else None

    nrows, ncols = landcover_codes.shape
    grid: list[list[Optional[object]]] = []
    for r in range(nrows):
        row_list: list[Optional[object]] = []
        for c in range(ncols):
            code = int(landcover_codes[r, c])
            name = code_to_name.get(code)
            veg = veg_lib.get(name) if name else None
            row_list.append(veg if veg is not None else default_veg)
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

def build_vegetation_property_grid(
    landcover_codes: np.ndarray,
    veg_json_path: "Path | dict",
    code_to_name: dict[int, str],
    field: str,
    fallback: float,
    overrides: Optional[dict[str, float]] = None,
    default_name: Optional[str] = None,
) -> np.ndarray:
    """Build a (nrows, ncols) grid of a numeric vegetation-type property.

    Values come from *field* in *vegetation_types.json* (``fallback`` when a
    type does not define it).  Per-name overrides in *overrides* take
    precedence.

    Parameters
    ----------
    overrides : dict mapping vegetation *name* → value
    default_name : vegetation name used for unmapped / nodata codes
    """
    lookup: dict[str, float] = {}
    for entry in _vegetation_entries(veg_json_path):
        name = entry["name"]
        val = float(entry.get(field, fallback))
        if overrides and name in overrides:
            val = float(overrides[name])
        lookup[name] = val

    default_val = lookup.get(default_name, fallback) if default_name else fallback
    nrows, ncols = landcover_codes.shape
    grid = np.full((nrows, ncols), default_val, dtype=float)
    for r in range(nrows):
        for c in range(ncols):
            code = int(landcover_codes[r, c])
            name = code_to_name.get(code)
            if name and name in lookup:
                grid[r, c] = lookup[name]
    return grid


def build_friction_factor_grid(
    landcover_codes: np.ndarray,
    veg_json_path: "Path | dict",
    code_to_name: dict[int, str],
    overrides: Optional[dict[str, float]] = None,
    default_name: Optional[str] = None,
) -> np.ndarray:
    """Build a (nrows, ncols) grid of Darcy–Weisbach friction factors.

    Default values come from the ``darcy_weisbach_f`` field in
    *vegetation_types.json*.  Per-name overrides in *overrides* take
    precedence.
    """
    return build_vegetation_property_grid(
        landcover_codes, veg_json_path, code_to_name, "darcy_weisbach_f", 0.5,
        overrides=overrides, default_name=default_name,
    )


def build_depression_storage_grid(
    landcover_codes: np.ndarray,
    veg_json_path: "Path | dict",
    code_to_name: dict[int, str],
    fallback_m: float = 0.0,
    overrides: Optional[dict[str, float]] = None,
    default_name: Optional[str] = None,
) -> np.ndarray:
    """Build a (nrows, ncols) grid of microtopographic depression storage (m).

    Values come from the ``depression_storage_m`` field in
    *vegetation_types.json* (``fallback_m`` where a type does not define it).
    Surface water shallower than this threshold is held in micro-depressions
    and cannot run off; see :func:`update_flow_depth`.
    """
    return build_vegetation_property_grid(
        landcover_codes, veg_json_path, code_to_name, "depression_storage_m", fallback_m,
        overrides=overrides, default_name=default_name,
    )


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


def _neighbour_gradients(surface: np.ndarray, cell_size_m: float) -> np.ndarray:
    """Gradient (drop / distance, positive = downhill) from each cell to each
    of its 8 neighbours, as a (nrows, ncols, 8) array.  Directions that leave
    the domain get zero."""
    nrows, ncols = surface.shape
    grads = np.zeros((nrows, ncols, 8), dtype=float)
    rr = np.arange(nrows)
    cc = np.arange(ncols)
    for k, (dr, dc) in enumerate(_NEIGHBOURS):
        dist = cell_size_m * _DIST_MULT[k]
        rr_n = np.clip(rr + dr, 0, nrows - 1)
        cc_n = np.clip(cc + dc, 0, ncols - 1)
        g = (surface - surface[np.ix_(rr_n, cc_n)]) / dist
        # Boundary cells: zero weight toward virtual copies of themselves
        if dr == -1:
            g[0, :] = 0.0
        elif dr == 1:
            g[-1, :] = 0.0
        if dc == -1:
            g[:, 0] = 0.0
        elif dc == 1:
            g[:, -1] = 0.0
        grads[:, :, k] = g
    return grads


def _fd8_weights_from_gradients(grads: np.ndarray, exponent: float) -> np.ndarray:
    weights = np.maximum(grads, 0.0) ** exponent
    total = weights.sum(axis=2, keepdims=True)
    total = np.where(total <= 0.0, 1.0, total)  # avoid /0 for flat areas
    return weights / total


def compute_fd8_weights(
    dem: np.ndarray,
    cell_size_m: float,
    exponent: float = 1.1,
) -> np.ndarray:
    """Compute FD8 multiple-flow-direction weights (Freeman 1991).

    For each cell the slope to each of its 8 neighbours is computed.  Only
    downslope directions (positive slope, i.e. lower neighbours) receive
    weight.  Weights are normalised to sum to 1.0; cells with no lower
    neighbour (pits, flats, outlets on the domain edge) get all-zero weights.

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
    return _fd8_weights_from_gradients(_neighbour_gradients(dem, cell_size_m), exponent)


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


def estimate_stable_overland_dt_seconds(
    velocity: np.ndarray,
    cell_size_m: float,
    max_dt_s: float,
    cfl_safety: float = 0.5,
    min_dt_s: float = 0.1,
) -> float:
    """Estimate a CFL-stable timestep for the explicit FD8 overland-flow update.

    The explicit continuity scheme in :func:`update_flow_depth` is only stable
    when a cell cannot drain more water than it holds within one substep, i.e.
    ``velocity * dt / cell_size_m <= 1``.  Returns ``max_dt_s`` unchanged when
    there is no flow (velocity is zero everywhere).
    """
    v_max = float(np.max(velocity)) if velocity.size else 0.0
    if v_max <= 0.0:
        return max_dt_s
    dt_cfl = cfl_safety * cell_size_m / v_max
    return float(np.clip(dt_cfl, min_dt_s, max_dt_s))


def update_flow_depth(
    flow_depth_m: np.ndarray,
    runoff_flux: np.ndarray,
    slope_grid: np.ndarray,
    f_grid: np.ndarray,
    fd8_weights: np.ndarray,
    cell_size_m: float,
    dt_s: float,
    cfl_safety: float = 0.5,
    min_substep_seconds: float = 0.1,
    dem: Optional[np.ndarray] = None,
    depression_storage_m: "float | np.ndarray" = 0.0,
    active_mask: Optional[np.ndarray] = None,
    fd8_exponent: float = 1.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance overland flow depth over ``dt_s`` using a simple explicit
    continuity scheme with FD8 redistribution, subdivided into CFL-stable
    substeps.

    Outflow from each cell = v * h_mobile / cell_width (per substep).
    Inflow = FD8-weighted contributions from upslope cells.

    Without substepping, the explicit update is unconditionally unstable for
    the timesteps typically used here (e.g. hourly forcing over 50 m cells):
    depth blows up over a handful of steps and then collapses to exactly zero
    once the resulting negative overshoot is clamped. Velocity is recomputed
    every substep since it depends on the evolving depth.

    Microtopography is represented as a depression-storage threshold: only
    the depth above ``depression_storage_m`` is mobile and can leave a cell,
    the rest is held in micro-depressions (and is available to infiltrate).

    Interior cells with no lower neighbour (pits) store water instead of
    draining it.  When *dem* is given, flow directions and slopes are
    recomputed every substep from the water surface ``dem + depth``, so pits
    fill and then spill.  Edge cells with no lower neighbour are outlets and
    drain out of the domain at the DEM slope.  Water reaching cells outside
    *active_mask* leaves the domain.

    Parameters
    ----------
    flow_depth_m : current flow depth (m)
    runoff_flux : surface runoff produced this timestep (m/s)
    slope_grid : DEM-derived slope magnitude (rise/run)
    f_grid : Darcy–Weisbach friction factor grid
    fd8_weights : (nrows, ncols, 8); used only when *dem* is None
    cell_size_m : grid spacing (m)
    dt_s : total timestep to advance (s)
    dem : optional elevation grid for water-surface routing
    depression_storage_m : scalar or (nrows, ncols) microtopographic
        depression storage (m) that must fill before runoff can leave a cell
    active_mask : optional (nrows, ncols) bool array of simulated cells
    fd8_exponent : FD8 exponent for water-surface routing

    Returns
    -------
    new_depth, vx, vy : updated depth array and time-averaged flow vectors
    """
    h = np.maximum(flow_depth_m, 0.0)
    nrows, ncols = h.shape
    storage = np.broadcast_to(np.maximum(depression_storage_m, 0.0), h.shape)
    inactive = None if active_mask is None else ~np.asarray(active_mask, dtype=bool)
    if inactive is not None:
        h = np.where(inactive, 0.0, h)

    boundary = np.zeros((nrows, ncols), dtype=bool)
    boundary[0, :] = boundary[-1, :] = True
    boundary[:, 0] = boundary[:, -1] = True

    dist = cell_size_m * _DIST_MULT
    vx_accum = np.zeros_like(h)
    vy_accum = np.zeros_like(h)

    remaining_s = dt_s
    n_iter = 0
    while remaining_s > 1.0e-9:
        n_iter += 1
        if n_iter > 100_000:
            warnings.warn("update_flow_depth: exceeded max substeps, truncating.")
            break

        if dem is not None:
            grads = _neighbour_gradients(dem + h, cell_size_m)
            weights = _fd8_weights_from_gradients(grads, fd8_exponent)
            # Slope along the routed flow directions, and the drop to the
            # lowest neighbour (limits outflow so the water surface cannot
            # overshoot below its receivers within a substep)
            route_slope = np.sum(weights * grads, axis=2)
            max_drop = np.max(np.maximum(grads, 0.0) * dist, axis=2)
        else:
            weights = fd8_weights
            route_slope = slope_grid
            max_drop = None

        has_receiver = weights.sum(axis=2) > 0.0
        outlet = boundary & ~has_receiver
        slope = np.where(outlet, slope_grid, route_slope)

        mobile = np.maximum(h - storage, 0.0)
        mobile = np.where(has_receiver | outlet, mobile, 0.0)
        velocity = darcy_weisbach_velocity(mobile, slope, f_grid)
        sub_dt = estimate_stable_overland_dt_seconds(
            velocity, cell_size_m, remaining_s, cfl_safety, min_substep_seconds
        )
        # Also respect CFL at the depth reached by the end of the substep, so
        # a dry (zero-velocity) surface is not advanced over the whole step
        # in one go while runoff is being added
        for _ in range(2):
            mobile_end = np.maximum(h + runoff_flux * sub_dt - storage, 0.0)
            mobile_end = np.where(has_receiver | outlet, mobile_end, 0.0)
            sub_dt = estimate_stable_overland_dt_seconds(
                darcy_weisbach_velocity(mobile_end, slope, f_grid),
                cell_size_m, sub_dt, cfl_safety, min_substep_seconds,
            )

        # Depth of water leaving each cell this substep (m), never more than
        # the mobile depth
        out_depth = np.minimum(velocity * mobile / cell_size_m * sub_dt, mobile)
        if max_drop is not None:
            out_depth = np.where(has_receiver, np.minimum(out_depth, 0.5 * max_drop), out_depth)
        outflow_rate = out_depth / sub_dt

        # Inflow from upslope cells (outlet outflow has no receiver and leaves)
        inflow_rate = fd8_route_flux(outflow_rate, weights)

        # Continuity: dh/dt = runoff_flux + inflow - outflow
        h = h + sub_dt * (runoff_flux + inflow_rate - outflow_rate)
        h = np.maximum(h, 0.0)
        if inactive is not None:
            h = np.where(inactive, 0.0, h)

        vx, vy = compute_flow_vectors(outflow_rate, weights, cell_size_m)
        vx_accum += vx * sub_dt
        vy_accum += vy * sub_dt
        remaining_s -= sub_dt

    vx_accum /= dt_s
    vy_accum /= dt_s
    return h, vx_accum, vy_accum


def fill_nodata_nearest(grid: np.ndarray, valid: np.ndarray, max_iter: int = 10000) -> np.ndarray:
    """Fill invalid cells with the mean of their valid 8-neighbours, growing
    inwards from the valid region until every cell has a value.

    Used to give NoData DEM cells a neutral elevation so that they neither
    create artificial cliffs nor act as spurious sources in FD8 routing.
    """
    out = np.where(valid, grid, 0.0).astype(float)
    filled = valid.copy()
    if not filled.any():
        raise ValueError("Grid has no valid cells to fill from.")
    nrows, ncols = out.shape
    for _ in range(max_iter):
        if filled.all():
            break
        pad_v = np.pad(np.where(filled, out, 0.0), 1)
        pad_n = np.pad(filled.astype(float), 1)
        total = np.zeros_like(out)
        count = np.zeros_like(out)
        for dr, dc in _NEIGHBOURS:
            total += pad_v[1 + dr:1 + dr + nrows, 1 + dc:1 + dc + ncols]
            count += pad_n[1 + dr:1 + dr + nrows, 1 + dc:1 + dc + ncols]
        grow = (~filled) & (count > 0)
        out[grow] = total[grow] / count[grow]
        filled = filled | grow
    return out


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
