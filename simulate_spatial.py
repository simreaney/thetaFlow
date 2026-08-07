#!/usr/bin/env python3
"""thetaFlow 2-D spatial simulation.

Each grid cell contains an independent 1-D Richards-equation soil column
(from simulate_soil_column.py).  Surface runoff and subsurface lateral
throughflow are routed between cells using FD8 multiple-flow-direction
routing.  Overland-flow velocity is computed with the Darcy–Weisbach
equation.

Usage
-----
python simulate_spatial.py \\
    --dem dem.tif \\
    --landcover landcover.tif \\
    --soilmap soilmap.tif \\
    --soil-types soil_types.json \\
    --veg-types vegetation_types.json \\
    --lc-map landcover_code_map.json \\
    --forcing forcing.csv \\
    --config spatial_config_example.json \\
    --output-dir outputs_2d/ \\
    --workers 4

For gridded forcing use ``--forcing-dir`` instead of ``--forcing``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports from the 1-D module (must be on PYTHONPATH / same directory)
# ---------------------------------------------------------------------------
from simulate_soil_column import (
    ColumnConfig,
    SimulationConfig,
    SimulationState,
    SoilProperties,
    VegetationType,
    VEGETATION_LIBRARY,
    build_column_geometry,
    head_from_theta,
    one_substep,
    estimate_stable_dt_seconds,
    theta_from_head,
    _process_forcing_df,
)

from spatial_utils import (
    RasterMeta,
    compute_slope_grid,
    compute_fd8_weights,
    fd8_route_flux,
    update_flow_depth,
    darcy_weisbach_velocity,
    build_friction_factor_grid,
    read_uniform_forcing,
)

# ---------------------------------------------------------------------------
# Optional GPU support via CuPy
# ---------------------------------------------------------------------------
try:
    import cupy as cp  # type: ignore
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpatialConfig:
    """Configuration for a spatial simulation run."""

    cell_size_m: float = 50.0
    column: ColumnConfig = field(default_factory=lambda: ColumnConfig(nz=50, dz_m=0.02))
    simulation: SimulationConfig = field(
        default_factory=lambda: SimulationConfig(
            initial_head_m=-1.0,
            default_dt_hours=1.0,
            max_substep_seconds=60.0,
            min_theta_buffer=0.002,
        )
    )
    fd8_exponent: float = 1.1
    darcy_weisbach_f_overrides: dict[str, float] = field(default_factory=dict)
    save_npy_arrays: bool = True
    snapshot_interval_steps: int = 0
    animation_fps: int = 4
    quiver_stride: int = 2


def load_spatial_config(config_path: Path) -> SpatialConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    col = raw.get("column", {})
    sim_raw = raw.get("simulation", {})
    out = raw.get("output", {})

    column = ColumnConfig(
        nz=int(col.get("nz", 50)),
        dz_m=float(col.get("dz_m", 0.02)),
    )
    simulation = SimulationConfig(
        initial_head_m=float(sim_raw.get("initial_head_m", -1.0)),
        default_dt_hours=float(sim_raw.get("default_dt_hours", 1.0)),
        max_substep_seconds=float(sim_raw.get("max_substep_seconds", 60.0)),
        min_theta_buffer=float(sim_raw.get("min_theta_buffer", 0.002)),
        max_dtheta_per_substep=float(sim_raw.get("max_dtheta_per_substep", 0.002)),
        min_substep_seconds=float(sim_raw.get("min_substep_seconds", 1.0)),
        green_ampt_wetting_front_suction_m=float(
            sim_raw.get("green_ampt_wetting_front_suction_m", 0.11)
        ),
        slope_angle_deg=float(sim_raw.get("slope_angle_deg", 1.0)),
        hillslope_flow_path_m=float(
            sim_raw.get("hillslope_flow_path_m", 50.0)
        ),
        latitude=sim_raw.get("latitude"),
        longitude=sim_raw.get("longitude"),
    )

    return SpatialConfig(
        cell_size_m=float(raw.get("cell_size_m", 50.0)),
        column=column,
        simulation=simulation,
        fd8_exponent=float(raw.get("fd8_exponent", 1.1)),
        darcy_weisbach_f_overrides=raw.get("darcy_weisbach_f_overrides", {}),
        save_npy_arrays=bool(out.get("save_npy_arrays", True)),
        snapshot_interval_steps=int(out.get("snapshot_interval_steps", 1)),
        animation_fps=int(out.get("animation_fps", 4)),
        quiver_stride=int(out.get("quiver_stride", 2)),
    )


# ---------------------------------------------------------------------------
# Per-cell worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _run_cell(
    cell_idx: int,
    state_theta: np.ndarray,
    state_head: np.ndarray,
    soil_props: tuple,
    sim_cfg_dict: dict,
    col_nz: int,
    col_dz: float,
    rainfall_mm_h: float,
    pet_mm_h: float,
    duration_s: float,
    veg_dict: Optional[dict],
    lateral_inflow_m_per_s: float,
    cumulative_infiltration_m_in: float,
) -> tuple[int, np.ndarray, np.ndarray, float, float, float]:
    """Process one soil column for one forcing timestep.

    All arguments are plain Python/numpy objects to be picklable for
    subprocess transport.

    Returns
    -------
    cell_idx, new_theta, new_head, runoff_m_per_s, lateral_out_m_per_s,
    cumulative_infiltration_m
    """
    # Reconstruct objects from plain dicts / tuples
    soil = SoilProperties(*soil_props)
    sim = SimulationConfig(**sim_cfg_dict)

    veg: Optional[VegetationType] = None
    if veg_dict is not None:
        veg = VegetationType(
            name=veg_dict["name"],
            rooting_depth_m=veg_dict["rooting_depth_m"],
            pet_scale=veg_dict["pet_scale"],
            description=veg_dict.get("description", ""),
        )

    state = SimulationState(head_m=state_head.copy(), theta=state_theta.copy())
    _, depth_m = build_column_geometry(col_nz, col_dz)

    cumulative_infiltration_m = max(cumulative_infiltration_m_in, 1.0e-6)
    remaining_s = duration_s
    n_iter = 0
    last_diag: Optional[dict] = None

    # Adjust rainfall for any lateral inflow received (add as equivalent rainfall rate)
    eff_rainfall = rainfall_mm_h + lateral_inflow_m_per_s * 1000.0 * 3600.0

    while remaining_s > 1.0e-9:
        n_iter += 1
        if n_iter > 500_000:
            warnings.warn(f"Cell {cell_idx}: exceeded max substeps, truncating.")
            break

        candidate_max_dt = min(sim.max_substep_seconds, remaining_s)
        stable_dt = estimate_stable_dt_seconds(
            state, soil, sim, cumulative_infiltration_m, col_dz,
            eff_rainfall, pet_mm_h,
            sim.min_theta_buffer, sim.max_dtheta_per_substep,
            candidate_max_dt, sim.min_substep_seconds,
            depth_m=depth_m, vegetation=veg,
        )
        dt_s = min(candidate_max_dt, stable_dt, remaining_s)

        state, diag, cumulative_infiltration_m = one_substep(
            state, soil, sim, cumulative_infiltration_m, col_dz, dt_s,
            eff_rainfall, pet_mm_h,
            sim.min_theta_buffer,
            depth_m=depth_m, vegetation=veg,
        )
        last_diag = diag
        remaining_s -= dt_s

    runoff = last_diag["runoff_flux_m_per_s"] if last_diag else 0.0
    lateral_out = last_diag["lateral_throughflow_flux_m_per_s"] if last_diag else 0.0

    return cell_idx, state.theta, state.head_m, runoff, lateral_out, cumulative_infiltration_m


# ---------------------------------------------------------------------------
# Synthetic test-data generator
# ---------------------------------------------------------------------------

def _generate_synthetic_dem(nrows: int = 20, ncols: int = 20, cell_size: float = 50.0) -> np.ndarray:
    """Generate a simple synthetic hillslope DEM for testing."""
    x = np.linspace(0, 1, ncols)
    y = np.linspace(0, 1, nrows)
    XX, YY = np.meshgrid(x, y)
    dem = 100.0 + 50.0 * (1.0 - XX) + 10.0 * np.sin(np.pi * YY) + \
          np.random.default_rng(42).normal(0, 0.5, (nrows, ncols))
    return dem


# ---------------------------------------------------------------------------
# Main simulation driver
# ---------------------------------------------------------------------------

def run_spatial_simulation(
    dem: np.ndarray,
    landcover_codes: Optional[np.ndarray],
    soil_codes: Optional[np.ndarray],
    veg_json_path: Path,
    soil_types_json_path: Optional[Path],
    code_to_veg_name: dict[int, str],
    forcing_uniform: Optional[pd.DataFrame],
    forcing_gridded: Optional[list[dict]],
    cfg: SpatialConfig,
    output_dir: Path,
    n_workers: int = 1,
    use_gpu: bool = False,
) -> None:
    """Run the 2-D spatial simulation.

    Parameters
    ----------
    dem : (nrows, ncols) elevation array in metres
    landcover_codes : (nrows, ncols) integer land-cover codes (or None)
    soil_codes : (nrows, ncols) integer soil-type codes (or None)
    veg_json_path : path to vegetation_types.json
    soil_types_json_path : path to soil_types.json (or None → uniform soil)
    code_to_veg_name : {integer code → vegetation name string}
    forcing_uniform : uniform DataFrame (None if using gridded forcing)
    forcing_gridded : list of per-step dicts (None if using uniform forcing)
    cfg : SpatialConfig
    output_dir : directory for outputs
    n_workers : number of ProcessPoolExecutor workers
    use_gpu : attempt to use CuPy GPU arrays (falls back to CPU)
    """
    if use_gpu and not _CUPY_AVAILABLE:
        print("Warning: --gpu specified but CuPy not found. Falling back to CPU.")
        use_gpu = False

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nrows, ncols = dem.shape
    ncells = nrows * ncols
    cell_size = cfg.cell_size_m

    print(f"Grid: {nrows} × {ncols} = {ncells} cells  |  cell size: {cell_size} m")

    # -----------------------------------------------------------------------
    # Derived terrain fields
    # -----------------------------------------------------------------------
    slope_grid = compute_slope_grid(dem, cell_size)
    slope_grid = np.maximum(slope_grid, 1.0e-4)  # avoid perfectly flat cells
    fd8_weights = compute_fd8_weights(dem, cell_size, exponent=cfg.fd8_exponent)

    # -----------------------------------------------------------------------
    # Load soil properties per cell
    # -----------------------------------------------------------------------
    if soil_types_json_path is not None and soil_codes is not None:
        from spatial_utils import load_soil_map
        soil_grid = load_soil_map(soil_codes, soil_types_json_path)
        # Flatten to 1-D list
        soil_list: list[Optional[SoilProperties]] = [
            soil_grid[r][c] for r in range(nrows) for c in range(ncols)
        ]
    else:
        # Default: use a loam
        default_soil = SoilProperties(
            theta_r=0.078, theta_s=0.43, alpha_per_m=3.6, n=1.56,
            ks_m_per_s=2.89e-6, pore_connectivity=0.5,
        )
        soil_list = [default_soil] * ncells

    # -----------------------------------------------------------------------
    # Load vegetation per cell
    # -----------------------------------------------------------------------
    if landcover_codes is not None:
        from spatial_utils import load_vegetation_map
        veg_grid = load_vegetation_map(landcover_codes, veg_json_path, code_to_veg_name)
        veg_list: list[Optional[VegetationType]] = [
            veg_grid[r][c] for r in range(nrows) for c in range(ncols)
        ]
    else:
        default_veg = VEGETATION_LIBRARY.get("grass")
        veg_list = [default_veg] * ncells

    # -----------------------------------------------------------------------
    # Friction factor grid
    # -----------------------------------------------------------------------
    if landcover_codes is not None:
        f_grid = build_friction_factor_grid(
            landcover_codes, veg_json_path, code_to_veg_name,
            overrides=cfg.darcy_weisbach_f_overrides if cfg.darcy_weisbach_f_overrides else None,
        )
    else:
        f_grid = np.full((nrows, ncols), 0.5)

    # -----------------------------------------------------------------------
    # Initialise state arrays
    # -----------------------------------------------------------------------
    col = cfg.column
    base_sim = cfg.simulation
    _, depth_m = build_column_geometry(col.nz, col.dz_m)

    # theta_grid and head_grid: shape (nrows*ncols, nz)
    theta_grid = np.zeros((ncells, col.nz), dtype=float)
    head_grid = np.zeros((ncells, col.nz), dtype=float)
    for idx in range(ncells):
        soil = soil_list[idx] or soil_list[0]
        h0 = np.full(col.nz, base_sim.initial_head_m)
        t0 = theta_from_head(h0, soil)
        theta_grid[idx] = t0
        head_grid[idx] = h0

    flow_depth = np.zeros((nrows, ncols), dtype=float)

    # -----------------------------------------------------------------------
    # Per-cell SimulationConfig (slope from DEM)
    # -----------------------------------------------------------------------
    def _make_sim_cfg(r: int, c: int) -> SimulationConfig:
        slope_angle_deg = float(np.degrees(np.arctan(slope_grid[r, c])))
        return SimulationConfig(
            initial_head_m=base_sim.initial_head_m,
            default_dt_hours=base_sim.default_dt_hours,
            max_substep_seconds=base_sim.max_substep_seconds,
            min_theta_buffer=base_sim.min_theta_buffer,
            max_dtheta_per_substep=base_sim.max_dtheta_per_substep,
            min_substep_seconds=base_sim.min_substep_seconds,
            green_ampt_wetting_front_suction_m=base_sim.green_ampt_wetting_front_suction_m,
            slope_angle_deg=slope_angle_deg,
            hillslope_flow_path_m=cell_size,
            latitude=base_sim.latitude,
            longitude=base_sim.longitude,
        )

    sim_configs = [_make_sim_cfg(r, c) for r in range(nrows) for c in range(ncols)]

    # -----------------------------------------------------------------------
    # Forcing iterator
    # -----------------------------------------------------------------------
    def _forcing_steps():
        if forcing_uniform is not None:
            for _, row in forcing_uniform.iterrows():
                yield (
                    float(row["dt_seconds"]),
                    np.full((nrows, ncols), float(row["rainfall_mm_h"])),
                    np.full((nrows, ncols), float(row["pet_mm_h"])),
                )
        else:
            for rec in forcing_gridded:  # type: ignore[union-attr]
                yield (
                    float(rec["dt_seconds"]),
                    rec["rainfall_grid"].reshape(nrows, ncols),
                    rec["pet_grid"].reshape(nrows, ncols),
                )

    # -----------------------------------------------------------------------
    # Helper to serialise objects for subprocess
    # -----------------------------------------------------------------------
    def _soil_tuple(sp: SoilProperties) -> tuple:
        return (sp.theta_r, sp.theta_s, sp.alpha_per_m, sp.n,
                sp.ks_m_per_s, sp.pore_connectivity)

    def _sim_dict(sc: SimulationConfig) -> dict:
        return {
            "initial_head_m": sc.initial_head_m,
            "default_dt_hours": sc.default_dt_hours,
            "max_substep_seconds": sc.max_substep_seconds,
            "min_theta_buffer": sc.min_theta_buffer,
            "max_dtheta_per_substep": sc.max_dtheta_per_substep,
            "min_substep_seconds": sc.min_substep_seconds,
            "green_ampt_wetting_front_suction_m": sc.green_ampt_wetting_front_suction_m,
            "slope_angle_deg": sc.slope_angle_deg,
            "hillslope_flow_path_m": sc.hillslope_flow_path_m,
            "latitude": sc.latitude,
            "longitude": sc.longitude,
        }

    def _veg_dict(v: Optional[VegetationType]) -> Optional[dict]:
        if v is None:
            return None
        return {
            "name": v.name,
            "rooting_depth_m": v.rooting_depth_m,
            "pet_scale": v.pet_scale,
            "description": v.description,
        }

    # -----------------------------------------------------------------------
    # Main timestep loop
    # -----------------------------------------------------------------------
    diag_rows: list[dict] = []
    step = 0
    t_elapsed = 0.0

    # Per-cell cumulative infiltration carried across timesteps (Green-Ampt)
    cum_infilt_grid = np.full(ncells, 1.0e-6, dtype=float)
    lateral_out_grid = np.zeros((nrows, ncols))

    print("Starting simulation…")
    t_wall_start = time.monotonic()

    with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
        for dt_s, rain_grid, pet_grid in _forcing_steps():
            step_start = time.monotonic()

            # --- 1. Subsurface lateral throughflow redistribution from previous step ---
            subsurface_inflow = fd8_route_flux(lateral_out_grid, fd8_weights)

            # --- 2. Run soil columns in parallel ---
            runoff_grid = np.zeros((nrows, ncols))
            new_lateral_out_grid = np.zeros((nrows, ncols))

            if use_gpu and _CUPY_AVAILABLE:
                # GPU path: vectorised across cells using CuPy
                _run_cells_gpu(
                    theta_grid, head_grid, soil_list, sim_configs, veg_list,
                    rain_grid, pet_grid, subsurface_inflow,
                    dt_s, col, runoff_grid, new_lateral_out_grid,
                )
            else:
                # CPU multiprocessing path
                futures = {}
                for idx in range(ncells):
                    r, c = divmod(idx, ncols)
                    soil = soil_list[idx] or soil_list[0]
                    sim_c = sim_configs[idx]
                    lat_inflow = float(subsurface_inflow[r, c])

                    f = pool.submit(
                        _run_cell,
                        idx,
                        theta_grid[idx].copy(),
                        head_grid[idx].copy(),
                        _soil_tuple(soil),
                        _sim_dict(sim_c),
                        col.nz,
                        col.dz_m,
                        float(rain_grid[r, c]),
                        float(pet_grid[r, c]),
                        dt_s,
                        _veg_dict(veg_list[idx]),
                        lat_inflow,
                        float(cum_infilt_grid[idx]),
                    )
                    futures[f] = idx

                for future in as_completed(futures):
                    result = future.result()
                    (idx, new_theta, new_head, runoff,
                     lat_out, new_cum_inf) = result
                    r, c = divmod(idx, ncols)
                    theta_grid[idx] = new_theta
                    head_grid[idx] = new_head
                    runoff_grid[r, c] = runoff
                    new_lateral_out_grid[r, c] = lat_out
                    cum_infilt_grid[idx] = new_cum_inf

            lateral_out_grid = new_lateral_out_grid

            # --- 3. Overland-flow routing (Darcy–Weisbach + FD8) ---
            velocity = darcy_weisbach_velocity(flow_depth, slope_grid, f_grid)
            flow_depth, vx, vy = update_flow_depth(
                flow_depth, runoff_grid, velocity, fd8_weights, cell_size, dt_s
            )

            # --- 4. Record diagnostics ---
            theta_mean_grid = theta_grid.mean(axis=1).reshape(nrows, ncols)
            t_elapsed += dt_s
            diag_rows.append(
                {
                    "step": step,
                    "time_hours": t_elapsed / 3600.0,
                    "mean_theta": float(theta_mean_grid.mean()),
                    "mean_runoff_mm_h": float(runoff_grid.mean()) * 1000.0 * 3600.0,
                    "mean_flow_depth_m": float(flow_depth.mean()),
                    "max_flow_depth_m": float(flow_depth.max()),
                    "mean_lateral_out_mm_h": float(lateral_out_grid.mean()) * 1000.0 * 3600.0,
                }
            )

            # --- 5. Save arrays ---
            if cfg.save_npy_arrays:
                np.save(str(output_dir / f"theta_mean_{step:05d}.npy"), theta_mean_grid)
                np.save(str(output_dir / f"flow_depth_{step:05d}.npy"), flow_depth)
                np.save(str(output_dir / f"flow_vx_{step:05d}.npy"), vx)
                np.save(str(output_dir / f"flow_vy_{step:05d}.npy"), vy)

            step_dur = time.monotonic() - step_start
            print(
                f"  Step {step:4d} | dt={dt_s:.0f}s | "
                f"mean_θ={diag_rows[-1]['mean_theta']:.3f} | "
                f"max_depth={diag_rows[-1]['max_flow_depth_m']:.4f} m | "
                f"wall={step_dur:.1f}s"
            )
            step += 1

    total_wall = time.monotonic() - t_wall_start
    print(f"\nSimulation complete: {step} steps, {total_wall:.1f}s wall time.")

    # -----------------------------------------------------------------------
    # Write diagnostics CSV
    # -----------------------------------------------------------------------
    diag_df = pd.DataFrame(diag_rows)
    diag_path = output_dir / "spatial_diagnostics.csv"
    diag_df.to_csv(str(diag_path), index=False)
    print(f"Wrote diagnostics: {diag_path}")

    # -----------------------------------------------------------------------
    # Animations
    # -----------------------------------------------------------------------
    if cfg.save_npy_arrays and step > 0:
        from spatial_visualise import animate_spatial_results
        print("Rendering animations…")
        animate_spatial_results(
            output_dir=output_dir,
            dem=dem,
            cell_size_m=cell_size,
            n_steps=step,
            fps=cfg.animation_fps,
            quiver_stride=cfg.quiver_stride,
            snapshot_interval=cfg.snapshot_interval_steps,
        )


# ---------------------------------------------------------------------------
# Optional GPU path (CuPy vectorised over cells)
# ---------------------------------------------------------------------------

def _run_cells_gpu(
    theta_grid: np.ndarray,
    head_grid: np.ndarray,
    soil_list,
    sim_configs,
    veg_list,
    rain_grid: np.ndarray,
    pet_grid: np.ndarray,
    subsurface_inflow: np.ndarray,
    dt_s: float,
    col: ColumnConfig,
    runoff_grid: np.ndarray,
    lateral_out_grid: np.ndarray,
) -> None:
    """GPU (CuPy) implementation: run all cell soil columns in a single
    vectorised pass.

    This function updates *runoff_grid* and *lateral_out_grid* in-place and
    writes updated theta/head back to *theta_grid* / *head_grid*.

    Note: this is a simplified vectorised approximation that skips adaptive
    sub-stepping for performance; a single sub-step of fixed dt_s/10 is used.
    For higher accuracy use the CPU path with adaptive sub-stepping.
    """
    import cupy as cp
    from simulate_soil_column import (
        hydraulic_conductivity, theta_from_head as th_from_h, head_from_theta
    )

    ncells, nz = theta_grid.shape
    nrows, ncols = rain_grid.shape

    # Transfer to GPU
    theta_gpu = cp.asarray(theta_grid, dtype=cp.float64)
    head_gpu = cp.asarray(head_grid, dtype=cp.float64)

    # Use a uniform soil/sim for GPU path (first non-None)
    soil0 = next(s for s in soil_list if s is not None)
    sim0 = sim_configs[0]
    _, depth_m = build_column_geometry(col.nz, col.dz_m)
    dz_m = col.dz_m

    rain_flat = cp.asarray(rain_grid.ravel() / 1000.0 / 3600.0, dtype=cp.float64)
    pet_flat = cp.asarray(pet_grid.ravel() / 1000.0 / 3600.0, dtype=cp.float64)
    lat_in = cp.asarray(subsurface_inflow.ravel(), dtype=cp.float64)

    # Effective rain including lateral inflow
    eff_rain = rain_flat + lat_in

    n_sub = max(1, int(dt_s / sim0.max_substep_seconds))
    sub_dt = dt_s / n_sub

    for _ in range(n_sub):
        # Vectorised Richards update across all cells
        # Conductivity: shape (ncells, nz)
        se_gpu = _gpu_effective_saturation(head_gpu, soil0)
        k_gpu = _gpu_hydraulic_conductivity(se_gpu, soil0)

        # Fluxes: shape (ncells, nz+1)
        q = cp.zeros((ncells, nz + 1), dtype=cp.float64)
        # Surface flux (infiltration)
        theta_s_eff = soil0.theta_s
        theta_deficit = cp.maximum(theta_s_eff - theta_gpu[:, 0], 1.0e-6)
        infilt_cap = soil0.ks_m_per_s * (
            1.0 + sim0.green_ampt_wetting_front_suction_m * theta_deficit / 1.0e-6
        )
        infilt_cap = cp.maximum(infilt_cap, soil0.ks_m_per_s)
        infilt = cp.minimum(eff_rain, infilt_cap)
        runoff_gpu = cp.maximum(eff_rain - infilt, 0.0)
        q[:, 0] = -infilt

        for j in range(1, nz):
            k_int = 0.5 * (k_gpu[:, j - 1] + k_gpu[:, j])
            dh_dz = (head_gpu[:, j - 1] - head_gpu[:, j]) / dz_m
            q[:, j] = -k_int * (dh_dz + 1.0)

        q[:, nz] = -k_gpu[:, -1]
        dtheta_rate = (q[:, 1:] - q[:, :-1]) / dz_m

        # Lateral sink
        slope_rad_gpu = cp.asarray(
            [np.deg2rad(sc.slope_angle_deg) for sc in sim_configs], dtype=cp.float64
        )[:, cp.newaxis]
        lat_sink = k_gpu * cp.sin(slope_rad_gpu) / col.dz_m  # simplified

        theta_gpu = theta_gpu + sub_dt * (dtheta_rate - lat_sink)
        theta_gpu = cp.clip(theta_gpu, soil0.theta_r + 1.0e-8, soil0.theta_s - 1.0e-8)
        head_gpu = _gpu_head_from_theta(theta_gpu, soil0)

        lat_out_gpu = lat_sink.sum(axis=1) * dz_m

    # Transfer back
    theta_grid[:] = cp.asnumpy(theta_gpu)
    head_grid[:] = cp.asnumpy(head_gpu)
    runoff_grid[:] = cp.asnumpy(runoff_gpu).reshape(nrows, ncols)
    lateral_out_grid[:] = cp.asnumpy(lat_out_gpu).reshape(nrows, ncols)


def _gpu_effective_saturation(head_gpu, soil: SoilProperties):
    import cupy as cp
    suction = cp.maximum(-head_gpu, 0.0)
    se_unsat = (1.0 + (soil.alpha_per_m * suction) ** soil.n) ** (-soil.m)
    return cp.where(head_gpu >= 0.0, 1.0, se_unsat)


def _gpu_hydraulic_conductivity(se_gpu, soil: SoilProperties):
    import cupy as cp
    se = cp.clip(se_gpu, 1.0e-9, 1.0)
    term = 1.0 - (1.0 - se ** (1.0 / soil.m)) ** soil.m
    return soil.ks_m_per_s * se ** soil.pore_connectivity * term ** 2


def _gpu_head_from_theta(theta_gpu, soil: SoilProperties):
    import cupy as cp
    theta = cp.clip(theta_gpu, soil.theta_r + 1.0e-12, soil.theta_s - 1.0e-12)
    se = (theta - soil.theta_r) / (soil.theta_s - soil.theta_r)
    suction = ((se ** (-1.0 / soil.m) - 1.0) ** (1.0 / soil.n)) / soil.alpha_per_m
    return cp.where(se >= 0.999999, 0.0, -suction)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="thetaFlow 2-D spatial simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dem", type=Path, help="Path to DEM GeoTIFF.")
    p.add_argument("--landcover", type=Path, help="Path to land-cover GeoTIFF (integer codes).")
    p.add_argument("--soilmap", type=Path, help="Path to soil-type GeoTIFF (integer codes).")
    p.add_argument(
        "--lc-map", type=Path,
        help="JSON file mapping integer land-cover codes to vegetation names "
             '(e.g. {"1":"grass","2":"broadleaf_woodland"}). '
             "If omitted, codes 1–11 are mapped to built-in vegetation types in order.",
    )
    p.add_argument(
        "--soil-types", type=Path, default=Path("soil_types_example.json"),
        help="Path to soil types JSON.",
    )
    p.add_argument(
        "--veg-types", type=Path, default=Path("vegetation_types.json"),
        help="Path to vegetation types JSON.",
    )
    p.add_argument("--forcing", type=Path, help="Path to uniform forcing CSV.")
    p.add_argument(
        "--forcing-dir", type=Path,
        help="Directory of gridded forcing GeoTIFFs (rainfall_NNNNN.tif / pet_NNNNN.tif).",
    )
    p.add_argument(
        "--config", type=Path, default=Path("spatial_config_example.json"),
        help="Path to spatial configuration JSON.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs_2d"),
        help="Directory for simulation outputs.",
    )
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1),
                   help="Number of CPU worker processes.")
    p.add_argument("--gpu", action="store_true", help="Enable CuPy GPU acceleration.")
    p.add_argument(
        "--synthetic", action="store_true",
        help="Run a synthetic test case (no real DEM/inputs required).",
    )
    p.add_argument(
        "--synthetic-size", type=int, default=10,
        help="Grid size N for synthetic test (N×N cells).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # -----------------------------------------------------------------------
    # Load config
    # -----------------------------------------------------------------------
    cfg_path = args.config
    if cfg_path.exists():
        cfg = load_spatial_config(cfg_path)
    else:
        print(f"Config file {cfg_path} not found, using defaults.")
        cfg = SpatialConfig()

    # -----------------------------------------------------------------------
    # Synthetic mode
    # -----------------------------------------------------------------------
    if args.synthetic or args.dem is None:
        print("Running in synthetic test mode…")
        N = args.synthetic_size
        dem = _generate_synthetic_dem(nrows=N, ncols=N, cell_size=cfg.cell_size_m)
        landcover_codes = np.ones((N, N), dtype=np.int32)  # all grass (code 1)
        soil_codes = np.ones((N, N), dtype=np.int32)       # all soil type 1
        code_to_veg_name: dict[int, str] = {1: "grass"}
        veg_json = args.veg_types if args.veg_types.exists() else Path("vegetation_types.json")
        soil_types_json = args.soil_types if args.soil_types.exists() else None

        # Use example forcing
        forcing_path = args.forcing or Path("forcing_example.csv")
        if forcing_path.exists():
            forcing_df = read_uniform_forcing(forcing_path, cfg.simulation.default_dt_hours)
            # Limit to first 5 steps for quick test
            forcing_df = forcing_df.head(5).copy()
        else:
            # Minimal synthetic forcing
            forcing_df = pd.DataFrame({
                "rainfall_mm_h": [0.0, 5.0, 15.0, 2.0, 0.0],
                "pet_mm_h": [0.2, 0.1, 0.1, 0.2, 0.3],
            })
            forcing_df = _process_forcing_df(forcing_df, cfg.simulation.default_dt_hours, "<synthetic>")

        run_spatial_simulation(
            dem=dem,
            landcover_codes=landcover_codes,
            soil_codes=soil_codes,
            veg_json_path=veg_json,
            soil_types_json_path=soil_types_json,
            code_to_veg_name=code_to_veg_name,
            forcing_uniform=forcing_df,
            forcing_gridded=None,
            cfg=cfg,
            output_dir=args.output_dir,
            n_workers=args.workers,
            use_gpu=args.gpu,
        )
        return

    # -----------------------------------------------------------------------
    # Real data mode
    # -----------------------------------------------------------------------
    from spatial_utils import read_dem, read_raster_int

    print(f"Loading DEM: {args.dem}")
    dem, meta = read_dem(args.dem, target_cell_size_m=cfg.cell_size_m)
    print(f"  DEM shape: {dem.shape}  cell size: {meta.cell_size_m} m")

    landcover_codes: Optional[np.ndarray] = None
    if args.landcover:
        print(f"Loading land cover: {args.landcover}")
        landcover_codes = read_raster_int(args.landcover, meta)

    soil_codes: Optional[np.ndarray] = None
    if args.soilmap:
        print(f"Loading soil map: {args.soilmap}")
        soil_codes = read_raster_int(args.soilmap, meta)

    # Build code → name mapping
    code_to_veg_name_real: dict[int, str] = {}
    if args.lc_map and args.lc_map.exists():
        with args.lc_map.open("r", encoding="utf-8") as f:
            code_to_veg_name_real = {int(k): v for k, v in json.load(f).items()}
    else:
        # Default: map 1–len(VEGETATION_LIBRARY) in order
        for i, name in enumerate(VEGETATION_LIBRARY, start=1):
            code_to_veg_name_real[i] = name

    veg_json = args.veg_types if args.veg_types.exists() else Path("vegetation_types.json")
    soil_types_json = args.soil_types if args.soil_types.exists() else None

    # Forcing
    forcing_df_real: Optional[pd.DataFrame] = None
    forcing_grid_real: Optional[list[dict]] = None

    if args.forcing:
        forcing_df_real = read_uniform_forcing(args.forcing, cfg.simulation.default_dt_hours)
    elif args.forcing_dir:
        from spatial_utils import read_gridded_forcing
        forcing_grid_real = read_gridded_forcing(
            args.forcing_dir, meta, cfg.simulation.default_dt_hours
        )
    else:
        parser.error("Provide --forcing (CSV) or --forcing-dir (gridded GeoTIFFs)")

    run_spatial_simulation(
        dem=dem,
        landcover_codes=landcover_codes,
        soil_codes=soil_codes,
        veg_json_path=veg_json,
        soil_types_json_path=soil_types_json,
        code_to_veg_name=code_to_veg_name_real,
        forcing_uniform=forcing_df_real,
        forcing_gridded=forcing_grid_real,
        cfg=cfg,
        output_dir=args.output_dir,
        n_workers=args.workers,
        use_gpu=args.gpu,
    )


if __name__ == "__main__":
    main()
