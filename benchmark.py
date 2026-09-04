#!/usr/bin/env python3
"""Benchmark for the thetaFlow 2-D spatial simulation.

Runs the simulation over a defined short rainfall series and a defined
synthetic DEM and reports wall-clock timing and throughput statistics.

Usage
-----
python benchmark.py
python benchmark.py --grid-size 20 --workers 4
python benchmark.py --grid-size 5 --steps 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from simulate_soil_column import _process_forcing_df
from simulate_spatial import (
    SpatialConfig,
    ColumnConfig,
    SimulationConfig,
    run_spatial_simulation,
)

# ---------------------------------------------------------------------------
# Benchmark rainfall series
# ---------------------------------------------------------------------------

# A short 12-step event: dry lead-in, sharp peak, recession.
# Columns: rainfall_mm_h, pet_mm_h
_BENCHMARK_FORCING = pd.DataFrame(
    {
        "rainfall_mm_h": [0.0, 0.0, 2.0, 8.0, 18.0, 22.0, 10.0, 4.0, 1.0, 0.0, 0.0, 0.0],
        "pet_mm_h":      [0.3, 0.3, 0.2, 0.1, 0.1,  0.1,  0.1,  0.2, 0.3, 0.3, 0.3, 0.2],
    }
)

# ---------------------------------------------------------------------------
# Benchmark DEM generator
# ---------------------------------------------------------------------------

def _benchmark_dem(nrows: int, ncols: int, cell_size_m: float = 50.0) -> np.ndarray:
    """Return a reproducible synthetic hillslope DEM (metres).

    The surface is a gentle planar slope with small sinusoidal undulations
    and no random noise so that results are deterministic across runs.
    """
    x = np.linspace(0.0, 1.0, ncols)
    y = np.linspace(0.0, 1.0, nrows)
    XX, YY = np.meshgrid(x, y)
    # Main slope west→east (100 m to 60 m), gentle cross-slope undulation
    dem = 100.0 - 40.0 * XX + 5.0 * np.sin(2.0 * np.pi * YY)
    return dem.astype(np.float64)

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    grid_size: int = 10,
    n_steps: Optional[int] = None,
    n_workers: int = 1,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """Run the benchmark and return a dict of timing statistics.

    Parameters
    ----------
    grid_size:
        Number of rows and columns of the square synthetic DEM.
    n_steps:
        Number of forcing timesteps to run.  Defaults to the full
        12-step benchmark series.
    n_workers:
        CPU worker processes passed to :func:`run_spatial_simulation`.
    output_dir:
        Output directory for simulation artefacts.  Defaults to a
        temporary directory that is cleaned up after the run.
    verbose:
        If *True* the simulation output is printed to stdout; otherwise
        it is suppressed to keep benchmark output clean.

    Returns
    -------
    dict
        Keys: ``grid_size``, ``ncells``, ``nsteps``, ``wall_time_s``,
        ``s_per_step``, ``cells_per_s``, ``n_workers``.
    """
    import tempfile

    # --- Forcing ---
    forcing_raw = _BENCHMARK_FORCING.copy()
    if n_steps is not None:
        forcing_raw = forcing_raw.head(n_steps).copy()
    forcing_df = _process_forcing_df(
        forcing_raw, default_dt_hours=1.0, source_name="<benchmark>"
    )
    actual_steps = len(forcing_df)

    # --- DEM ---
    dem = _benchmark_dem(grid_size, grid_size)

    # --- Config: small column for speed ---
    cfg = SpatialConfig(
        cell_size_m=50.0,
        column=ColumnConfig(nz=20, dz_m=0.05),
        simulation=SimulationConfig(
            initial_head_m=-0.5,
            default_dt_hours=1.0,
            max_substep_seconds=120.0,
            min_theta_buffer=0.002,
            max_dtheta_per_substep=0.004,
            min_substep_seconds=5.0,
            green_ampt_wetting_front_suction_m=0.11,
            slope_angle_deg=2.0,
            hillslope_flow_path_m=50.0,
        ),
        save_npy_arrays=False,
    )

    # --- Output directory ---
    _tmp_dir = None
    if output_dir is None:
        _tmp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(_tmp_dir.name)

    # --- Run ---
    # Suppress simulation output unless verbose is requested.  We redirect
    # at the OS file-descriptor level so that both main-process prints and any
    # output written directly to fd 1 by worker subprocesses are silenced.
    _devnull = None
    _saved_fd = None
    try:
        if not verbose:
            _devnull = open(os.devnull, "w")  # noqa: SIM115
            _saved_fd = os.dup(sys.stdout.fileno())
            os.dup2(_devnull.fileno(), sys.stdout.fileno())
            sys.stdout.flush()

        t0 = time.monotonic()
        run_spatial_simulation(
            dem=dem,
            landcover_codes=None,
            soil_codes=None,
            veg_json_path=Path("vegetation_types.json"),
            soil_types_json_path=None,
            code_to_veg_name={},
            forcing_uniform=forcing_df,
            forcing_gridded=None,
            cfg=cfg,
            output_dir=output_dir,
            n_workers=n_workers,
            use_gpu=False,
        )
        wall_time = time.monotonic() - t0
    finally:
        if _saved_fd is not None:
            sys.stdout.flush()
            os.dup2(_saved_fd, sys.stdout.fileno())
            os.close(_saved_fd)
        if _devnull is not None:
            _devnull.close()
        if _tmp_dir is not None:
            _tmp_dir.cleanup()

    ncells = grid_size * grid_size
    return {
        "grid_size": grid_size,
        "ncells": ncells,
        "nsteps": actual_steps,
        "wall_time_s": wall_time,
        "s_per_step": wall_time / actual_steps if actual_steps else float("nan"),
        "cells_per_s": ncells * actual_steps / wall_time if wall_time > 0 else float("nan"),
        "n_workers": n_workers,
    }


def print_results(results: dict) -> None:
    """Print a formatted benchmark summary."""
    print()
    print("=" * 50)
    print("thetaFlow Benchmark Results")
    print("=" * 50)
    print(f"  Grid size  : {results['grid_size']} × {results['grid_size']} "
          f"({results['ncells']} cells)")
    print(f"  Steps      : {results['nsteps']}")
    print(f"  Workers    : {results['n_workers']}")
    print(f"  Wall time  : {results['wall_time_s']:.2f} s")
    print(f"  Per step   : {results['s_per_step']:.3f} s/step")
    print(f"  Throughput : {results['cells_per_s']:.1f} cell-steps/s")
    print("=" * 50)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="thetaFlow spatial simulation benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--grid-size", type=int, default=10,
        help="Side length of the square synthetic DEM (N×N cells).",
    )
    p.add_argument(
        "--steps", type=int, default=None,
        help="Number of forcing timesteps to run (default: all 12 in the benchmark series).",
    )
    p.add_argument(
        "--workers", type=int, default=1,
        help="Number of CPU worker processes.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print simulation output during the run.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Save simulation outputs here (default: a temporary directory).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)

    print(
        f"Running benchmark: {args.grid_size}×{args.grid_size} grid, "
        f"{args.steps or 12} steps, {args.workers} worker(s)…"
    )

    results = run_benchmark(
        grid_size=args.grid_size,
        n_steps=args.steps,
        n_workers=args.workers,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    print_results(results)


if __name__ == "__main__":
    main()
