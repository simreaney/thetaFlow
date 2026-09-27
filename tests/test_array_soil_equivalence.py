"""Per-layer (array) soil properties must reproduce the scalar path exactly
when every layer has the same value, and run stably when they differ."""

import numpy as np
import pandas as pd

from simulate_soil_column import (
    ColumnConfig,
    SimulationConfig,
    SoilProperties,
    VEGETATION_LIBRARY,
    run_simulation,
)
from simulate_spatial import _run_cell

NZ, DZ = 20, 0.05
SCALAR = (0.078, 0.43, 3.6, 1.56, 2.89e-6, 0.5)
SIM = dict(
    initial_head_m=-1.0, default_dt_hours=1.0, max_substep_seconds=60.0,
    min_theta_buffer=0.002, slope_angle_deg=5.0, hillslope_flow_path_m=50.0,
)


def _arrays(values):
    return tuple(np.full(NZ, v) for v in values)


def _cell(soil_tuple, veg=None, rain=12.0):
    sp = SoilProperties(*soil_tuple)
    from simulate_soil_column import theta_from_head
    h0 = np.full(NZ, -1.0)
    return _run_cell(0, theta_from_head(h0, sp), h0, soil_tuple, SIM, NZ, DZ,
                     rain, 0.2, 3600.0, veg, 0.0, 1e-6)


def test_run_cell_uniform_array_matches_scalar():
    grass = VEGETATION_LIBRARY["grass"]
    veg = {"name": grass.name, "rooting_depth_m": grass.rooting_depth_m,
           "pet_scale": grass.pet_scale, "description": ""}
    for v in (None, veg):
        a = _cell(SCALAR, v)
        b = _cell(_arrays(SCALAR), v)
        # Equal to rounding: numpy may take different SIMD paths for array exponents
        np.testing.assert_allclose(a[1], b[1], rtol=1e-12)
        np.testing.assert_allclose(a[2], b[2], rtol=1e-12)
        np.testing.assert_allclose(a[3:], b[3:], rtol=1e-12)


def test_run_simulation_uniform_array_matches_scalar(tmp_path):
    forcing = pd.DataFrame({
        "rainfall_mm_h": [0.0, 8.0, 20.0, 1.0, 0.0],
        "pet_mm_h": [0.2, 0.1, 0.0, 0.2, 0.3],
        "dt_seconds": [3600.0] * 5,
    })
    col = ColumnConfig(nz=NZ, dz_m=DZ)
    sim = SimulationConfig(**SIM)
    run_simulation(SoilProperties(*SCALAR), col, sim, forcing.copy(), tmp_path / "s")
    run_simulation(SoilProperties(*_arrays(SCALAR)), col, sim, forcing.copy(), tmp_path / "a")
    for name in ("soil_moisture_profiles.csv", "water_balance_diagnostics.csv"):
        pd.testing.assert_frame_equal(
            pd.read_csv(tmp_path / "s" / name), pd.read_csv(tmp_path / "a" / name),
            check_exact=False, rtol=1e-12,
        )


def test_depth_varying_column_is_finite_and_conserves_bounds():
    depth = np.arange(NZ) * DZ
    soil = (
        np.full(NZ, 0.05),
        0.45 - 0.1 * depth,                  # porosity decreasing with depth
        np.full(NZ, 3.6),
        np.full(NZ, 1.5),
        1e-5 * np.exp(-depth / 0.3),         # Ks decaying with depth
        np.full(NZ, 0.5),
    )
    _, theta, head, runoff, lat_out, _ = _cell(soil, rain=30.0)
    assert np.all(np.isfinite(theta)) and np.all(np.isfinite(head))
    assert np.all(theta <= soil[1]) and np.all(theta >= soil[0])
    assert runoff >= 0.0 and lat_out >= 0.0
