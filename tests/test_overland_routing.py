from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from simulate_soil_column import (
    SimulationConfig,
    SimulationState,
    SoilProperties,
    one_substep,
    theta_from_head,
)
from simulate_spatial import SpatialConfig, _generate_synthetic_dem, run_spatial_simulation
from spatial_utils import compute_fd8_weights, compute_slope_grid, update_flow_depth

VEG_JSON = Path(__file__).resolve().parents[1] / "vegetation_types.json"
LOAM = SoilProperties(theta_r=0.078, theta_s=0.43, alpha_per_m=3.6, n=1.56, ks_m_per_s=2.89e-6)
CELL = 50.0
MM_H = 1.0 / 1000.0 / 3600.0


def _terrain(dem):
    slope = np.maximum(compute_slope_grid(dem, CELL), 1.0e-4)
    return slope, np.full(dem.shape, 0.5), compute_fd8_weights(dem, CELL)


def _plane(nrows=6, ncols=10, drop=0.5):
    return np.tile(100.0 - drop * np.arange(ncols), (nrows, 1)).astype(float)


def _route(h, rain_mm_h, dem, hours=1, **kw):
    slope, f, w = _terrain(dem)
    runoff = np.full(dem.shape, rain_mm_h * MM_H)
    for _ in range(hours):
        h, _, _ = update_flow_depth(h, runoff, slope, f, w, CELL, 3600.0, dem=dem, **kw)
    return h


def test_closed_bowl_conserves_water():
    y, x = np.mgrid[0:12, 0:12]
    bowl = ((x - 5.5) ** 2 + (y - 5.5) ** 2) * 0.05
    h = _route(np.zeros(bowl.shape), 10.0, bowl, hours=3)
    assert h.sum() == pytest.approx(bowl.size * 0.03, rel=1e-9)
    assert h[5, 5] > 10 * h[0, 0]  # water collects at the bottom


def test_pit_retains_water_with_static_weights():
    dem = _plane()
    dem[3, 4] -= 2.0
    slope, f, w = _terrain(dem)
    h = np.zeros(dem.shape)
    h[3, 4] = 0.05
    h, _, _ = update_flow_depth(h, np.zeros(dem.shape), slope, f, w, CELL, 3600.0)
    assert h[3, 4] == pytest.approx(0.05)


def test_depression_fills_and_spills():
    dem = _plane()
    dem[2:5, 3:6] -= 0.6  # column 5 becomes a basin 0.1 m below its downslope rim
    h = _route(np.zeros(dem.shape), 50.0, dem, hours=4)
    assert h[3, 5] > 0.1                      # basin full
    assert h[3, 7] > 0.0 and h[3, 9] > 0.0    # and spilling downslope


def test_dry_start_routes_within_the_step():
    # Starting dry must not dump the whole step's runoff without routing
    h = _route(np.zeros((6, 10)), 6.0, _plane())
    assert h[1, 0] < 0.6 * 0.006
    assert h[1, -1] > h[1, 0]


def test_depression_storage_threshold():
    dem = _plane()
    # Below the threshold nothing moves
    h = _route(np.zeros(dem.shape), 3.0, dem, depression_storage_m=0.005)
    np.testing.assert_allclose(h, 0.003)
    # Above it only the excess runs off; storage stays behind
    h = _route(np.full(dem.shape, 0.009), 0.0, dem, depression_storage_m=0.005)
    assert np.all(h >= 0.005 - 1e-12)
    assert h[1, 0] < 0.009
    free = _route(np.full(dem.shape, 0.009), 0.0, dem)
    assert free[1, 0] < h[1, 0]


def test_saturated_column_returns_saturation_excess():
    sim = SimulationConfig(initial_head_m=0.0, default_dt_hours=1.0, max_substep_seconds=60.0,
                           min_theta_buffer=0.002, slope_angle_deg=5.0, hillslope_flow_path_m=50.0)
    head = np.zeros(10)
    state = SimulationState(head_m=head, theta=theta_from_head(head, LOAM))
    _, diag, _ = one_substep(state, LOAM, sim, 0.1, 0.05, 60.0, 20.0, 0.0, 0.002,
                             lateral_inflow_m_per_s=1.0e-6)
    assert diag["saturation_excess_flux_m_per_s"] > 0.0
    assert diag["runoff_flux_m_per_s"] >= diag["saturation_excess_flux_m_per_s"]


def _spatial(tmp_path, ds, **kw):
    cfg = SpatialConfig()
    cfg.column.nz, cfg.column.dz_m = 10, 0.05
    cfg.save_npy_arrays = False
    cfg.depression_storage_m = ds
    forcing = pd.DataFrame({"rainfall_mm_h": [30.0, 30.0, 0.0, 0.0],
                            "pet_mm_h": [0.0] * 4, "dt_seconds": [3600.0] * 4})
    return run_spatial_simulation(
        dem=_generate_synthetic_dem(6, 8, CELL), landcover_codes=None, soil_codes=None,
        veg_json_path=VEG_JSON, soil_types_json_path=None, code_to_veg_name={},
        forcing_uniform=forcing, forcing_gridded=None, cfg=cfg, output_dir=tmp_path,
        render_animations=False, **kw,
    )


def test_spatial_surface_water_balance_and_microtopography(tmp_path):
    results = {}
    for ds in (0.0, 0.02):
        res = _spatial(tmp_path, ds)
        d = res["diagnostics"]
        generated = res["cumulative_runoff_mm"].mean()
        balance = (generated - res["cumulative_runon_infiltration_mm"].mean()
                   - d["surface_outflow_mm"].sum() - res["final_flow_depth_m"].mean() * 1000.0)
        assert abs(balance) < 1e-6
        assert generated > 0.0
        results[ds] = res
    out = {ds: r["diagnostics"]["surface_outflow_mm"].sum() for ds, r in results.items()}
    assert out[0.02] < out[0.0]
    assert (results[0.02]["cumulative_runon_infiltration_mm"].mean()
            > results[0.0]["cumulative_runon_infiltration_mm"].mean())


def test_depression_storage_grid_overrides_with_nan_fallback(tmp_path):
    grid = np.full((6, 8), 0.01)
    grid[0, 0] = np.nan
    res = _spatial(tmp_path, 0.003, depression_storage_grid=grid)
    assert res["depression_storage_m"][0, 0] == pytest.approx(0.003)
    assert res["depression_storage_m"][3, 3] == pytest.approx(0.01)
