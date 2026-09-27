from pathlib import Path

import numpy as np
import pandas as pd

from simulate_soil_column import SoilProperties
from simulate_spatial import SpatialConfig, _generate_synthetic_dem, run_spatial_simulation
from soil_depth import (
    RegressionRule,
    base_grids_from_soil_list,
    profiles_from_regression,
    soil_list_from_profiles,
)
from simulate_soil_column import build_column_geometry

VEG_JSON = Path(__file__).resolve().parents[1] / "vegetation_types.json"
LOAM = SoilProperties(theta_r=0.078, theta_s=0.43, alpha_per_m=3.6, n=1.56, ks_m_per_s=2.89e-6)
FORCING = pd.DataFrame({
    "rainfall_mm_h": [0.0, 10.0, 25.0, 0.0],
    "pet_mm_h": [0.2, 0.1, 0.0, 0.3],
    "dt_seconds": [3600.0] * 4,
    "timestamp": pd.date_range("2026-06-01", periods=4, freq="h", tz="UTC"),
})


def _cfg():
    cfg = SpatialConfig()
    cfg.column.nz, cfg.column.dz_m = 10, 0.05
    cfg.save_npy_arrays = False
    return cfg


def _run(tmp_path, **kw):
    dem = _generate_synthetic_dem(4, 5, 50.0)
    return run_spatial_simulation(
        dem=dem, landcover_codes=None, soil_codes=None,
        veg_json_path=VEG_JSON,
        soil_types_json_path=None, code_to_veg_name={},
        forcing_uniform=FORCING.copy(), forcing_gridded=None, cfg=_cfg(),
        output_dir=tmp_path, n_workers=1, render_animations=False, **kw,
    )


def test_returns_result_arrays(tmp_path):
    res = _run(tmp_path)
    assert res["theta_mean_final"].shape == (4, 5)
    assert res["theta_layers_final"].shape == (4, 5, 10)
    assert res["theta_mean_steps"].shape == (4, 4, 5)
    assert res["n_steps"] == 4 and not res["cancelled"]
    assert len(res["timestamps"]) == 4
    assert (tmp_path / "spatial_diagnostics.csv").exists()
    assert np.all(res["cumulative_runoff_mm"] >= 0)


def test_depth_varying_override_runs(tmp_path):
    _, depth = build_column_geometry(10, 0.05)
    base = base_grids_from_soil_list([LOAM] * 20, (4, 5))
    prof = profiles_from_regression([RegressionRule("ks", "exp", 0.2)], depth, base)
    res = _run(tmp_path, soil_list_override=soil_list_from_profiles(prof))
    assert np.all(np.isfinite(res["theta_layers_final"]))


def test_uniform_override_matches_default(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b", soil_list_override=[LOAM] * 20)
    np.testing.assert_allclose(a["theta_layers_final"], b["theta_layers_final"], rtol=1e-12)


def test_active_mask_skips_cells(tmp_path):
    mask = np.ones((4, 5), dtype=bool)
    mask[0, :] = False
    res = _run(tmp_path, active_mask=mask)
    # Inactive cells keep their initial state and never produce runoff or ponding
    initial_row = res["theta_mean_steps"][0, 0]
    assert np.all(res["theta_mean_steps"][:, 0] == initial_row)
    assert np.all(res["cumulative_runoff_mm"][0] == 0)
    assert np.all(res["max_flow_depth_m"][0] == 0)
    assert np.all(res["theta_mean_steps"][-1, 1:] != initial_row[0])


def test_progress_callback_cancels(tmp_path):
    seen = []

    def cb(step, total):
        seen.append((step, total))
        return step < 2

    res = _run(tmp_path, progress_callback=cb)
    assert seen == [(1, 4), (2, 4)]
    assert res["cancelled"] and res["n_steps"] == 2
    assert res["theta_mean_steps"].shape[0] == 2


def test_default_vegetation_without_landcover(tmp_path):
    a = _run(tmp_path / "grass")
    b = _run(tmp_path / "bare", default_vegetation="bare_soil")
    assert not np.array_equal(a["theta_layers_final"], b["theta_layers_final"])
