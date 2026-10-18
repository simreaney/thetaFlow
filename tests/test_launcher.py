import numpy as np
import pytest

import simulate_soil_column
import simulate_spatial
import thetaflow

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from spatial_utils import read_dem  # noqa: E402


@pytest.fixture
def captured(monkeypatch):
    calls = {}
    monkeypatch.setattr(simulate_soil_column, "main", lambda argv: calls.update(column=argv))
    monkeypatch.setattr(simulate_spatial, "main", lambda argv: calls.update(spatial=argv))
    return calls


def test_column_mode_passes_args(captured):
    thetaflow.main(["column", "--output-dir", "x"])
    assert captured == {"column": ["--output-dir", "x"]}


def test_column_mode_rejects_vegetation(captured):
    with pytest.raises(SystemExit):
        thetaflow.main(["column", "--vegetation", "wheat"])


def test_vegetation_mode_defaults_to_grass(captured):
    thetaflow.main(["vegetation"])
    assert captured["column"] == ["--vegetation", "grass"]


def test_vegetation_mode_keeps_given_type(captured):
    thetaflow.main(["veg", "--vegetation=wheat"])
    assert captured["column"] == ["--vegetation=wheat"]


def test_2d_mode_dispatches(captured):
    thetaflow.main(["2D", "--dem", "dem.tif"])
    assert captured == {"spatial": ["--dem", "dem.tif"]}


def test_unknown_mode_exits(captured):
    with pytest.raises(SystemExit):
        thetaflow.main(["3d"])


def _write_dem(path, crs, transform, nodata_corner=True):
    n = 8
    rows, cols = np.mgrid[0:n, 0:n]
    z = (100.0 + 2.0 * cols + 0.5 * rows).astype("float32")
    if nodata_corner:
        z[0, 0] = -9999.0
    with rasterio.open(
        path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
        crs=crs, transform=transform, nodata=-9999.0,
    ) as dst:
        dst.write(z, 1)


def test_read_dem_native_resolution_projected(tmp_path):
    path = tmp_path / "dem.tif"
    _write_dem(path, "EPSG:27700", from_origin(400000, 500000, 10, 10))
    dem, meta = read_dem(path, target_cell_size_m=None)
    assert dem.shape == (8, 8)
    assert meta.cell_size_m == pytest.approx(10.0)
    assert np.isnan(dem[0, 0])
    assert dem[7, 7] == pytest.approx(100.0 + 14.0 + 3.5)


def test_read_dem_geographic_is_reprojected_to_utm(tmp_path):
    path = tmp_path / "dem_geo.tif"
    _write_dem(path, "EPSG:4326", from_origin(-1.6, 54.8, 0.001, 0.001))
    dem, meta = read_dem(path, target_cell_size_m=20.0)
    assert meta.crs.to_epsg() == 32630
    assert meta.cell_size_m == 20.0
    assert np.nanmin(dem) >= 100.0 and np.nanmax(dem) <= 118.0


def test_spatial_cli_runs_on_geotiff(tmp_path):
    dem_path = tmp_path / "dem.tif"
    _write_dem(dem_path, "EPSG:27700", from_origin(400000, 500000, 25, 25))
    forcing = tmp_path / "forcing.csv"
    forcing.write_text("rainfall_mm_h,pet_mm_h\n5.0,0.1\n0.0,0.2\n")
    out = tmp_path / "out"
    thetaflow.main([
        "2d", "--dem", str(dem_path), "--forcing", str(forcing),
        "--native-resolution", "--vegetation", "wheat", "--workers", "1",
        "--config", str(tmp_path / "missing.json"), "--output-dir", str(out),
    ])
    assert (out / "spatial_diagnostics.csv").exists()
