"""End-to-end tests of the ThetaFlow Processing algorithms inside QGIS.

Skipped when the QGIS Python bindings are not importable.  Run with the
Python that QGIS uses, e.g.::

    QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_qgis_plugin.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qgis.core")

from osgeo import gdal, osr  # noqa: E402
from qgis.core import QgsApplication, QgsProcessingException  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BNG = 27700
NROWS, NCOLS, CELL = 8, 10, 20.0
X0, Y0 = 400000.0, 500000.0


@pytest.fixture(scope="session")
def processing():
    app = QgsApplication([], False)
    app.initQgis()
    for p in ("/usr/share/qgis/python/plugins", os.path.join(QgsApplication.pkgDataPath(), "python", "plugins")):
        if p not in sys.path:
            sys.path.append(p)
    from processing.core.Processing import Processing
    import processing as proc

    Processing.initialize()
    sys.path.insert(0, str(ROOT / "qgis_plugin"))
    from thetaflow.provider import ThetaFlowProvider

    provider = ThetaFlowProvider()
    QgsApplication.processingRegistry().addProvider(provider)
    yield proc
    QgsApplication.processingRegistry().removeProvider(provider)


def _write(path, data, epsg=BNG, gt=None, nodata=None, dtype=gdal.GDT_Float32):
    data = np.asarray(data)
    ds = gdal.GetDriverByName("GTiff").Create(str(path), data.shape[1], data.shape[0], 1, dtype)
    ds.SetGeoTransform(gt or (X0, CELL, 0, Y0, 0, -CELL))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(data)
    ds = None
    return str(path)


def _read(path):
    ds = gdal.Open(str(path))
    arrays = np.stack([ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)])
    return ds, arrays


@pytest.fixture
def rasters(tmp_path):
    rr, cc = np.mgrid[0:NROWS, 0:NCOLS]
    dem = 100.0 - 0.8 * cc - 0.3 * rr + 0.05 * (cc - 5) ** 2
    dem[0, 0] = -9999.0  # one NoData cell
    lc = np.where(cc < 5, 4, 1)  # UKCEH: 4 improved grassland, 1 deciduous woodland
    lc[NROWS - 1, NCOLS - 1] = 77  # unmapped code → default vegetation
    paths = {
        "dem": _write(tmp_path / "dem.tif", dem, nodata=-9999.0),
        "lc": _write(tmp_path / "lcm.tif", lc, dtype=gdal.GDT_Byte),
        # Horizon rasters on a coarser grid in the same CRS (tests resampling)
        "ks_top": _write(tmp_path / "ks_0-10cm.tif", np.full((4, 5), 100.0),
                         gt=(X0, 2 * CELL, 0, Y0, 0, -2 * CELL)),
        "ks_sub": _write(tmp_path / "ks_10-40cm.tif", np.full((4, 5), 10.0),
                         gt=(X0, 2 * CELL, 0, Y0, 0, -2 * CELL)),
        "ths_top": _write(tmp_path / "theta_s_0-20cm.tif", np.full((NROWS, NCOLS), 0.5)),
        "decay": _write(tmp_path / "decay_m.tif", np.where(cc < 5, 0.1, 0.3)),
        "soil_depth": _write(tmp_path / "soil_depth.tif", np.full((NROWS, NCOLS), 0.25)),
    }
    forcing = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-01", periods=4, freq="h").strftime("%Y-%m-%d %H:%M"),
        "rainfall_mm_h": [0.0, 12.0, 30.0, 0.0],
        "pet_mm_h": [0.2, 0.1, 0.0, 0.3],
    })
    forcing.to_csv(tmp_path / "forcing.csv", index=False)
    paths["forcing"] = str(tmp_path / "forcing.csv")
    return paths


def _column(nz=10, dz=0.05):
    return {"NZ": nz, "DZ": dz}


def test_provider_lists_algorithms(processing):
    reg = QgsApplication.processingRegistry()
    for name in ("run_spatial", "prepare_inputs", "fetch_weather"):
        assert reg.algorithmById(f"thetaflow:{name}") is not None


def test_run_with_horizons_landcover_and_csv(processing, rasters, tmp_path):
    out = tmp_path / "run1"
    res = processing.run("thetaflow:run_spatial", {
        "DEM": rasters["dem"],
        "SOIL_DEPTH_MODE": 1,
        "HORIZON_LAYERS": [rasters["ks_top"], rasters["ks_sub"], rasters["ths_top"]],
        "KS_UNIT": 1,  # cm/day
        "LANDCOVER": rasters["lc"],
        "LC_PRESET": 1,  # UKCEH
        "DEFAULT_VEG": "moorland",
        "WEATHER_SOURCE": 0,
        "WEATHER_CSV": rasters["forcing"],
        "OUTPUT_FOLDER": str(out),
        **_column(),
    })
    ds, theta = _read(res["THETA_MEAN_FINAL"])
    assert theta.shape == (1, NROWS, NCOLS)
    assert ds.GetGeoTransform() == (X0, CELL, 0, Y0, 0, -CELL)
    assert osr.SpatialReference(wkt=ds.GetProjection()).GetAuthorityCode(None) == str(BNG)
    assert theta[0, 0, 0] == -9999.0  # DEM NoData masked
    valid = theta[0][theta[0] != -9999.0]
    assert np.all((valid > 0.0) & (valid < 0.6))

    _, layers = _read(res["THETA_LAYERS_FINAL"])
    assert layers.shape[0] == 10
    ds_ts, steps = _read(res["THETA_MEAN_TIMESERIES"])
    assert steps.shape[0] == 4
    assert ds_ts.GetRasterBand(1).GetDescription() == "2026-06-01 00:00 UTC"
    diag = pd.read_csv(res["DIAGNOSTICS"])
    assert len(diag) == 4
    assert Path(res["FORCING"]).exists()
    for key in ("MAX_FLOW_DEPTH", "CUMULATIVE_RUNOFF", "CUMULATIVE_LATERAL"):
        assert Path(res[key]).exists()


def test_prepare_inputs_horizons_and_vegetation(processing, rasters, tmp_path):
    res = processing.run("thetaflow:prepare_inputs", {
        "DEM": rasters["dem"],
        "SOIL_DEPTH_MODE": 1,
        "HORIZON_LAYERS": [rasters["ks_top"], rasters["ks_sub"]],
        "KS_UNIT": 1,
        "LANDCOVER": rasters["lc"],
        "LC_PRESET": 1,
        "DEFAULT_VEG": "moorland",
        "VEG_TABLE": ["broadleaf_woodland", "0.4", "", "3.0"],
        "OUTPUT_FOLDER": str(tmp_path / "prep"),
        **_column(),
    })
    ds, ks = _read(res["SOIL_KS_M_PER_S"])
    assert ks.shape == (10, NROWS, NCOLS)
    assert ds.GetRasterBand(3).GetDescription() == "layer 3: 0.100 m"
    np.testing.assert_allclose(ks[:2, 3, 3], 100.0 / 100 / 86400, rtol=1e-6)   # 0–10 cm
    np.testing.assert_allclose(ks[2:, 3, 3], 10.0 / 100 / 86400, rtol=1e-6)    # 10 cm and below
    _, ths = _read(res["SOIL_THETA_S"])
    np.testing.assert_allclose(ths[:, 3, 3], 0.43, rtol=1e-6)  # no theta_s horizon → loam

    _, veg = _read(res["VEGETATION_CLASS"])
    _, rooting = _read(res["ROOTING_DEPTH"])
    _, friction = _read(res["FRICTION_FACTOR"])
    assert rooting[0, 3, 2] == pytest.approx(0.3)                    # grass
    assert rooting[0, 3, 7] == pytest.approx(0.4)                    # woodland, edited in table
    assert friction[0, 3, 7] == pytest.approx(3.0)
    assert rooting[0, NROWS - 1, NCOLS - 1] == pytest.approx(0.2)    # unmapped → moorland
    assert veg[0, 0, 0] == 0                                         # NoData


def test_run_with_regression_and_typed_series(processing, rasters, tmp_path):
    res = processing.run("thetaflow:prepare_inputs", {
        "DEM": rasters["dem"],
        "SOIL_DEPTH_MODE": 2,
        "REGRESSION_TABLE": [
            "ks", "exp", "decay_m", "",
            "theta_s", "expr", "", "where(z < soil_depth, p0, p0 - 0.1)",
            "n", "linear", "-0.2", "",
        ],
        "REGRESSION_VARIABLES": [rasters["decay"], rasters["soil_depth"]],
        "OUTPUT_FOLDER": str(tmp_path / "prep_reg"),
        **_column(),
    })
    _, ks = _read(res["SOIL_KS_M_PER_S"])
    z = np.arange(10) * 0.05
    np.testing.assert_allclose(ks[:, 2, 2], 2.89e-6 * np.exp(-z / 0.1), rtol=1e-6)
    np.testing.assert_allclose(ks[:, 2, 8], 2.89e-6 * np.exp(-z / 0.3), rtol=1e-6)
    _, ths = _read(res["SOIL_THETA_S"])
    np.testing.assert_allclose(ths[:, 2, 2], np.where(z < 0.25, 0.43, 0.33), rtol=1e-6)
    _, n = _read(res["SOIL_N"])
    assert n[:, 2, 2].min() >= 1.01  # clipped to the physical bound

    res = processing.run("thetaflow:run_spatial", {
        "DEM": rasters["dem"],
        "SOIL_DEPTH_MODE": 2,
        "REGRESSION_TABLE": ["ks", "exp", "0.2", ""],
        "WEATHER_SOURCE": 2,
        "WEATHER_SERIES": ["", "5", "0.1", "", "20", "0", "", "0", "0.2"],
        "DEFAULT_DT_HOURS": 0.5,
        "OUTPUT_FOLDER": str(tmp_path / "run_reg"),
        **_column(),
    })
    diag = pd.read_csv(res["DIAGNOSTICS"])
    np.testing.assert_allclose(diag["time_hours"], [0.5, 1.0, 1.5])


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _fake_openmeteo(calls):
    def fake_get(url, params, timeout):
        calls.append(params)
        lats = [float(v) for v in params["latitude"].split(",")]
        hours = 5
        times = pd.date_range("2026-09-20", periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
        payload = [
            {"hourly": {"time": times, "temperature_2m": [12.0] * hours,
                        "precipitation": [float(i + k) for k in range(hours)]}}
            for i, _ in enumerate(lats)
        ]
        return _Resp(payload if len(payload) > 1 else payload[0])
    return fake_get


def test_run_with_openmeteo_grid(processing, rasters, tmp_path, monkeypatch):
    import requests

    calls = []
    monkeypatch.setattr(requests, "get", _fake_openmeteo(calls))
    res = processing.run("thetaflow:run_spatial", {
        "DEM": rasters["dem"],
        "WEATHER_SOURCE": 4,
        "PAST_DAYS": 2,
        "FORECAST_DAYS": 3,
        "GRID_POINTS": 2,
        "OUTPUT_FOLDER": str(tmp_path / "run_om"),
        **_column(),
    })
    assert len(calls) == 1
    assert len(calls[0]["latitude"].split(",")) == 4
    assert calls[0]["past_days"] == 2 and calls[0]["forecast_days"] == 3
    lat = float(calls[0]["latitude"].split(",")[0])
    assert 54.0 < lat < 55.0  # BNG (400000, 500000) is in northern England
    points = pd.read_csv(res["FORCING"])
    assert set(points["point"]) == {0, 1, 2, 3}
    _, steps = _read(res["THETA_MEAN_TIMESERIES"])
    assert steps.shape[0] == 5


def test_fetch_weather_centroid(processing, tmp_path, monkeypatch):
    import requests

    calls = []
    monkeypatch.setattr(requests, "get", _fake_openmeteo(calls))
    out = tmp_path / "weather.csv"
    processing.run("thetaflow:fetch_weather", {
        "EXTENT": "-1.6,-1.4,54.7,54.8 [EPSG:4326]",
        "MODE": 0,
        "OUTPUT": str(out),
    })
    df = pd.read_csv(out)
    assert list(df.columns) == ["timestamp", "rainfall_mm_h", "pet_mm_h"]
    assert float(calls[0]["latitude"]) == pytest.approx(54.75)


def test_geographic_dem_is_rejected(processing, tmp_path):
    dem = _write(tmp_path / "geo.tif", np.ones((4, 4)), epsg=4326, gt=(-1.6, 0.001, 0, 54.8, 0, -0.001))
    with pytest.raises(QgsProcessingException, match="projected CRS"):
        processing.run("thetaflow:run_spatial", {
            "DEM": dem, "WEATHER_SOURCE": 2, "WEATHER_SERIES": ["", "1", "0"],
            "OUTPUT_FOLDER": str(tmp_path / "x"),
        })


def test_bad_horizon_name_is_reported(processing, rasters, tmp_path):
    with pytest.raises(QgsProcessingException, match="Cannot tell the property"):
        processing.run("thetaflow:prepare_inputs", {
            "DEM": rasters["dem"],
            "SOIL_DEPTH_MODE": 1,
            "HORIZON_LAYERS": [rasters["decay"]],
            "OUTPUT_FOLDER": str(tmp_path / "bad"),
        })
