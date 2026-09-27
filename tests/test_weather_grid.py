import numpy as np
import pandas as pd
import pytest

import simulate_soil_column
from weather_grid import (
    align_point_series,
    centroid_forcing,
    idw_weights,
    sample_point_grid,
    sampled_grid_forcing,
)


def _payload(lat, precip_scale, hours=6, start="2026-09-20T00:00"):
    times = pd.date_range(start, periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "latitude": lat,
        "hourly": {
            "time": times,
            "temperature_2m": [10.0] * hours,
            "precipitation": [precip_scale * i for i in range(hours)],
        },
    }


class _Resp:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_sample_point_grid_is_cell_centred():
    x, y = sample_point_grid(0, 0, 30, 60, 3)
    assert sorted(set(x)) == [5, 15, 25]
    assert sorted(set(y)) == [10, 30, 50]


def test_idw_weights_rows_sum_to_one_and_exact_points():
    px, py = np.array([0.0, 10.0]), np.array([0.0, 0.0])
    w = idw_weights(px, py, np.array([0.0, 5.0, 7.5]), np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    np.testing.assert_allclose(w[0], [1.0, 0.0])
    np.testing.assert_allclose(w[1], [0.5, 0.5])
    assert w[2, 1] > w[2, 0]


def test_align_point_series_inner_join():
    a = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
                      "rainfall_mm_h": [1, 2, 3, 4], "pet_mm_h": 0.1})
    b = a.iloc[1:].assign(rainfall_mm_h=[20, 30, 40])
    ts, rain, pet = align_point_series([a, b])
    assert len(ts) == 3
    np.testing.assert_allclose(rain, [[2, 20], [3, 30], [4, 40]])


def test_multi_fetch_and_grid_interpolation(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        lats = params["latitude"].split(",")
        return _Resp([_payload(float(lat), i + 1.0) for i, lat in enumerate(lats)])

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    forcing, points = sampled_grid_forcing(
        point_lat=[54.0, 54.1], point_lon=[-1.5, -1.5],
        point_x=[0.0, 100.0], point_y=[0.0, 0.0],
        cell_x=np.array([0.0, 50.0, 100.0]), cell_y=np.zeros(3),
        past_days=1, forecast_days=1,
    )
    assert len(calls) == 1 and calls[0]["past_days"] == 1
    assert len(forcing) == 6
    rec = forcing[3]
    np.testing.assert_allclose(rec["rainfall_grid"], [3.0, 4.5, 6.0])
    assert rec["dt_seconds"] == pytest.approx(3600.0)
    assert rec["pet_grid"].shape == (3,) and np.all(rec["pet_grid"] >= 0)
    assert set(points["point"]) == {0, 1}
    assert len(list(forcing)) == 6  # iterable like a list of dicts


def test_centroid_forcing(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda url, params, timeout: _Resp(_payload(54.0, 1.0)))
    df = centroid_forcing(54.0, -1.5, past_days=2, forecast_days=3)
    assert list(df["rainfall_mm_h"]) == [0, 1, 2, 3, 4, 5]
    assert (df["dt_seconds"] == 3600.0).all()
    assert df["timestamp"].dt.tz is not None
