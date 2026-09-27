#!/usr/bin/env python3
"""Open-Meteo weather forcing for thetaFlow spatial runs.

Two modes:

* **Centroid** – a single hourly series (recent past + forecast) at one
  point, applied uniformly to every cell.
* **Sampled grid** – an n × n set of points across the domain fetched in one
  Open-Meteo request and interpolated to cell centres by inverse-distance
  weighting.  The result behaves like the ``list[dict]`` returned by
  :func:`spatial_utils.read_gridded_forcing` but computes each step's grids
  on demand to keep memory low on large DEMs.

This module has no QGIS dependency.  Coordinate transformation between the
DEM CRS and WGS84 is left to the caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

from simulate_soil_column import _process_forcing_df, fetch_openmeteo_forcing_multi


def centroid_forcing(
    lat: float,
    lon: float,
    past_days: int = 7,
    forecast_days: int = 7,
    default_dt_hours: float = 1.0,
) -> pd.DataFrame:
    """Uniform forcing DataFrame (``timestamp``, ``rainfall_mm_h``, ``pet_mm_h``,
    ``dt_seconds``) for one point."""
    df = fetch_openmeteo_forcing_multi([lat], [lon], past_days, forecast_days)[0]
    return _process_forcing_df(df, default_dt_hours, source_name="Open-Meteo")


def sample_point_grid(
    xmin: float, ymin: float, xmax: float, ymax: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return x, y coordinates (each length n²) of an n × n cell-centred grid
    covering the given extent."""
    n = max(1, int(n))
    xs = xmin + (np.arange(n) + 0.5) * (xmax - xmin) / n
    ys = ymin + (np.arange(n) + 0.5) * (ymax - ymin) / n
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel()


def idw_weights(
    point_x: np.ndarray,
    point_y: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """Inverse-distance weights, shape (ncells, npoints), rows summing to 1.

    A cell that coincides with a sample point takes that point's value.
    """
    px = np.asarray(point_x, float)[None, :]
    py = np.asarray(point_y, float)[None, :]
    cx = np.asarray(cell_x, float).ravel()[:, None]
    cy = np.asarray(cell_y, float).ravel()[:, None]
    dist = np.hypot(cx - px, cy - py)
    exact = dist < 1.0e-9
    with np.errstate(divide="ignore"):
        w = np.where(exact, 0.0, 1.0 / np.maximum(dist, 1.0e-12) ** power)
    has_exact = exact.any(axis=1)
    w[has_exact] = exact[has_exact].astype(float)
    return w / w.sum(axis=1, keepdims=True)


def align_point_series(frames: Sequence[pd.DataFrame]) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Inner-join point series on timestamp.

    Returns ``(timestamps, rainfall (T, P), pet (T, P))``.
    """
    idx: Optional[pd.DatetimeIndex] = None
    for df in frames:
        ts = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
        idx = ts if idx is None else idx.intersection(ts)
    if idx is None or len(idx) == 0:
        raise ValueError("Point weather series share no common timestamps.")
    idx = idx.sort_values()
    rain = np.column_stack([
        df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
        .set_index("timestamp")["rainfall_mm_h"].reindex(idx).to_numpy(float)
        for df in frames
    ])
    pet = np.column_stack([
        df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))
        .set_index("timestamp")["pet_mm_h"].reindex(idx).to_numpy(float)
        for df in frames
    ])
    return idx, rain, pet


class InterpolatedForcing(Sequence):
    """Gridded forcing interpolated from point series on demand.

    Each item is a dict with ``step``, ``timestamp``, ``dt_seconds``,
    ``rainfall_grid`` and ``pet_grid`` (flattened, one value per cell) —
    the structure ``run_spatial_simulation`` expects for gridded forcing.
    """

    def __init__(
        self,
        timestamps: pd.DatetimeIndex,
        rainfall: np.ndarray,
        pet: np.ndarray,
        weights: np.ndarray,
        default_dt_hours: float = 1.0,
    ) -> None:
        self.timestamps = timestamps
        self.rainfall = np.asarray(rainfall, float)
        self.pet = np.asarray(pet, float)
        self.weights = np.asarray(weights, float)
        dt_df = _process_forcing_df(
            pd.DataFrame({"timestamp": timestamps, "rainfall_mm_h": 0.0, "pet_mm_h": 0.0}),
            default_dt_hours,
            source_name="Open-Meteo grid",
        )
        self.dt_seconds = dt_df["dt_seconds"].to_numpy(float)

    def __len__(self) -> int:
        return len(self.timestamps)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return {
            "step": i,
            "timestamp": self.timestamps[i],
            "dt_seconds": float(self.dt_seconds[i]),
            "rainfall_grid": np.maximum(self.weights @ self.rainfall[i], 0.0),
            "pet_grid": np.maximum(self.weights @ self.pet[i], 0.0),
        }

    def mean_series(self) -> pd.DataFrame:
        """Domain-mean forcing series (for reporting / saving)."""
        return pd.DataFrame(
            {
                "timestamp": self.timestamps,
                "rainfall_mm_h": (self.weights.mean(axis=0) @ self.rainfall.T),
                "pet_mm_h": (self.weights.mean(axis=0) @ self.pet.T),
            }
        )


def sampled_grid_forcing(
    point_lat: np.ndarray,
    point_lon: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    past_days: int = 7,
    forecast_days: int = 7,
    default_dt_hours: float = 1.0,
    frames: Optional[Sequence[pd.DataFrame]] = None,
) -> tuple[InterpolatedForcing, pd.DataFrame]:
    """Fetch point series and build IDW-interpolated gridded forcing.

    ``point_lat/lon`` are used for the Open-Meteo request; ``point_x/y`` and
    ``cell_x/y`` must share one projected CRS for the distance weighting.
    Pass *frames* to skip the network request (e.g. in tests).

    Returns the forcing sequence and a long-format DataFrame of the point
    series (``point``, ``lat``, ``lon``, ``timestamp``, ``rainfall_mm_h``,
    ``pet_mm_h``) for saving.
    """
    lats = [float(v) for v in np.ravel(point_lat)]
    lons = [float(v) for v in np.ravel(point_lon)]
    if frames is None:
        frames = fetch_openmeteo_forcing_multi(lats, lons, past_days, forecast_days)
    timestamps, rain, pet = align_point_series(frames)
    weights = idw_weights(point_x, point_y, cell_x, cell_y)
    forcing = InterpolatedForcing(timestamps, rain, pet, weights, default_dt_hours)
    points = pd.concat(
        [
            df.assign(point=i, lat=lat, lon=lon)[["point", "lat", "lon", "timestamp", "rainfall_mm_h", "pet_mm_h"]]
            for i, (df, lat, lon) in enumerate(zip(frames, lats, lons))
        ],
        ignore_index=True,
    )
    return forcing, points
