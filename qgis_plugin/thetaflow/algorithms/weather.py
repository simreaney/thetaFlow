"""Weather-forcing parameters and loaders for the ThetaFlow algorithms."""

from pathlib import Path

import numpy as np
import pandas as pd
from qgis.core import (
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterMatrix,
    QgsProcessingParameterNumber,
    QgsProcessingParameterVectorLayer,
)
from qgis.PyQt.QtCore import QDate, QDateTime, Qt

from .common import matrix_rows, to_wgs84

from simulate_soil_column import _process_forcing_df  # noqa: E402
from spatial_utils import read_uniform_forcing  # noqa: E402
from weather_grid import centroid_forcing, sample_point_grid, sampled_grid_forcing  # noqa: E402

WEATHER_SOURCES = [
    "CSV file (timestamp, rainfall_mm_h, pet_mm_h)",
    "Table layer (timestamp, rainfall_mm_h, pet_mm_h fields)",
    "Series typed in the table below",
    "Open-Meteo: recent past + forecast at the DEM centroid",
    "Open-Meteo: recent past + forecast on a sampled grid (interpolated)",
]
W_CSV, W_TABLE, W_MATRIX, W_OM_CENTROID, W_OM_GRID = range(5)


def add_weather_parameters(alg):
    alg.addParameter(QgsProcessingParameterEnum(
        "WEATHER_SOURCE", "Weather source", options=WEATHER_SOURCES, defaultValue=W_CSV))
    alg.addParameter(QgsProcessingParameterFile(
        "WEATHER_CSV", "Weather CSV", extension="csv", optional=True))
    alg.addParameter(QgsProcessingParameterVectorLayer(
        "WEATHER_TABLE", "Weather table layer", types=[QgsProcessing.TypeVector], optional=True))
    alg.addParameter(QgsProcessingParameterMatrix(
        "WEATHER_SERIES", "Weather series (timestamp optional; blank uses the default step length)",
        headers=["timestamp (YYYY-MM-DD HH:MM)", "rainfall_mm_h", "pet_mm_h"],
        hasFixedNumberRows=False, numberRows=3, optional=True))
    alg.addParameter(QgsProcessingParameterNumber(
        "DEFAULT_DT_HOURS", "Default step length when no timestamps are given (hours)",
        QgsProcessingParameterNumber.Double, defaultValue=1.0, minValue=0.001))
    alg.addParameter(QgsProcessingParameterNumber(
        "PAST_DAYS", "Open-Meteo: days of recent past weather", QgsProcessingParameterNumber.Integer,
        defaultValue=7, minValue=0, maxValue=92))
    alg.addParameter(QgsProcessingParameterNumber(
        "FORECAST_DAYS", "Open-Meteo: forecast days", QgsProcessingParameterNumber.Integer,
        defaultValue=7, minValue=0, maxValue=16))
    alg.addParameter(QgsProcessingParameterNumber(
        "GRID_POINTS", "Open-Meteo grid: points per side (n × n requests)",
        QgsProcessingParameterNumber.Integer, defaultValue=3, minValue=2, maxValue=10))


def _to_text(value):
    if isinstance(value, QDateTime):
        return value.toString(Qt.ISODate)
    if isinstance(value, QDate):
        return value.toString(Qt.ISODate)
    return None if value is None else str(value)


def _openmeteo_error(exc) -> QgsProcessingException:
    return QgsProcessingException(
        f"Open-Meteo request failed: {exc}. Check the internet connection and QGIS proxy "
        "settings, or supply the weather as a CSV file instead.")


def load_weather(alg, parameters, context, feedback, inputs):
    """Return ``(forcing_uniform, forcing_gridded, table_to_save)``."""
    source = alg.parameterAsEnum(parameters, "WEATHER_SOURCE", context)
    dt_hours = alg.parameterAsDouble(parameters, "DEFAULT_DT_HOURS", context)
    past = alg.parameterAsInt(parameters, "PAST_DAYS", context)
    forecast = alg.parameterAsInt(parameters, "FORECAST_DAYS", context)

    if source == W_CSV:
        path = alg.parameterAsFile(parameters, "WEATHER_CSV", context)
        if not path:
            raise QgsProcessingException("Choose a weather CSV file.")
        try:
            df = read_uniform_forcing(Path(path), dt_hours)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        return df, None, df

    if source == W_TABLE:
        layer = alg.parameterAsVectorLayer(parameters, "WEATHER_TABLE", context)
        if layer is None:
            raise QgsProcessingException("Choose a weather table layer.")
        names = [f.name() for f in layer.fields()]
        missing = {"rainfall_mm_h", "pet_mm_h"}.difference(names)
        if missing:
            raise QgsProcessingException(f"Weather table is missing fields: {sorted(missing)}.")
        cols = [c for c in ("timestamp", "dt_hours", "rainfall_mm_h", "pet_mm_h") if c in names]
        rows = [{c: feat[c] for c in cols} for feat in layer.getFeatures()]
        df = pd.DataFrame(rows)
        if "timestamp" in df:
            df["timestamp"] = [_to_text(v) for v in df["timestamp"]]
            df = df.sort_values("timestamp").reset_index(drop=True)
        for c in ("rainfall_mm_h", "pet_mm_h", "dt_hours"):
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return _checked(df, dt_hours, layer.name())

    if source == W_MATRIX:
        rows = matrix_rows(alg.parameterAsMatrix(parameters, "WEATHER_SERIES", context), 3)
        if not rows:
            raise QgsProcessingException("The weather series table is empty.")
        try:
            df = pd.DataFrame({
                "timestamp": [r[0] or None for r in rows],
                "rainfall_mm_h": [float(r[1] or 0.0) for r in rows],
                "pet_mm_h": [float(r[2] or 0.0) for r in rows],
            })
        except ValueError as exc:
            raise QgsProcessingException(f"Weather series values must be numbers: {exc}") from exc
        if df["timestamp"].isna().all():
            df = df.drop(columns="timestamp")
        elif df["timestamp"].isna().any():
            raise QgsProcessingException("Give a timestamp for every row of the weather series, or none.")
        return _checked(df, dt_hours, "weather series")

    if past + forecast <= 0:
        raise QgsProcessingException("Request at least one day of past or forecast weather.")
    import requests

    if source == W_OM_CENTROID:
        lon, lat = inputs.centroid_lonlat
        feedback.pushInfo(f"Fetching Open-Meteo weather at {lat:.4f}°N, {lon:.4f}°E …")
        try:
            df = centroid_forcing(lat, lon, past, forecast, dt_hours)
        except requests.RequestException as exc:
            raise _openmeteo_error(exc) from exc
        return df, None, df

    n = alg.parameterAsInt(parameters, "GRID_POINTS", context)
    spec = inputs.spec
    px, py = sample_point_grid(*spec.bounds, n)
    plon, plat = to_wgs84(spec.crs_wkt, px, py, context)
    cx, cy = spec.cell_centres()
    feedback.pushInfo(f"Fetching Open-Meteo weather for a {n} × {n} grid of points …")
    try:
        forcing, points = sampled_grid_forcing(
            plat, plon, px, py, cx.ravel(), cy.ravel(), past, forecast, dt_hours)
    except requests.RequestException as exc:
        raise _openmeteo_error(exc) from exc
    feedback.pushInfo(
        f"{len(forcing)} hourly steps; domain-mean rainfall "
        f"{float(np.mean(forcing.rainfall)):.3f} mm/h.")
    return None, forcing, points


def _checked(df, dt_hours, name):
    try:
        df = _process_forcing_df(df, dt_hours, source_name=name)
    except ValueError as exc:
        raise QgsProcessingException(str(exc)) from exc
    return df, None, df
