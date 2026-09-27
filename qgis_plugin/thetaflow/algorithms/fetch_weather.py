import requests
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterExtent,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
)
from qgis.PyQt.QtGui import QIcon

from ..resources import ICON_PATH

import pandas as pd  # noqa: E402
from simulate_soil_column import fetch_openmeteo_forcing_multi  # noqa: E402
from weather_grid import sample_point_grid  # noqa: E402


class FetchWeatherAlgorithm(QgsProcessingAlgorithm):
    def name(self):
        return "fetch_weather"

    def displayName(self):  # noqa: N802 (QGIS API name)
        return "Fetch Open-Meteo weather forcing"

    def group(self):
        return "Weather"

    def groupId(self):  # noqa: N802 (QGIS API name)
        return "weather"

    def icon(self):
        return QIcon(ICON_PATH)

    def shortHelpString(self):  # noqa: N802 (QGIS API name)
        return (
            "Downloads hourly rainfall and temperature for the recent past and the forecast "
            "from Open-Meteo (no API key needed) and estimates PET with Hargreaves–Samani.\n\n"
            "• Centroid: one series at the centre of the extent, written as a CSV that can be "
            "used directly as the weather CSV of the simulation.\n"
            "• Grid: an n × n set of points across the extent, written in long format "
            "(point, lat, lon, timestamp, rainfall_mm_h, pet_mm_h) for inspection or reuse."
        )

    def createInstance(self):  # noqa: N802 (QGIS API name)
        return FetchWeatherAlgorithm()

    def initAlgorithm(self, config=None):  # noqa: N802 (QGIS API name)
        self.addParameter(QgsProcessingParameterExtent("EXTENT", "Area"))
        self.addParameter(QgsProcessingParameterEnum(
            "MODE", "Sampling", options=["Centroid (single series)", "Grid of points"], defaultValue=0))
        self.addParameter(QgsProcessingParameterNumber(
            "PAST_DAYS", "Days of recent past weather", QgsProcessingParameterNumber.Integer,
            defaultValue=7, minValue=0, maxValue=92))
        self.addParameter(QgsProcessingParameterNumber(
            "FORECAST_DAYS", "Forecast days", QgsProcessingParameterNumber.Integer,
            defaultValue=7, minValue=0, maxValue=16))
        self.addParameter(QgsProcessingParameterNumber(
            "GRID_POINTS", "Grid points per side", QgsProcessingParameterNumber.Integer,
            defaultValue=3, minValue=2, maxValue=10))
        self.addParameter(QgsProcessingParameterFileDestination(
            "OUTPUT", "Weather CSV", fileFilter="CSV files (*.csv)"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802 (QGIS API name)
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        rect = self.parameterAsExtent(parameters, "EXTENT", context, wgs84)
        past = self.parameterAsInt(parameters, "PAST_DAYS", context)
        forecast = self.parameterAsInt(parameters, "FORECAST_DAYS", context)
        if past + forecast <= 0:
            raise QgsProcessingException("Request at least one day of past or forecast weather.")
        grid = self.parameterAsEnum(parameters, "MODE", context) == 1
        if grid:
            n = self.parameterAsInt(parameters, "GRID_POINTS", context)
            lons, lats = sample_point_grid(rect.xMinimum(), rect.yMinimum(),
                                           rect.xMaximum(), rect.yMaximum(), n)
        else:
            c = rect.center()
            lons, lats = [c.x()], [c.y()]
        lats, lons = [float(v) for v in lats], [float(v) for v in lons]
        feedback.pushInfo(f"Requesting {len(lats)} point(s) from Open-Meteo …")
        try:
            frames = fetch_openmeteo_forcing_multi(lats, lons, past, forecast)
        except requests.RequestException as exc:
            raise QgsProcessingException(
                f"Open-Meteo request failed: {exc}. Check the internet connection and "
                "QGIS proxy settings.") from exc

        if grid:
            df = pd.concat(
                [f.assign(point=i, lat=la, lon=lo)[["point", "lat", "lon", "timestamp",
                                                     "rainfall_mm_h", "pet_mm_h"]]
                 for i, (f, la, lo) in enumerate(zip(frames, lats, lons))],
                ignore_index=True)
        else:
            df = frames[0]
        out = self.parameterAsFileOutput(parameters, "OUTPUT", context)
        df.to_csv(out, index=False)
        feedback.pushInfo(
            f"Wrote {len(frames[0])} hourly steps "
            f"({frames[0]['timestamp'].min()} – {frames[0]['timestamp'].max()}).")
        return {"OUTPUT": out}
