from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .algorithms.fetch_weather import FetchWeatherAlgorithm
from .algorithms.prepare_inputs import PrepareInputsAlgorithm
from .algorithms.run_spatial import RunSpatialAlgorithm
from .resources import ICON_PATH


class ThetaFlowProvider(QgsProcessingProvider):
    def loadAlgorithms(self):  # noqa: N802 (QGIS API name)
        self.addAlgorithm(RunSpatialAlgorithm())
        self.addAlgorithm(PrepareInputsAlgorithm())
        self.addAlgorithm(FetchWeatherAlgorithm())

    def id(self):
        return "thetaflow"

    def name(self):
        return "ThetaFlow"

    def icon(self):
        return QIcon(ICON_PATH)

    def longName(self):  # noqa: N802 (QGIS API name)
        return "ThetaFlow soil moisture model"
