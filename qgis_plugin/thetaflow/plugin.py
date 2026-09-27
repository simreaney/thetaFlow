from qgis.core import QgsApplication

from .provider import ThetaFlowProvider


class ThetaFlowPlugin:
    """Registers the ThetaFlow Processing provider."""

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):  # noqa: N802 (QGIS API name)
        self.provider = ThetaFlowProvider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):  # noqa: N802 (QGIS API name)
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
