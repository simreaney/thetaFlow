"""ThetaFlow QGIS plugin: Processing algorithms for the thetaFlow
Richards-equation hillslope model."""

import importlib
import os
import sys


def _core_dir() -> str:
    """Directory holding the thetaFlow model modules.

    A packaged plugin carries them in ``core/`` (see
    ``scripts/build_qgis_plugin.py``); a development checkout, where the
    plugin folder is linked into the QGIS profile, uses the repository root.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    core = os.path.join(here, "core")
    if os.path.isfile(os.path.join(core, "simulate_spatial.py")):
        return core
    repo = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    if os.path.isfile(os.path.join(repo, "simulate_spatial.py")):
        return repo
    return core


CORE_DIR = _core_dir()
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
# A reinstall in the same QGIS session can leave a stale "no importer" entry
# for CORE_DIR (cached while it did not exist), hiding the modules now in it.
importlib.invalidate_caches()


def classFactory(iface):  # noqa: N802 (QGIS API name)
    if not os.path.isfile(os.path.join(CORE_DIR, "simulate_spatial.py")):
        raise ImportError(
            f"ThetaFlow model modules not found in {CORE_DIR}. Install the "
            "plugin from the ZIP built by scripts/build_qgis_plugin.py "
            "(dist/thetaflow_qgis.zip), not a ZIP of qgis_plugin/thetaflow."
        )
    from .plugin import ThetaFlowPlugin

    return ThetaFlowPlugin(iface)
