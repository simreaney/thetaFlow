#!/usr/bin/env python3
"""Package the ThetaFlow QGIS plugin as an installable ZIP.

Copies ``qgis_plugin/thetaflow`` plus the model modules and data files it
needs (into the plugin's ``core/`` folder) to ``dist/build/thetaflow`` and
zips it as ``dist/thetaflow_qgis.zip`` for *Plugins → Manage and Install
Plugins → Install from ZIP*.

Usage::

    python scripts/build_qgis_plugin.py
"""

from __future__ import annotations

import configparser
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_SRC = ROOT / "qgis_plugin" / "thetaflow"
DIST = ROOT / "dist"

CORE_FILES = [
    "simulate_soil_column.py",
    "simulate_spatial.py",
    "spatial_utils.py",
    "spatial_visualise.py",
    "soil_depth.py",
    "weather_grid.py",
    "vegetation_types.json",
    "soil_types_example.json",
    "spatial_config_example.json",
]
CORE_DIRS = ["landcover_maps"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


def build() -> Path:
    build_dir = DIST / "build" / "thetaflow"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(PLUGIN_SRC, build_dir, ignore=IGNORE)

    core = build_dir / "core"
    core.mkdir(exist_ok=True)
    for name in CORE_FILES:
        shutil.copy2(ROOT / name, core / name)
    for name in CORE_DIRS:
        shutil.copytree(ROOT / name, core / name, ignore=IGNORE)
    shutil.copy2(ROOT / "LICENSE", build_dir / "LICENSE")

    meta = configparser.ConfigParser()
    meta.read(build_dir / "metadata.txt", encoding="utf-8")
    version = meta["general"]["version"]

    zip_path = DIST / "thetaflow_qgis.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(build_dir.parent))
    print(f"Built ThetaFlow plugin {version}: {zip_path}")
    return zip_path


if __name__ == "__main__":
    build()
