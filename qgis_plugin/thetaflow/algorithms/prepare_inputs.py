import os

import numpy as np
from osgeo import gdal
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingOutputRasterLayer,
    QgsProcessingParameterFolderDestination,
)
from qgis.PyQt.QtGui import QIcon

from ..qgis_io import write_geotiff
from ..resources import ICON_PATH
from .common import VEG_COLOURS, ThetaFlowInputsMixin, load_on_completion

import soil_depth as sd  # noqa: E402
from spatial_utils import build_friction_factor_grid, load_vegetation_map  # noqa: E402

SOIL_RAMPS = {
    "theta_r": "Oranges",
    "theta_s": "Blues",
    "alpha_per_m": "Greens",
    "n": "Purples",
    "ks_m_per_s": "Viridis",
    "pore_connectivity": "Greys",
}


def _fallback_colour(i: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(1000 + i)
    return tuple(int(v) for v in rng.integers(40, 220, size=3))


class PrepareInputsAlgorithm(ThetaFlowInputsMixin, QgsProcessingAlgorithm):
    def name(self):
        return "prepare_inputs"

    def displayName(self):  # noqa: N802 (QGIS API name)
        return "Prepare / preview model inputs"

    def group(self):
        return "Simulation"

    def groupId(self):  # noqa: N802 (QGIS API name)
        return "simulation"

    def icon(self):
        return QIcon(ICON_PATH)

    def shortHelpString(self):  # noqa: N802 (QGIS API name)
        return (
            "Builds the soil and vegetation inputs exactly as 'Run ThetaFlow spatial simulation' "
            "would, and writes them as rasters on the DEM grid so they can be checked before a "
            "long run:\n"
            "• soil_<property>.tif – one band per soil layer (band descriptions give depths)\n"
            "• vegetation_class.tif – vegetation type per cell (labelled palette)\n"
            "• rooting_depth_m.tif, pet_scale.tif, friction_factor.tif\n"
            "Ks is written in m/s."
        )

    def createInstance(self):  # noqa: N802 (QGIS API name)
        return PrepareInputsAlgorithm()

    def initAlgorithm(self, config=None):  # noqa: N802 (QGIS API name)
        self.add_model_input_parameters()
        self.addParameter(QgsProcessingParameterFolderDestination("OUTPUT_FOLDER", "Output folder"))
        for prop in sd.SOIL_PROPERTIES:
            self.addOutput(QgsProcessingOutputRasterLayer(f"SOIL_{prop.upper()}", f"Soil {prop} by layer"))
        for key, label in (("VEGETATION_CLASS", "Vegetation type"), ("ROOTING_DEPTH", "Rooting depth (m)"),
                           ("PET_SCALE", "PET scale"), ("FRICTION_FACTOR", "Darcy–Weisbach friction factor")):
            self.addOutput(QgsProcessingOutputRasterLayer(key, label))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802 (QGIS API name)
        inputs = self.build_model_inputs(parameters, context, feedback)
        spec, mask = inputs.spec, inputs.valid
        out_dir = self.parameterAsString(parameters, "OUTPUT_FOLDER", context)
        os.makedirs(out_dir, exist_ok=True)
        outputs = {"OUTPUT_FOLDER": out_dir}

        layer_names = [f"layer {i + 1}: {z:.3f} m" for i, z in enumerate(inputs.depth_m)]
        for prop in sd.SOIL_PROPERTIES:
            key = f"SOIL_{prop.upper()}"
            path = write_geotiff(
                os.path.join(out_dir, f"soil_{prop}.tif"),
                np.moveaxis(inputs.soil_profiles[prop], 2, 0), spec, mask=mask,
                band_names=layer_names, dtype=gdal.GDT_Float64,
            )
            outputs[key] = path
            if prop in ("ks_m_per_s", "theta_s"):
                load_on_completion(context, path, f"Soil {prop} (layer 1)", key, ramp=SOIL_RAMPS[prop])

        codes = inputs.landcover_codes
        if codes is None:
            codes = np.full((spec.nrows, spec.ncols), -1, dtype=np.int32)
        veg_grid = load_vegetation_map(codes, inputs.veg_types, inputs.code_map,
                                       default_name=inputs.default_veg)
        names = [e["name"] for e in inputs.veg_types["vegetation_types"]]
        index = {n: i + 1 for i, n in enumerate(names)}
        veg_class = np.array([[index[v.name] for v in row] for row in veg_grid], dtype=float)
        rooting = np.array([[v.rooting_depth_m for v in row] for row in veg_grid], dtype=float)
        pet_scale = np.array([[v.pet_scale for v in row] for row in veg_grid], dtype=float)
        friction = build_friction_factor_grid(
            codes, inputs.veg_types, inputs.code_map,
            overrides=inputs.cfg.darcy_weisbach_f_overrides or None,
            default_name=inputs.default_veg,
        )

        used = sorted({int(v) for v in np.unique(veg_class[mask])})
        palette = [(i, VEG_COLOURS.get(names[i - 1], _fallback_colour(i)), names[i - 1]) for i in used]
        path = write_geotiff(
            os.path.join(out_dir, "vegetation_class.tif"), veg_class, spec, mask=mask,
            band_names=["vegetation type"], nodata=0,
            dtype=gdal.GDT_Byte,
            category_names=[""] + names,
        )
        outputs["VEGETATION_CLASS"] = path
        load_on_completion(context, path, "Vegetation type", "VEGETATION_CLASS", palette=palette)
        for key, fname, data, label, ramp in (
            ("ROOTING_DEPTH", "rooting_depth_m.tif", rooting, "Rooting depth (m)", "Greens"),
            ("PET_SCALE", "pet_scale.tif", pet_scale, "PET scale", "Oranges"),
            ("FRICTION_FACTOR", "friction_factor.tif", friction, "Darcy–Weisbach friction factor", "YlOrBr"),
        ):
            outputs[key] = write_geotiff(os.path.join(out_dir, fname), data, spec, mask=mask)
            load_on_completion(context, outputs[key], label, key, ramp=ramp)

        feedback.pushInfo(
            "Vegetation cells: " + ", ".join(
                f"{names[i - 1]}: {int(np.count_nonzero(veg_class[mask] == i))}" for i in used))
        return outputs
