import os
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingOutputFile,
    QgsProcessingOutputRasterLayer,
    QgsProcessingParameterFolderDestination,
)
from qgis.PyQt.QtGui import QIcon

from ..resources import ICON_PATH
from ..qgis_io import write_geotiff
from .common import (
    FeedbackStream,
    ThetaFlowInputsMixin,
    add_worker_parameters,
    load_on_completion,
    worker_count,
)
from .weather import add_weather_parameters, load_weather

from simulate_spatial import run_spatial_simulation  # noqa: E402

RASTER_OUTPUTS = [
    # key, file name, layer name, colour ramp, invert
    ("THETA_MEAN_FINAL", "theta_mean_final.tif", "Mean soil moisture (final)", "Blues", False),
    ("THETA_LAYERS_FINAL", "theta_layers_final.tif", "Soil moisture by layer (final)", "Blues", False),
    ("THETA_MEAN_TIMESERIES", "theta_mean_timeseries.tif", "Mean soil moisture per step", "Blues", False),
    ("MAX_FLOW_DEPTH", "max_flow_depth_m.tif", "Maximum overland flow depth (m)", "Viridis", False),
    ("CUMULATIVE_RUNOFF", "cumulative_runoff_mm.tif", "Cumulative runoff (mm)", "Reds", False),
    ("CUMULATIVE_LATERAL", "cumulative_lateral_outflow_mm.tif",
     "Cumulative subsurface lateral outflow (mm)", "Purples", False),
]


class RunSpatialAlgorithm(ThetaFlowInputsMixin, QgsProcessingAlgorithm):
    def name(self):
        return "run_spatial"

    def displayName(self):  # noqa: N802 (QGIS API name)
        return "Run ThetaFlow spatial simulation"

    def group(self):
        return "Simulation"

    def groupId(self):  # noqa: N802 (QGIS API name)
        return "simulation"

    def icon(self):
        return QIcon(ICON_PATH)

    def shortHelpString(self):  # noqa: N802 (QGIS API name)
        return (
            "Runs the thetaFlow model over a DEM. Each DEM cell is a 1-D Richards-equation soil "
            "column. Subsurface throughflow and Darcy–Weisbach overland flow are routed between "
            "cells with FD8.\n\n"
            "<b>Soil with depth</b>: uniform (soil map / default loam), horizon rasters named "
            "<i>property_top-bottomcm</i> (e.g. ks_0-5cm), or regressions: exp p0·exp(−z/a), "
            "linear p0+a·z, power p0·(1+z)^a, or an expression in z, p0, a and named rasters, "
            "e.g. <i>where(z &lt; soil_depth, p0, p0*0.01)</i>. p0 is the soil-map value.\n\n"
            "<b>Land cover</b> sets rooting depth, PET scaling and surface roughness through "
            "the vegetation types; presets map UKCEH LCM, ESA WorldCover and CORINE codes.\n\n"
            "<b>Weather</b>: CSV, table layer, a series typed into the dialog, or Open-Meteo "
            "recent past + forecast (at the centroid, or on a grid of points interpolated to "
            "cells). Rainfall and PET are in mm/h.\n\n"
            "Outputs are GeoTIFFs on the DEM grid plus spatial_diagnostics.csv and "
            "forcing_used.csv in the output folder."
        )

    def createInstance(self):  # noqa: N802 (QGIS API name)
        return RunSpatialAlgorithm()

    def initAlgorithm(self, config=None):  # noqa: N802 (QGIS API name)
        self.add_model_input_parameters()
        add_weather_parameters(self)
        add_worker_parameters(self)
        self.addParameter(QgsProcessingParameterFolderDestination("OUTPUT_FOLDER", "Output folder"))
        for key, _, label, _, _ in RASTER_OUTPUTS:
            self.addOutput(QgsProcessingOutputRasterLayer(key, label))
        self.addOutput(QgsProcessingOutputFile("DIAGNOSTICS", "Diagnostics CSV"))
        self.addOutput(QgsProcessingOutputFile("FORCING", "Weather forcing used (CSV)"))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802 (QGIS API name)
        inputs = self.build_model_inputs(parameters, context, feedback)
        if feedback.isCanceled():
            return {}
        forcing_uniform, forcing_gridded, forcing_table = load_weather(
            self, parameters, context, feedback, inputs)

        out_dir = self.parameterAsString(parameters, "OUTPUT_FOLDER", context)
        os.makedirs(out_dir, exist_ok=True)
        forcing_path = os.path.join(out_dir, "forcing_used.csv")
        forcing_table.drop(columns=[c for c in ("dt_seconds",) if c in forcing_table]).to_csv(
            forcing_path, index=False)

        animations = self.parameterAsBoolean(parameters, "ANIMATIONS", context)
        workers = worker_count(self.parameterAsInt(parameters, "WORKERS", context), feedback)
        cfg = inputs.cfg
        cfg.save_npy_arrays = animations

        started = time.monotonic()
        next_report = [1]

        def progress(step, n_steps):
            feedback.setProgress(100.0 * step / max(n_steps, 1))
            if step >= next_report[0] and step < n_steps:
                per_step = (time.monotonic() - started) / step
                feedback.pushInfo(
                    f"Step {step}/{n_steps}: about {_duration(per_step * (n_steps - step))} remaining.")
                next_report[0] = step + max(1, n_steps // 10)
            return not feedback.isCanceled()

        stream = FeedbackStream(feedback)
        try:
            with redirect_stdout(stream):
                result = run_spatial_simulation(
                    dem=inputs.dem,
                    landcover_codes=inputs.landcover_codes,
                    soil_codes=None,
                    veg_json_path=inputs.veg_types,
                    soil_types_json_path=None,
                    code_to_veg_name=inputs.code_map,
                    forcing_uniform=forcing_uniform,
                    forcing_gridded=forcing_gridded,
                    cfg=cfg,
                    output_dir=out_dir,
                    n_workers=workers,
                    soil_list_override=inputs.soil_list,
                    default_vegetation=inputs.default_veg,
                    active_mask=inputs.valid,
                    progress_callback=progress,
                    render_animations=animations,
                )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            raise QgsProcessingException(f"Simulation failed: {exc}") from exc
        finally:
            stream.flush()

        if result["n_steps"] == 0:
            raise QgsProcessingException("The simulation was cancelled before the first step finished.")
        if result["cancelled"]:
            feedback.pushWarning(
                f"Cancelled after {result['n_steps']} steps; writing results so far.")

        spec, mask = inputs.spec, inputs.valid
        step_names = [_step_label(i, ts, result["diagnostics"]) for i, ts in enumerate(result["timestamps"])]
        layer_names = [f"layer {i + 1}: {z:.3f} m" for i, z in enumerate(result["depth_m"])]
        arrays = {
            "THETA_MEAN_FINAL": (result["theta_mean_final"], None),
            "THETA_LAYERS_FINAL": (np.moveaxis(result["theta_layers_final"], 2, 0), layer_names),
            "THETA_MEAN_TIMESERIES": (result["theta_mean_steps"], step_names),
            "MAX_FLOW_DEPTH": (result["max_flow_depth_m"], None),
            "CUMULATIVE_RUNOFF": (result["cumulative_runoff_mm"], None),
            "CUMULATIVE_LATERAL": (result["cumulative_lateral_out_mm"], None),
        }
        outputs = {}
        for key, filename, label, ramp, invert in RASTER_OUTPUTS:
            data, names = arrays[key]
            path = write_geotiff(os.path.join(out_dir, filename), data, spec, mask=mask, band_names=names)
            outputs[key] = path
            load_on_completion(context, path, label, key, ramp=ramp, invert=invert)

        outputs["DIAGNOSTICS"] = os.path.join(out_dir, "spatial_diagnostics.csv")
        outputs["FORCING"] = forcing_path
        outputs["OUTPUT_FOLDER"] = out_dir
        return outputs


def _step_label(i, timestamp, diagnostics: pd.DataFrame) -> str:
    if timestamp is not None and not pd.isna(timestamp):
        return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M UTC")
    return f"step {i} (t = {diagnostics['time_hours'].iloc[i]:g} h)"


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min" if hours else f"{minutes} min {secs:02d} s"
