"""Parameters and input assembly shared by the ThetaFlow Processing algorithms."""

import io
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qgis.core import (
    QgsColorRampShader,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsPalettedRasterRenderer,
    QgsPointXY,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingLayerPostProcessorInterface,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterMatrix,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProcessing,
    QgsRasterBandStats,
    QgsSingleBandPseudoColorRenderer,
    QgsStyle,
)
from qgis.PyQt.QtGui import QColor

from .. import CORE_DIR
from ..qgis_io import GridSpec, is_metric_projected, read_dem, warp_to_grid

import soil_depth as sd  # noqa: E402  (core module on sys.path, see __init__)
from simulate_soil_column import ColumnConfig, SoilProperties, build_column_geometry  # noqa: E402
from simulate_spatial import SpatialConfig, load_spatial_config  # noqa: E402
from spatial_utils import (  # noqa: E402
    fill_nodata_nearest,
    load_code_map,
    load_soil_map,
    load_vegetation_types,
)

DATA_DIR = Path(CORE_DIR)
DEFAULT_VEG_JSON = DATA_DIR / "vegetation_types.json"
DEFAULT_SOIL_JSON = DATA_DIR / "soil_types_example.json"
LANDCOVER_DIR = DATA_DIR / "landcover_maps"

# Same default as simulate_spatial.run_spatial_simulation when no soil map is given
DEFAULT_LOAM = SoilProperties(
    theta_r=0.078, theta_s=0.43, alpha_per_m=3.6, n=1.56, ks_m_per_s=2.89e-6, pore_connectivity=0.5
)

SOIL_MODES = [
    "Uniform with depth (soil map or default loam)",
    "Horizon rasters",
    "Regression with depth",
]
SOIL_UNIFORM, SOIL_HORIZONS, SOIL_REGRESSION = range(3)

LC_PRESETS = [
    ("Built-in order (codes 1–11)", "builtin_order.json"),
    ("UKCEH Land Cover Map (21 classes)", "ukceh_lcm.json"),
    ("ESA WorldCover", "esa_worldcover.json"),
    ("CORINE Land Cover (codes or GRID_CODE)", "corine.json"),
    ("Custom (code map file and/or table)", None),
]

KS_UNITS = list(sd.KS_UNIT_FACTORS)
LARGE_GRID_CELLS = 5_000

VEG_COLOURS = {
    "grass": (140, 200, 90),
    "broadleaf_woodland": (30, 120, 40),
    "coniferous_woodland": (20, 80, 50),
    "moorland": (150, 110, 160),
    "wheat": (230, 210, 110),
    "maize": (240, 180, 60),
    "oilseed_rape": (250, 240, 60),
    "sugar_beet": (200, 150, 90),
    "potato": (180, 130, 80),
    "bare_soil": (170, 140, 110),
    "urban": (120, 120, 120),
    "shrubland": (110, 150, 70),
    "open_water": (60, 120, 220),
}


@dataclass
class ModelInputs:
    """Everything the model needs, aligned to the DEM grid."""

    dem: np.ndarray                     # NoData filled, for routing
    valid: np.ndarray                   # DEM data mask
    spec: GridSpec
    cfg: SpatialConfig
    depth_m: np.ndarray
    soil_list: list                     # per-cell SoilProperties (row-major)
    soil_profiles: dict                 # {prop: (nrows, ncols, nz)}
    landcover_codes: Optional[np.ndarray]
    code_map: dict
    veg_types: dict                     # {"vegetation_types": [...]}
    default_veg: str
    centroid_lonlat: tuple


class FeedbackStream(io.TextIOBase):
    """File-like object that forwards model ``print`` output to Processing feedback."""

    def __init__(self, feedback):
        self.feedback = feedback
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.feedback.pushConsoleInfo(line)
        return len(text)

    def flush(self):
        if self._buf.strip():
            self.feedback.pushConsoleInfo(self._buf)
        self._buf = ""


def _cell_text(value) -> str:
    """Text of a matrix cell; empty for None and QVariant NULL."""
    if value is None or (hasattr(value, "isNull") and value.isNull()):
        return ""
    text = str(value).strip()
    return "" if text == "NULL" else text


def matrix_rows(values, ncols: int) -> list[list[str]]:
    """Split a flat Processing matrix value into rows, dropping empty rows."""
    values = [_cell_text(v) for v in (values or [])]
    rows = [values[i:i + ncols] for i in range(0, len(values), ncols)]
    return [r + [""] * (ncols - len(r)) for r in rows if any(r)]


def safe_variable_name(name: str) -> str:
    ident = re.sub(r"\W", "_", name.strip())
    return ident if ident and not ident[0].isdigit() else f"r_{ident}"


def to_wgs84(spec_crs_wkt: str, xs, ys, context) -> tuple[np.ndarray, np.ndarray]:
    """Transform coordinates from the DEM CRS to lon/lat (EPSG:4326)."""
    src = QgsCoordinateReferenceSystem.fromWkt(spec_crs_wkt)
    dst = QgsCoordinateReferenceSystem("EPSG:4326")
    tr = QgsCoordinateTransform(src, dst, context.transformContext())
    pts = [tr.transform(QgsPointXY(float(x), float(y))) for x, y in zip(np.ravel(xs), np.ravel(ys))]
    return np.array([p.x() for p in pts]), np.array([p.y() for p in pts])


def _advanced(param):
    param.setFlags(param.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
    return param


# ---------------------------------------------------------------------------
# Styling of loaded outputs
# ---------------------------------------------------------------------------

_POST_PROCESSORS: list = []  # keep references alive until QGIS calls them


class _PseudocolorStyler(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, ramp_name: str, invert: bool = False):
        super().__init__()
        self.ramp_name = ramp_name
        self.invert = invert

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802 (QGIS API name)
        provider = layer.dataProvider()
        stats = provider.bandStatistics(1, QgsRasterBandStats.Min | QgsRasterBandStats.Max)
        ramp = QgsStyle.defaultStyle().colorRamp(self.ramp_name)
        if ramp is None:
            return
        if self.invert:
            ramp.invert()
        renderer = QgsSingleBandPseudoColorRenderer(provider, 1)
        lo, hi = stats.minimumValue, stats.maximumValue
        if hi <= lo:
            hi = lo + 1e-9
        renderer.setClassificationMin(lo)
        renderer.setClassificationMax(hi)
        renderer.createShader(ramp, QgsColorRampShader.Interpolated, QgsColorRampShader.Continuous, 5)
        layer.setRenderer(renderer)
        layer.triggerRepaint()


class _PalettedStyler(QgsProcessingLayerPostProcessorInterface):
    def __init__(self, classes: list[tuple[int, tuple[int, int, int], str]]):
        super().__init__()
        self.classes = classes

    def postProcessLayer(self, layer, context, feedback):  # noqa: N802 (QGIS API name)
        classes = [
            QgsPalettedRasterRenderer.Class(value, QColor(*rgb), label)
            for value, rgb, label in self.classes
        ]
        layer.setRenderer(QgsPalettedRasterRenderer(layer.dataProvider(), 1, classes))
        layer.triggerRepaint()


def load_on_completion(context, path: str, name: str, output_name: str,
                       ramp: Optional[str] = None, invert: bool = False,
                       palette: Optional[list] = None) -> None:
    details = QgsProcessingContext.LayerDetails(name, context.project(), output_name)
    styler = None
    if palette is not None:
        styler = _PalettedStyler(palette)
    elif ramp is not None:
        styler = _PseudocolorStyler(ramp, invert)
    if styler is not None:
        _POST_PROCESSORS.append(styler)
        details.setPostProcessor(styler)
    context.addLayerToLoadOnCompletion(path, details)


# ---------------------------------------------------------------------------
# Shared parameters and input assembly
# ---------------------------------------------------------------------------

class ThetaFlowInputsMixin:
    """Adds terrain, soil, land-cover and column parameters to an algorithm and
    turns them into :class:`ModelInputs`."""

    def add_model_input_parameters(self):
        # Terrain ------------------------------------------------------------
        self.addParameter(QgsProcessingParameterRasterLayer(
            "DEM", "DEM (projected CRS in metres, square cells)"))

        # Soil ---------------------------------------------------------------
        self.addParameter(QgsProcessingParameterRasterLayer(
            "SOIL_MAP", "Soil type raster (integer codes)", optional=True))
        self.addParameter(QgsProcessingParameterFile(
            "SOIL_TYPES", "Soil types JSON (code → van Genuchten parameters)",
            extension="json", optional=True, defaultValue=str(DEFAULT_SOIL_JSON)))
        self.addParameter(QgsProcessingParameterEnum(
            "SOIL_DEPTH_MODE", "How soil properties change with depth",
            options=SOIL_MODES, defaultValue=SOIL_UNIFORM))
        self.addParameter(QgsProcessingParameterMultipleLayers(
            "HORIZON_LAYERS",
            "Horizon rasters, named <property>_<top>-<bottom>cm, e.g. ks_0-5cm, theta_s_15-30cm",
            layerType=QgsProcessing.TypeRaster, optional=True))
        self.addParameter(QgsProcessingParameterFile(
            "HORIZON_TABLE", "Horizon table CSV (top_m, bottom_m, property, raster_path) [optional]",
            extension="csv", optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            "KS_UNIT", "Units of Ks horizon rasters", options=KS_UNITS, defaultValue=0))
        self.addParameter(QgsProcessingParameterEnum(
            "HORIZON_INTERP", "Horizon interpolation between depths",
            options=["Step (value of containing horizon)", "Linear between horizon mid-depths"],
            defaultValue=0))
        self.addParameter(QgsProcessingParameterMatrix(
            "REGRESSION_TABLE",
            "Depth regressions: form = exp p0·exp(−z/a) | linear p0+a·z | power p0·(1+z)^a | expr",
            headers=["property", "form", "coefficient a (number or raster name)", "expression (for expr)"],
            hasFixedNumberRows=False, numberRows=1,
            defaultValue=["ks", "exp", "0.5", ""], optional=True))
        self.addParameter(QgsProcessingParameterMultipleLayers(
            "REGRESSION_VARIABLES", "Rasters usable as coefficients / expression variables (by layer name)",
            layerType=QgsProcessing.TypeRaster, optional=True))

        # Land cover ---------------------------------------------------------
        self.addParameter(QgsProcessingParameterRasterLayer(
            "LANDCOVER", "Land cover raster (integer codes)", optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            "LC_PRESET", "Land cover code mapping",
            options=[label for label, _ in LC_PRESETS], defaultValue=0))
        self.addParameter(QgsProcessingParameterFile(
            "LC_MAP_FILE", 'Custom code map JSON ({"code": "vegetation name"})',
            extension="json", optional=True))
        self.addParameter(QgsProcessingParameterMatrix(
            "LC_MAP_TABLE", "Custom code map table (added to / overrides the file)",
            headers=["code", "vegetation name"], hasFixedNumberRows=False, numberRows=1, optional=True))
        self.addParameter(QgsProcessingParameterFile(
            "VEG_TYPES", "Vegetation types JSON (adds to / overrides the bundled types)",
            extension="json", optional=True))
        self.addParameter(QgsProcessingParameterMatrix(
            "VEG_TABLE", "Vegetation type edits (blank cells keep existing values)",
            headers=["name", "rooting_depth_m", "pet_scale", "darcy_weisbach_f"],
            hasFixedNumberRows=False, numberRows=1, optional=True))
        self.addParameter(QgsProcessingParameterString(
            "DEFAULT_VEG", "Default vegetation (no land cover, unmapped codes, NoData)",
            defaultValue="grass"))

        # Column / numerics --------------------------------------------------
        self.addParameter(QgsProcessingParameterNumber(
            "NZ", "Number of soil layers", QgsProcessingParameterNumber.Integer,
            defaultValue=50, minValue=2, maxValue=1000))
        self.addParameter(QgsProcessingParameterNumber(
            "DZ", "Layer thickness (m)", QgsProcessingParameterNumber.Double,
            defaultValue=0.02, minValue=0.001, maxValue=5.0))
        self.addParameter(QgsProcessingParameterNumber(
            "INITIAL_HEAD", "Initial pressure head (m)", QgsProcessingParameterNumber.Double,
            defaultValue=-1.0))
        self.addParameter(_advanced(QgsProcessingParameterNumber(
            "MAX_SUBSTEP", "Maximum sub-step (s)", QgsProcessingParameterNumber.Double,
            defaultValue=60.0, minValue=1.0)))
        self.addParameter(_advanced(QgsProcessingParameterNumber(
            "FD8_EXPONENT", "FD8 exponent", QgsProcessingParameterNumber.Double,
            defaultValue=1.1, minValue=0.1)))
        self.addParameter(_advanced(QgsProcessingParameterFile(
            "CONFIG", "Spatial config JSON (overrides the column and numerics settings above)",
            extension="json", optional=True)))

    # ------------------------------------------------------------------
    def build_model_inputs(self, parameters, context, feedback) -> ModelInputs:
        dem_layer = self.parameterAsRasterLayer(parameters, "DEM", context)
        if dem_layer is None:
            raise QgsProcessingException("A DEM raster is required.")
        dem, valid, spec = read_dem(dem_layer.source())
        if not is_metric_projected(spec.crs_wkt):
            raise QgsProcessingException(
                "The DEM must use a projected CRS in metres (e.g. British National Grid, "
                "a UTM zone). Reproject it first, e.g. with 'Warp (reproject)'.")
        if not valid.any():
            raise QgsProcessingException("The DEM has no valid cells.")
        dem_filled = fill_nodata_nearest(dem, valid) if not valid.all() else dem
        ncells = spec.nrows * spec.ncols
        n_active = int(valid.sum())
        feedback.pushInfo(
            f"DEM grid {spec.nrows} × {spec.ncols}, cell size {spec.cell_size:g} m, "
            f"{n_active} active cells.")
        if n_active > LARGE_GRID_CELLS:
            feedback.pushWarning(
                f"{n_active} active cells: every cell is a Richards-equation column, so this "
                "run may be slow. Consider resampling the DEM to a coarser cell size.")

        cfg = self._build_config(parameters, context, spec)
        _, depth_m = build_column_geometry(cfg.column.nz, cfg.column.dz_m)
        feedback.pushInfo(
            f"Soil column: {cfg.column.nz} layers × {cfg.column.dz_m:g} m = "
            f"{cfg.column.nz * cfg.column.dz_m:g} m deep.")

        base_list = self._base_soil_list(parameters, context, feedback, spec, valid)
        shape = (spec.nrows, spec.ncols)
        base_grids = sd.base_grids_from_soil_list(base_list, shape)
        mode = self.parameterAsEnum(parameters, "SOIL_DEPTH_MODE", context)
        if mode == SOIL_HORIZONS:
            profiles = self._horizon_profiles(parameters, context, feedback, spec, depth_m, base_grids)
        elif mode == SOIL_REGRESSION:
            profiles = self._regression_profiles(parameters, context, feedback, spec, depth_m, base_grids)
        else:
            profiles = sd.profiles_from_regression([], depth_m, base_grids)

        if mode == SOIL_UNIFORM:
            soil_list = base_list
        else:
            profiles, report = sd.validate_profiles(profiles, base_grids)
            changed = {k: v for k, v in report.items() if v}
            if changed:
                feedback.pushWarning(
                    "Soil profile values outside physical bounds were clipped: "
                    + ", ".join(f"{k}: {v}" for k, v in changed.items()))
            soil_list = sd.soil_list_from_profiles(profiles)
            for prop in sd.SOIL_PROPERTIES:
                p = profiles[prop][valid]
                feedback.pushInfo(
                    f"  {prop}: surface {np.median(p[:, 0]):.4g}, bottom {np.median(p[:, -1]):.4g} (median)")

        veg_types = self._vegetation_types(parameters, context)
        veg_names = [e["name"] for e in veg_types["vegetation_types"]]
        default_veg = self.parameterAsString(parameters, "DEFAULT_VEG", context).strip()
        if default_veg not in veg_names:
            raise QgsProcessingException(
                f"Default vegetation '{default_veg}' is not a vegetation type. "
                f"Available: {', '.join(veg_names)}.")
        code_map = self._code_map(parameters, context)
        unknown = sorted({n for n in code_map.values() if n not in veg_names})
        if unknown:
            feedback.pushWarning(
                f"Code map names not in the vegetation types (default used instead): {', '.join(unknown)}")

        lc_layer = self.parameterAsRasterLayer(parameters, "LANDCOVER", context)
        landcover_codes = None
        used_names = {default_veg}
        if lc_layer is not None:
            lc = warp_to_grid(lc_layer.source(), spec, categorical=True)
            landcover_codes = np.where(np.isfinite(lc), lc, -1).astype(np.int32)
            codes, counts = np.unique(landcover_codes[valid], return_counts=True)
            unmapped = [(int(c), int(n)) for c, n in zip(codes, counts)
                        if code_map.get(int(c)) not in veg_names]
            used_names |= {code_map[int(c)] for c in codes if code_map.get(int(c)) in veg_names}
            if unmapped:
                feedback.pushWarning(
                    f"Cells using default vegetation '{default_veg}' (code: cells): "
                    + ", ".join(f"{'NoData' if c < 0 else c}: {n}" for c, n in unmapped))
            feedback.pushInfo("Vegetation types in use: " + ", ".join(sorted(used_names)))

        lib = load_vegetation_types(veg_types)
        max_root = max(lib[n].rooting_depth_m for n in used_names)
        column_depth = cfg.column.nz * cfg.column.dz_m
        if max_root > column_depth:
            feedback.pushWarning(
                f"Deepest rooting depth in use ({max_root:g} m) exceeds the soil column depth "
                f"({column_depth:g} m); root uptake will be truncated. Increase layers or thickness.")

        xs, ys = spec.cell_centres()
        lon, lat = to_wgs84(spec.crs_wkt, [xs.mean()], [ys.mean()], context)
        cfg.simulation.latitude, cfg.simulation.longitude = float(lat[0]), float(lon[0])

        return ModelInputs(
            dem=dem_filled, valid=valid, spec=spec, cfg=cfg, depth_m=depth_m,
            soil_list=soil_list, soil_profiles=profiles, landcover_codes=landcover_codes,
            code_map=code_map, veg_types=veg_types, default_veg=default_veg,
            centroid_lonlat=(float(lon[0]), float(lat[0])),
        )

    # ------------------------------------------------------------------
    def _build_config(self, parameters, context, spec: GridSpec) -> SpatialConfig:
        cfg_path = self.parameterAsFile(parameters, "CONFIG", context)
        if cfg_path:
            cfg = load_spatial_config(Path(cfg_path))
        else:
            cfg = SpatialConfig()
            cfg.column = ColumnConfig(
                nz=self.parameterAsInt(parameters, "NZ", context),
                dz_m=self.parameterAsDouble(parameters, "DZ", context),
            )
            cfg.simulation.initial_head_m = self.parameterAsDouble(parameters, "INITIAL_HEAD", context)
            cfg.simulation.max_substep_seconds = self.parameterAsDouble(parameters, "MAX_SUBSTEP", context)
            cfg.fd8_exponent = self.parameterAsDouble(parameters, "FD8_EXPONENT", context)
        cfg.cell_size_m = spec.cell_size
        return cfg

    def _base_soil_list(self, parameters, context, feedback, spec, valid) -> list:
        ncells = spec.nrows * spec.ncols
        layer = self.parameterAsRasterLayer(parameters, "SOIL_MAP", context)
        if layer is None:
            feedback.pushInfo("No soil map: using the default loam for every cell.")
            return [DEFAULT_LOAM] * ncells
        soil_json = self.parameterAsFile(parameters, "SOIL_TYPES", context) or str(DEFAULT_SOIL_JSON)
        raw = warp_to_grid(layer.source(), spec, categorical=True)
        codes = np.where(np.isfinite(raw), raw, -1).astype(np.int32)
        grid = load_soil_map(codes, Path(soil_json))
        soil_list = [grid[r][c] for r in range(spec.nrows) for c in range(spec.ncols)]
        missing = np.array([sp is None for sp in soil_list]).reshape(codes.shape) & valid
        if missing.all():
            raise QgsProcessingException(
                "No soil map codes match the soil types JSON. Check the soil map and JSON codes.")
        if missing.any():
            bad = sorted(set(int(c) for c in codes[missing]))
            fallback = next(sp for sp in soil_list if sp is not None)
            feedback.pushWarning(
                f"{int(missing.sum())} cells have soil codes missing from the JSON "
                f"({', '.join('NoData' if c < 0 else str(c) for c in bad)}); "
                "they use the first mapped soil type.")
            soil_list = [sp if sp is not None else fallback for sp in soil_list]
        return soil_list

    def _horizon_profiles(self, parameters, context, feedback, spec, depth_m, base_grids):
        ks_factor = sd.KS_UNIT_FACTORS[KS_UNITS[self.parameterAsEnum(parameters, "KS_UNIT", context)]]
        items = []
        table = self.parameterAsFile(parameters, "HORIZON_TABLE", context)
        if table:
            df = pd.read_csv(table)
            required = {"top_m", "bottom_m", "property", "raster_path"}
            if not required.issubset(df.columns):
                raise QgsProcessingException(
                    f"Horizon table needs columns {sorted(required)}; found {list(df.columns)}.")
            base_dir = os.path.dirname(os.path.abspath(table))
            for row in df.itertuples(index=False):
                path = str(row.raster_path)
                if not os.path.isabs(path):
                    path = os.path.join(base_dir, path)
                items.append((sd.canonical_property(str(row.property)), float(row.top_m),
                              float(row.bottom_m), warp_to_grid(path, spec)))
        for layer in self.parameterAsLayerList(parameters, "HORIZON_LAYERS", context):
            parsed = sd.parse_horizon_name(layer.name()) or sd.parse_horizon_name(
                Path(layer.source().split("|")[0]).stem)
            if parsed is None:
                raise QgsProcessingException(
                    f"Cannot tell the property and depth of layer '{layer.name()}'. Name horizon "
                    "layers like ks_0-5cm, theta_s_15-30cm or alpha_0-0.3m, or use a horizon table.")
            items.append((*parsed, warp_to_grid(layer.source(), spec)))
        if not items:
            raise QgsProcessingException(
                "Horizon mode needs horizon rasters or a horizon table.")
        items = [(p, t, b, g * ks_factor if p == "ks_m_per_s" else g) for p, t, b, g in items]
        horizons = sd.group_horizon_grids(items)
        for hz in horizons:
            feedback.pushInfo(
                f"Horizon {hz.top_m:g}–{hz.bottom_m:g} m: {', '.join(sorted(hz.grids))}")
        interp = "linear" if self.parameterAsEnum(parameters, "HORIZON_INTERP", context) == 1 else "step"
        try:
            return sd.profiles_from_horizons(horizons, depth_m, base_grids, interp=interp)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

    def _regression_profiles(self, parameters, context, feedback, spec, depth_m, base_grids):
        layers = self.parameterAsLayerList(parameters, "REGRESSION_VARIABLES", context)
        variables = {safe_variable_name(l.name()): warp_to_grid(l.source(), spec) for l in layers}
        if variables:
            feedback.pushInfo("Expression variables: " + ", ".join(sorted(variables)))
        rules = []
        for prop, form, coef, expr in matrix_rows(
                self.parameterAsMatrix(parameters, "REGRESSION_TABLE", context), 4):
            if not prop:
                continue
            if coef == "":
                value = 0.0
            else:
                try:
                    value = float(coef)
                except ValueError:
                    key = safe_variable_name(coef)
                    if key in variables:
                        value = variables[key]
                    elif os.path.exists(coef):
                        value = warp_to_grid(coef, spec)
                    else:
                        raise QgsProcessingException(
                            f"Coefficient '{coef}' for {prop} is not a number, a variable layer "
                            "name or a raster path.")
            try:
                rules.append(sd.RegressionRule(prop, form or "exp", value, expr))
            except ValueError as exc:
                raise QgsProcessingException(str(exc)) from exc
            feedback.pushInfo(f"Regression {rules[-1].prop}: {rules[-1].form} "
                              f"{expr if rules[-1].form == 'expr' else f'a = {coef}'}")
        if not rules:
            raise QgsProcessingException("Regression mode needs at least one row in the regression table.")
        try:
            return sd.profiles_from_regression(rules, depth_m, base_grids, variables)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

    def _vegetation_types(self, parameters, context) -> dict:
        with DEFAULT_VEG_JSON.open("r", encoding="utf-8") as f:
            entries = {e["name"]: dict(e) for e in json.load(f)["vegetation_types"]}
        user = self.parameterAsFile(parameters, "VEG_TYPES", context)
        if user:
            with open(user, "r", encoding="utf-8") as f:
                for e in json.load(f)["vegetation_types"]:
                    entries.setdefault(e["name"], {"name": e["name"]}).update(e)
        fields = ["rooting_depth_m", "pet_scale", "darcy_weisbach_f"]
        for row in matrix_rows(self.parameterAsMatrix(parameters, "VEG_TABLE", context), 4):
            name = row[0]
            if not name:
                continue
            entry = entries.setdefault(name, {"name": name, "rooting_depth_m": 0.0,
                                              "pet_scale": 1.0, "darcy_weisbach_f": 0.5})
            for key, value in zip(fields, row[1:]):
                if value != "":
                    try:
                        entry[key] = float(value)
                    except ValueError as exc:
                        raise QgsProcessingException(
                            f"Vegetation table: {key} for '{name}' must be a number.") from exc
        return {"vegetation_types": list(entries.values())}

    def _code_map(self, parameters, context) -> dict:
        _, preset_file = LC_PRESETS[self.parameterAsEnum(parameters, "LC_PRESET", context)]
        mapping: dict = {}
        if preset_file is not None:
            mapping.update(load_code_map(LANDCOVER_DIR / preset_file))
        custom = self.parameterAsFile(parameters, "LC_MAP_FILE", context)
        if custom:
            mapping.update(load_code_map(Path(custom)))
        for code, name in matrix_rows(self.parameterAsMatrix(parameters, "LC_MAP_TABLE", context), 2):
            try:
                mapping[int(float(code))] = name
            except ValueError as exc:
                raise QgsProcessingException(f"Land cover code '{code}' is not an integer.") from exc
        return mapping


def worker_count(requested: int, feedback) -> int:
    """Multiprocessing inside QGIS only works where child processes fork."""
    if requested > 1 and not sys.platform.startswith("linux"):
        feedback.pushWarning(
            "Parallel workers are only supported on Linux inside QGIS; running in a single process.")
        return 1
    return max(1, requested)


def add_worker_parameters(alg):
    alg.addParameter(_advanced(QgsProcessingParameterNumber(
        "WORKERS", "Worker processes (Linux only; 1 = run inside QGIS)",
        QgsProcessingParameterNumber.Integer, defaultValue=1, minValue=1, maxValue=256)))
    alg.addParameter(_advanced(QgsProcessingParameterBoolean(
        "ANIMATIONS", "Also save per-step arrays and render MP4 animations (needs ffmpeg)",
        defaultValue=False)))
