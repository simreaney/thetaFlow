# ThetaFlow QGIS plugin

A QGIS Processing provider that runs the thetaFlow 2-D model
(`simulate_spatial.py`) on a DEM. Each DEM cell is a 1-D Richards-equation soil
column. Subsurface throughflow and Darcy–Weisbach overland flow are routed
between cells with FD8.

After installation the **ThetaFlow** group in the Processing Toolbox contains
three algorithms:

| Algorithm | What it does |
|---|---|
| **Run ThetaFlow spatial simulation** | Builds the inputs, runs the model and loads the result rasters |
| **Prepare / preview model inputs** | Writes the per-layer soil properties and vegetation rasters exactly as the run would use them, so you can check them first |
| **Fetch Open-Meteo weather forcing** | Downloads recent past and forecast weather for an area to a CSV file |

Because the algorithms are Processing algorithms, they also work in batch
mode, in the Graphical Modeler, from the Python console
(`processing.run("thetaflow:run_spatial", {...})`) and from `qgis_process`.

## Installation

```bash
python scripts/build_qgis_plugin.py      # writes dist/thetaflow_qgis.zip
```

In QGIS, choose *Plugins → Manage and Install Plugins → Install from ZIP* and
select `dist/thetaflow_qgis.zip`. The plugin needs QGIS 3.22 or newer, plus
`numpy`, `pandas` and `requests`. The standard QGIS installers include all
three. If a Linux distribution package lacks pandas, install `python3-pandas`.

**For development**, link the source folder into your QGIS profile instead.
When it runs from the repository, the plugin uses the model modules in the
repository root.

```bash
ln -s "$PWD/qgis_plugin/thetaflow" ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/thetaflow
```

## Inputs

All rasters are warped onto the DEM grid in memory, so they can have any CRS,
resolution or extent. Categorical rasters (soil type, land cover) use
nearest-neighbour resampling. Continuous rasters (horizon and coefficient
rasters) use bilinear resampling.

### DEM

- The DEM must be in a **projected CRS in metres** with square cells.
- NoData cells are not simulated. Water that flows onto them leaves the domain.
- Every active cell is a full soil column, and runs take about 10 ms per cell per
  hourly step on one core. Keep grids to a few thousand cells, resampling the
  DEM if needed.
- The log reports the expected time remaining as the run proceeds.
- On Linux the *Worker processes* setting (advanced) spreads cells over
  several cores.

### Soil properties and how they change with depth

The base soil properties come from the **soil type raster** and the **soil types
JSON** (the same format as `soil_types_example.json`). Without a soil type
raster, every cell uses a default loam. *How soil properties change with depth*
then selects one of three modes.

**Uniform with depth.** Each cell keeps its base properties throughout the column.

**Horizon rasters.** Add rasters named `<property>_<top>-<bottom>cm` (or
`…m`), for example:

```
ks_0-5cm   ks_5-15cm   ks_15-30cm   theta_s_0-30cm   alpha_0-0.3m
```

- **Property names**:
  - `theta_r`
  - `theta_s` (or `porosity`)
  - `alpha`
  - `n`
  - `ks` (or `ksat`)
  - `l` (pore connectivity)
- **SoilGrids-style suffixes** such as `ks_0-5cm_mean` are accepted.
- **Horizon table**: instead of relying on layer names, you can give a CSV with
  the columns `top_m, bottom_m, property, raster_path`. Relative paths are
  resolved against the CSV's folder.
- **Ks units**: choose m/s, cm/day or mm/h for the Ks rasters.
- **Interpolation**:
  - *Step*: each layer takes the value of the horizon that contains it.
  - *Linear*: values are interpolated between horizon mid-depths.
- **Gaps and missing data**:
  - Layers below the deepest horizon take its values.
  - Any property without a horizon raster, and any NoData cell, keeps the base value.

**Regression with depth.** Each row of the regression table sets one property:

| form | p(z) | coefficient *a* |
|---|---|---|
| `exp` | p0 · exp(−z / a) | e-folding depth (m) |
| `linear` | p0 + a · z | change per metre |
| `power` | p0 · (1 + z)^a | exponent |
| `expr` | any expression | available as `a` |

- `z` is the layer depth (m).
- `p0` is the base (soil-map) value.
- The coefficient can be a number or the name of a raster added under
  *Rasters usable as coefficients / expression variables*, so decay depths can
  vary in space.
- Expressions can use `z`, `p0`, `a` and those raster names (non-alphanumeric
  characters become `_`).
- Expressions can use `+ - * / ** %` and comparisons, plus the functions `exp`,
  `log`, `log10`, `sqrt`, `abs`, `min`, `max`, `clip`, `where` and `tanh`.
  Nothing else is evaluated. Examples:

```
where(z < soil_depth, p0, p0 * 0.01)      # Ks drops below a soil-depth raster
p0 * exp(-z / a) + 1e-7                   # exponential decay with a floor
maximum(p0 - 0.15 * z, 0.3)               # linear decrease, limited to 0.3
```

Values outside the physical ranges are clipped, and the log reports how many:

- `0 ≤ θr < θs ≤ 1`
- `n ≥ 1.01`
- `α > 0`
- `1e-12 ≤ Ks ≤ 1 m/s`

### Land cover and vegetation

Land cover sets three things for each cell:

- the **rooting depth**, which controls root water uptake over the soil layers;
- the **PET scale**, which converts PET to plant water demand;
- the **Darcy–Weisbach friction factor** for overland flow.

These come from `vegetation_types.json`.

- **Land cover raster**: an integer-coded raster, resampled with nearest neighbour.
- **Code mapping**: a preset, or *Custom*. The presets are in `landcover_maps/`
  and the CLI's `--lc-map` accepts the same files:
  - *Built-in order (1–11)*: the CLI default.
  - *UKCEH Land Cover Map*: the 21 target classes.
  - *ESA WorldCover*.
  - *CORINE*: accepts both level-3 codes (111–523) and the Copernicus raster
    GRID_CODE values (1–44).

  *Custom* uses a JSON file (`{"code": "name"}`), a table typed into the dialog,
  or both.
- **Vegetation types**: the bundled types, plus an optional JSON file of your own
  (entries are added, or replace types by name). The *Vegetation type edits*
  table changes single values. For example, `broadleaf_woodland | 2.0 | |`
  makes woodland roots 2 m deep and leaves the other values unchanged.
- **Default vegetation**: used for NoData, for codes without a mapping, and for
  every cell when there is no land cover raster. The log lists how many cells
  used it.
- If a rooting depth is deeper than the soil column (layers × thickness), the
  log warns you.

Run *Prepare / preview model inputs* to check the mapping. It writes
`vegetation_class.tif` with a labelled palette, plus `rooting_depth_m.tif`,
`pet_scale.tif`, `friction_factor.tif` and one `soil_<property>.tif` per
property, with one band per layer.

### Weather

Rainfall and PET are in mm/h. The *Weather source* setting chooses between:

- **CSV file**: `timestamp, rainfall_mm_h, pet_mm_h`, the same format as
  `forcing_example.csv`. Without timestamps, each row lasts the default step
  length.
- **Table layer**: any table with those field names, such as a CSV or a
  GeoPackage table loaded in QGIS.
- **Series typed in the table**: short scenarios entered directly in the dialog.
- **Open-Meteo at the DEM centroid**: hourly *recent past* (up to 92 days) plus
  *forecast* (up to 16 days) from the Open-Meteo forecast API. No API key is
  needed. PET is estimated with Hargreaves–Samani from the temperature.
- **Open-Meteo on a sampled grid**: an n × n grid of points across the DEM
  extent, fetched in one request. Rainfall and PET are interpolated to each cell
  by inverse-distance weighting, which suits DEMs larger than one weather-model
  grid cell.

Every run writes the weather it used to `forcing_used.csv` in the output folder.
For grid runs, the file holds each point's series.

## Outputs

These rasters are on the DEM grid, with NoData outside the DEM:

| File | Bands |
|---|---|
| `theta_mean_final.tif` | column-mean volumetric water content at the end |
| `theta_layers_final.tif` | θ for each soil layer at the end (band description = depth) |
| `theta_mean_timeseries.tif` | column-mean θ for each step (band description = timestamp) |
| `max_flow_depth_m.tif` | maximum overland flow depth |
| `cumulative_runoff_mm.tif` | total infiltration-excess runoff generated |
| `cumulative_lateral_outflow_mm.tif` | total subsurface lateral outflow |

The output folder also holds:

- `spatial_diagnostics.csv`, the domain means for each step;
- `forcing_used.csv`;
- if *Also save per-step arrays and render MP4 animations* is on (it needs
  ffmpeg), the per-step `.npy` arrays and the MP4s.

## Tests

The model-side tests run under any Python. The plugin tests need the QGIS
Python bindings and run the algorithms in a headless QGIS:

```bash
python -m pytest tests                                   # QGIS tests skipped
QT_QPA_PLATFORM=offscreen python3 -m pytest tests        # with QGIS's Python
```
