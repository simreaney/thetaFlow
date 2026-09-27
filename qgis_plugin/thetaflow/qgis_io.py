"""GDAL raster helpers for the ThetaFlow plugin.

Every input raster is warped onto the DEM grid in memory, so inputs can have
any CRS, resolution or extent.  The model assumes square cells in metres.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

NODATA = -9999.0


@dataclass
class GridSpec:
    """The DEM grid that every input is aligned to."""

    nrows: int
    ncols: int
    geotransform: tuple
    crs_wkt: str

    @property
    def cell_size(self) -> float:
        return abs(float(self.geotransform[1]))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax)"""
        gt = self.geotransform
        x0, y0 = gt[0], gt[3]
        x1 = x0 + gt[1] * self.ncols
        y1 = y0 + gt[5] * self.nrows
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Row-major x, y arrays of cell-centre coordinates, shape (nrows, ncols)."""
        gt = self.geotransform
        cols = np.arange(self.ncols) + 0.5
        rows = np.arange(self.nrows) + 0.5
        cc, rr = np.meshgrid(cols, rows)
        return gt[0] + cc * gt[1] + rr * gt[2], gt[3] + cc * gt[4] + rr * gt[5]


def _open(source: str) -> gdal.Dataset:
    ds = gdal.Open(source)
    if ds is None:
        raise RuntimeError(f"Could not open raster '{source}'.")
    return ds


def read_dem(source: str) -> tuple[np.ndarray, np.ndarray, GridSpec]:
    """Read band 1 of a DEM. Returns (elevation, valid_mask, grid)."""
    ds = _open(source)
    gt = ds.GetGeoTransform()
    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12:
        raise ValueError("Rotated DEM grids are not supported; please warp the DEM first.")
    if not np.isclose(abs(gt[1]), abs(gt[5]), rtol=1e-3):
        raise ValueError(
            f"DEM cells must be square (found {abs(gt[1]):g} × {abs(gt[5]):g}); "
            "resample the DEM (e.g. with gdal:warpreproject) first."
        )
    band = ds.GetRasterBand(1)
    dem = band.ReadAsArray().astype(float)
    valid = np.isfinite(dem)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        valid &= ~np.isclose(dem, nodata)
    spec = GridSpec(ds.RasterYSize, ds.RasterXSize, tuple(gt), ds.GetProjection())
    return dem, valid, spec


def warp_to_grid(source: str, spec: GridSpec, categorical: bool = False, band: int = 1) -> np.ndarray:
    """Warp one band of *source* onto *spec*. NoData / outside cells are NaN.

    Categorical rasters (land cover, soil type) use nearest-neighbour
    resampling; continuous ones use bilinear.
    """
    src = _open(source)
    if src.GetProjection() == "" and spec.crs_wkt:
        raise ValueError(f"Raster '{source}' has no CRS; assign one before using it.")
    xmin, ymin, xmax, ymax = spec.bounds
    opts = gdal.WarpOptions(
        format="MEM",
        outputBounds=(xmin, ymin, xmax, ymax),
        width=spec.ncols,
        height=spec.nrows,
        dstSRS=spec.crs_wkt or None,
        resampleAlg="near" if categorical else "bilinear",
        outputType=gdal.GDT_Float64,
        dstNodata=np.nan,
        srcBands=[band],
        dstBands=[1],
    )
    out = gdal.Warp("", src, options=opts)
    return out.GetRasterBand(1).ReadAsArray().astype(float)


def write_geotiff(
    path: str,
    data: np.ndarray,
    spec: GridSpec,
    mask: Optional[np.ndarray] = None,
    band_names: Optional[Sequence[str]] = None,
    dtype=gdal.GDT_Float32,
    nodata: float = NODATA,
    color_table: Optional[dict[int, tuple[int, int, int]]] = None,
    category_names: Optional[Sequence[str]] = None,
) -> str:
    """Write a (nrows, ncols) or (bands, nrows, ncols) array as a GeoTIFF.

    Cells outside *mask* (and non-finite values) are written as *nodata*.
    """
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 2:
        arr = arr[None]
    nb = arr.shape[0]
    drv = gdal.GetDriverByName("GTiff")
    opts = ["COMPRESS=DEFLATE", "TILED=YES"]
    if nb > 1:
        opts.append("INTERLEAVE=BAND")
    ds = drv.Create(path, spec.ncols, spec.nrows, nb, dtype, options=opts)
    ds.SetGeoTransform(spec.geotransform)
    if spec.crs_wkt:
        ds.SetProjection(spec.crs_wkt)
    for i in range(nb):
        b = arr[i].copy()
        bad = ~np.isfinite(b)
        if mask is not None:
            bad |= ~mask
        b[bad] = nodata
        band = ds.GetRasterBand(i + 1)
        band.SetNoDataValue(nodata)
        band.WriteArray(b)
        if band_names is not None:
            band.SetDescription(str(band_names[i]))
        if color_table is not None:
            ct = gdal.ColorTable()
            for value, rgb in color_table.items():
                ct.SetColorEntry(int(value), tuple(rgb) + (255,))
            band.SetRasterColorTable(ct)
            band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
        if category_names is not None:
            band.SetRasterCategoryNames(list(category_names))
    ds.FlushCache()
    ds = None
    return path


def is_metric_projected(crs_wkt: str) -> bool:
    """True if the CRS is projected with linear units of metres."""
    if not crs_wkt:
        return False
    srs = osr.SpatialReference()
    srs.ImportFromWkt(crs_wkt)
    return bool(srs.IsProjected()) and abs(srs.GetLinearUnits() - 1.0) < 1e-9
