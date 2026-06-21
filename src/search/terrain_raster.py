# =============================================================================
# terrain_raster.py
# -----------------------------------------------------------------------------
# Responsible for: A real-data TerrainProvider — accessibility, visibility, and
#                  context sampled from the gathered DEM (slope) and ESA WorldCover
#                  (land cover) rasters, onto the shared grid.
# Role in project: Geo Unit 5. A drop-in for SyntheticTerrain behind the same
#                  TerrainProvider Protocol (terrain.py): the brain/demo construct
#                  RasterTerrain instead, and the prior + visibility become real.
#                  SyntheticTerrain stays as the no-dependency fallback.
# Assumptions: Both rasters are in a geographic CRS (lat/lon), so we sample by
#              cell-center lat/lon directly. WorldCover is huge (36000^2), so we read
#              only the AOI window. The DEM doesn't fully cover the AOI (east edge);
#              missing DEM -> no slope penalty (WorldCover still applies there).
# Corridor (C): NOT from data yet — OSM ingestion is the deferred sub-step 5b, so
#              corridor is left flat (1.0). The prior is then D x A (distance x access).
# =============================================================================

from __future__ import annotations

import math
import pathlib
from typing import Any, Dict, Tuple

import numpy as np
import rasterio
from rasterio.transform import rowcol
from rasterio.windows import Window

from src.common.config import BrainConfig
from src.common.grid import GridSpec
from src.search.terrain import TerrainLayers

# Repo-root-relative default raster locations (gathered into data/terrain/).
_DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "terrain"
_DEFAULT_DEM = _DATA / "dem_marin_usgs10m.tif"
_DEFAULT_WORLDCOVER = _DATA / "worldcover_2021_N36W123.tif"

# Slope -> accessibility (degrees). Flat is fully passable; a cliff is impassable.
_SLOPE_FULL_BLOCK_DEG = 60.0   # slope at which the linear accessibility would reach 0
_SLOPE_IMPASSABLE_DEG = 45.0   # at/above this, hard-zero (a cliff; non-compensatory)

# ESA WorldCover class -> accessibility multiplier A in [0, 1] (0 = impassable).
# Legend: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 70 snow,
#         80 water, 90 wetland, 95 mangrove, 100 moss.
_WORLDCOVER_ACCESS: Dict[int, float] = {
    10: 0.70, 20: 0.80, 30: 0.95, 40: 0.90, 50: 0.30, 60: 0.90,
    70: 0.50, 80: 0.00, 90: 0.30, 95: 0.30, 100: 0.80,
}
# ESA WorldCover class -> base (daylight) visibility in [0, 1] (canopy lowers it).
_WORLDCOVER_VISIBILITY: Dict[int, float] = {
    10: 0.40, 20: 0.70, 30: 0.95, 40: 0.90, 50: 0.70, 60: 0.98,
    70: 0.95, 80: 0.90, 90: 0.70, 95: 0.40, 100: 0.90,
}
_WORLDCOVER_LABEL: Dict[int, str] = {
    10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland",
    50: "built-up", 60: "bare/sparse", 70: "snow/ice", 80: "water",
    90: "wetland", 95: "mangrove", 100: "moss/lichen",
}
_DEFAULT_ACCESS = 0.50      # unknown land-cover class
_DEFAULT_VISIBILITY = 0.70


def cell_center_lonlat(grid: GridSpec) -> Tuple[np.ndarray, np.ndarray]:
    """
    The (lon, lat) of every cell center as two (n_rows, n_cols) arrays.

    Args:
        grid: The shared GridSpec.

    Returns:
        (LON, LAT) meshgrids of cell-center longitudes/latitudes.

    Why:
        The rasters are in lat/lon, so sampling is "where is each cell center on the
        raster?" Vectorizing the GridSpec.cell_to_latlon math here lets us sample all
        25,600 cells in one indexed read instead of per-cell calls.
    """
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(grid.origin[0]))
    rows = np.arange(grid.n_rows)
    cols = np.arange(grid.n_cols)
    lat = grid.origin[0] + ((rows + 0.5) * grid.cell_size_m) / m_per_deg_lat
    lon = grid.origin[1] + ((cols + 0.5) * grid.cell_size_m) / m_per_deg_lon
    return np.meshgrid(lon, lat)


def sample_raster_to_grid(path: str, grid: GridSpec) -> np.ndarray:
    """
    Sample a (lat/lon) raster at each grid cell center (nearest pixel).

    Args:
        path: Raster file path.
        grid: The shared GridSpec.

    Returns:
        (n_rows, n_cols) float array of sampled values; np.nan where a cell falls
        outside the raster or hits the raster's nodata value.

    Why:
        Pixel indices are computed against the FULL raster transform (so they match
        rasterio's own ds.sample exactly), then we read only the BOUNDING WINDOW of those
        pixels — essential for WorldCover (1.3 GB whole) and unambiguous (no margin/offset
        rounding to misalign the window). Nearest-neighbor is required for categorical land
        cover and fine for the DEM. nan-marking keeps "no data here" explicit so the caller
        fills it sensibly, rather than a silent sentinel like -999999 or 0.
    """
    lon, lat = cell_center_lonlat(grid)
    with rasterio.open(path) as ds:
        rows, cols = rowcol(ds.transform, lon.ravel(), lat.ravel())  # pixel index per cell
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        out_of_bounds = (rows < 0) | (rows >= ds.height) | (cols < 0) | (cols >= ds.width)

        # Read just the bounding window of the (clipped) needed pixels.
        rclip = np.clip(rows, 0, ds.height - 1)
        cclip = np.clip(cols, 0, ds.width - 1)
        r0, r1 = int(rclip.min()), int(rclip.max()) + 1
        c0, c1 = int(cclip.min()), int(cclip.max()) + 1
        data = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))

        sampled = data[rclip - r0, cclip - c0].astype(float).reshape(lat.shape)
        sampled[out_of_bounds.reshape(lat.shape)] = np.nan
        if ds.nodata is not None:
            sampled[sampled == ds.nodata] = np.nan
    return sampled


def _slope_deg_from_dem(dem: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    Per-cell slope in degrees from a sampled elevation grid.

    Args:
        dem: (n_rows, n_cols) elevation in meters (may contain nan).
        cell_size_m: Grid cell size, for the gradient spacing.

    Returns:
        (n_rows, n_cols) slope in degrees (0 where the DEM was missing).

    Why:
        Accessibility falls with steepness. np.gradient on the elevation array gives the
        rise/run; arctan converts to an angle. nan (off-DEM cells) are filled with the
        median first so the gradient is defined, then their slope is zeroed so they get
        no slope penalty (we simply don't know the terrain there).
    """
    missing = np.isnan(dem)
    filled = np.where(missing, np.nanmedian(dem), dem)
    grad_north, grad_east = np.gradient(filled, cell_size_m)
    slope = np.degrees(np.arctan(np.hypot(grad_east, grad_north)))
    slope[missing] = 0.0
    return slope


def _slope_to_accessibility(slope_deg: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Map slope (degrees) to an accessibility factor in [0, 1].

    Args:
        slope_deg: Per-cell slope.
        epsilon: The hard-but-passable floor from config.

    Returns:
        Accessibility from slope: ~1 flat, decreasing with steepness, 0 at a cliff.

    Why:
        A steep slope is hard to traverse (low A); past _SLOPE_IMPASSABLE_DEG it is a
        cliff and gets a hard zero (non-compensatory, like the prior's other zeros).
    """
    access = np.clip(1.0 - slope_deg / _SLOPE_FULL_BLOCK_DEG, epsilon, 1.0)
    access[slope_deg >= _SLOPE_IMPASSABLE_DEG] = 0.0
    return access


def _reclass(worldcover: np.ndarray, table: Dict[int, float], default: float) -> np.ndarray:
    """
    Map WorldCover class codes to values via a lookup table.

    Args:
        worldcover: (n_rows, n_cols) class codes (may contain nan).
        table: class -> value.
        default: value for unknown/nan classes.

    Returns:
        (n_rows, n_cols) float array of mapped values.

    Why:
        Land cover is categorical; accessibility and visibility are per-class lookups.
        One reclass helper keeps both mappings consistent (DRY).
    """
    out = np.full(worldcover.shape, default, dtype=float)
    for code, value in table.items():
        out[worldcover == code] = value
    return out


class RasterTerrain:
    """
    A TerrainProvider backed by the real DEM + WorldCover rasters.

    Why:
        Makes the prior and visibility reflect actual Marin geography (Mt. Tam's slopes
        and canopy) instead of a hand-drawn stub — the credibility a showcase needs.
        Implements the same Protocol as SyntheticTerrain, so it is a drop-in. Sampling
        is done once and cached (the rasters don't change).
    """

    def __init__(self, cfg: BrainConfig, dem_path: pathlib.Path = _DEFAULT_DEM,
                 worldcover_path: pathlib.Path = _DEFAULT_WORLDCOVER) -> None:
        """
        Args:
            cfg: Brain config (accessibility epsilon).
            dem_path / worldcover_path: Raster locations (default: data/terrain/).
        Why:
            Fails loudly if the rasters are absent so the caller can fall back to
            SyntheticTerrain explicitly, rather than silently producing an empty map.
        """
        for p in (dem_path, worldcover_path):
            if not pathlib.Path(p).exists():
                raise FileNotFoundError(f"RasterTerrain: missing raster {p}")
        self._cfg = cfg
        self._dem_path = str(dem_path)
        self._worldcover_path = str(worldcover_path)
        self._cache: Dict[tuple, dict] = {}

    def _key(self, grid: GridSpec) -> tuple:
        return (grid.n_rows, grid.n_cols, grid.origin, grid.cell_size_m)

    def _sampled(self, grid: GridSpec) -> dict:
        """
        Sample + derive all layers for a grid once, cached.

        Returns:
            dict with 'accessibility', 'corridor', 'visibility', 'worldcover', 'slope'.

        Why:
            layers(), visibility(), and context() all need the same sampled rasters;
            computing them once and caching avoids re-reading the rasters per call.
        """
        key = self._key(grid)
        if key not in self._cache:
            dem = sample_raster_to_grid(self._dem_path, grid)
            worldcover = sample_raster_to_grid(self._worldcover_path, grid)
            slope = _slope_deg_from_dem(dem, grid.cell_size_m)

            access_slope = _slope_to_accessibility(slope, self._cfg.accessibility_epsilon)
            access_land = _reclass(worldcover, _WORLDCOVER_ACCESS, _DEFAULT_ACCESS)
            # Multiplicative (non-compensatory): water OR cliff -> 0.
            accessibility = access_slope * access_land
            visibility = _reclass(worldcover, _WORLDCOVER_VISIBILITY, _DEFAULT_VISIBILITY)
            corridor = np.ones((grid.n_rows, grid.n_cols))  # OSM corridor = deferred (5b)

            self._cache[key] = {
                "accessibility": accessibility,
                "corridor": corridor,
                "visibility": visibility,
                "worldcover": worldcover,
                "slope": slope,
            }
        return self._cache[key]

    def layers(self, grid: GridSpec) -> TerrainLayers:
        """Accessibility (slope x land cover) and corridor (flat until OSM, 5b)."""
        s = self._sampled(grid)
        return TerrainLayers(accessibility=s["accessibility"], corridor=s["corridor"])

    def visibility(self, grid: GridSpec) -> np.ndarray:
        """Per-cell base visibility from land cover (tree canopy lowers it)."""
        return self._sampled(grid)["visibility"]

    def context(self, grid: GridSpec, cell: Tuple[int, int]) -> Dict[str, Any]:
        """
        Describe a cell's real terrain for the LocatedEvent.

        Args:
            grid: The shared GridSpec.
            cell: (row, col).

        Returns:
            A dict with the real WorldCover land-cover label, slope, accessibility, and
            the cell-center lat/lon — for the spoken message and ground-team routing.

        Why:
            With real data the message can say "tree cover, 18° slope near …" instead of
            a synthetic guess.
        """
        s = self._sampled(grid)
        r, c = cell
        wc = s["worldcover"][r, c]
        label = _WORLDCOVER_LABEL.get(int(wc), "unknown") if not np.isnan(wc) else "unknown"
        lat, lon = grid.cell_to_latlon(r, c)
        return {
            "land_cover": label,
            "slope_deg": round(float(s["slope"][r, c]), 1),
            "accessibility": round(float(s["accessibility"][r, c]), 3),
            "latlon": (round(lat, 5), round(lon, 5)),
        }
