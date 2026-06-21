# =============================================================================
# test_terrain_raster.py
# -----------------------------------------------------------------------------
# Responsible for: The real-raster TerrainProvider (Geo Unit 5). The sampling
#                  logic is tested against a TINY in-memory raster (no big files);
#                  the end-to-end RasterTerrain is tested on the real rasters when
#                  present (skipped otherwise, so CI without the data still passes).
# =============================================================================

import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.common.config import BrainConfig
from src.common.grid import GridSpec
from src.search.terrain_raster import (
    RasterTerrain,
    cell_center_lonlat,
    sample_raster_to_grid,
    _DEFAULT_DEM,
    _DEFAULT_WORLDCOVER,
)


def _write_raster(path, west, north, res, nx, ny, data, nodata=0):
    """Write a tiny single-band lat/lon GeoTIFF for sampler tests."""
    transform = from_origin(west, north, res, res)
    with rasterio.open(
        path, "w", driver="GTiff", height=ny, width=nx, count=1,
        dtype="uint8", crs="EPSG:4326", transform=transform, nodata=nodata,
    ) as dst:
        dst.write(data.astype("uint8"), 1)


# --- sampler logic, validated against rasterio's own point sampling ---


def test_sampler_matches_rasterio_point_sample(tmp_path):
    """
    Scenario: a tiny raster with a unique value per pixel; sample it to a small grid two
    ways — our windowed sampler vs rasterio's per-point ds.sample.
    Why it matters: pins that our vectorized window read indexes the SAME pixel rasterio
    would. A unique-per-pixel raster means any mis-indexing shows up as a value mismatch.
    """
    path = tmp_path / "tiny.tif"
    nx = ny = 50
    data = (np.arange(nx * ny).reshape(ny, nx) % 250 + 1)  # 1..250, avoids nodata 0
    _write_raster(path, west=-122.0, north=37.05, res=0.001, nx=nx, ny=ny, data=data)

    grid = GridSpec.from_config(BrainConfig(n_rows=10, n_cols=10, cell_size_m=100.0,
                                            origin_latlon=(37.005, -121.995)))
    got = sample_raster_to_grid(str(path), grid)

    lon, lat = cell_center_lonlat(grid)
    with rasterio.open(path) as ds:
        expected = np.array([next(ds.sample([(lo, la)]))[0]
                             for lo, la in zip(lon.ravel(), lat.ravel())]).reshape(lat.shape)
    assert np.array_equal(got, expected.astype(float))


def test_sampler_marks_out_of_bounds_as_nan(tmp_path):
    """
    Scenario: a grid that extends north of the raster's coverage.
    Why it matters: cells off the raster must be nan (so the caller fills them), not a
    wrapped or clamped neighbor pixel.
    """
    path = tmp_path / "small.tif"
    _write_raster(path, west=-122.0, north=37.05, res=0.001, nx=50, ny=50,
                  data=np.full((50, 50), 7))
    # 40x40 @ 200 m spans ~0.072 deg lat -> past the raster's 0.05 deg height
    grid = GridSpec.from_config(BrainConfig(n_rows=40, n_cols=40, cell_size_m=200.0,
                                            origin_latlon=(37.0, -122.0)))
    got = sample_raster_to_grid(str(path), grid)
    assert np.isnan(got).any()                 # northern cells fell off the raster
    assert np.nanmax(got) == pytest.approx(7)  # in-bounds cells read the real value


def test_sampler_marks_nodata_as_nan(tmp_path):
    """
    Scenario: a raster whose value is the nodata sentinel in one corner.
    Why it matters: nodata must become nan, not be treated as a real measurement.
    """
    data = np.full((50, 50), 5)
    data[:20, :20] = 0   # nodata sentinel in the NW corner (lat 37.03-37.05, lon -122.0..-121.98)
    path = tmp_path / "nd.tif"
    _write_raster(path, west=-122.0, north=37.05, res=0.001, nx=50, ny=50, data=data, nodata=0)
    # a small grid sitting squarely in the NW nodata block
    grid = GridSpec.from_config(BrainConfig(n_rows=5, n_cols=5, cell_size_m=100.0,
                                            origin_latlon=(37.035, -121.998)))
    got = sample_raster_to_grid(str(path), grid)
    assert np.isnan(got).any()


# --- RasterTerrain on the REAL rasters (skipped when the data isn't present) ---

_HAVE_RASTERS = _DEFAULT_DEM.exists() and _DEFAULT_WORLDCOVER.exists()
_real = pytest.mark.skipif(not _HAVE_RASTERS, reason="real terrain rasters not present")


@pytest.fixture
def real_cfg() -> BrainConfig:
    return BrainConfig()  # the demo's 160x160 Marin grid


@_real
def test_raster_terrain_layers_in_range(real_cfg):
    """
    Scenario: build RasterTerrain on the real DEM + WorldCover.
    Why it matters: accessibility and visibility must stay in [0,1] over the real AOI, or
    the prior's invariants break. Also confirms some cells are impassable (water/cliffs).
    """
    grid = GridSpec.from_config(real_cfg)
    terrain = RasterTerrain(real_cfg)
    layers = terrain.layers(grid)
    vis = terrain.visibility(grid)
    assert layers.accessibility.shape == (real_cfg.n_rows, real_cfg.n_cols)
    assert 0.0 <= layers.accessibility.min() and layers.accessibility.max() <= 1.0
    assert 0.0 <= vis.min() and vis.max() <= 1.0
    assert (layers.accessibility == 0.0).any()    # water/cliffs exist in this AOI


@_real
def test_water_cells_are_impassable(real_cfg):
    """
    Scenario: a WorldCover water cell (class 80).
    Why it matters: a person can't be in open water — water must map to accessibility 0
    (a hard zero the prior respects), regardless of slope.
    """
    grid = GridSpec.from_config(real_cfg)
    terrain = RasterTerrain(real_cfg)
    wc = terrain._sampled(grid)["worldcover"]
    access = terrain.layers(grid).accessibility
    water = np.argwhere(wc == 80)
    if len(water):  # the Marin AOI includes coast/bay water
        r, c = int(water[0][0]), int(water[0][1])
        assert access[r, c] == 0.0


@_real
def test_raster_terrain_is_deterministic_and_context_is_real(real_cfg):
    """
    Scenario: sample twice; describe a cell.
    Why it matters: the prior must be reproducible, and context() must carry a REAL
    WorldCover label (e.g. 'tree cover') for the LocatedEvent, not a synthetic guess.
    """
    grid = GridSpec.from_config(real_cfg)
    t1 = RasterTerrain(real_cfg).visibility(grid)
    t2 = RasterTerrain(real_cfg).visibility(grid)
    assert np.array_equal(t1, t2)
    ctx = RasterTerrain(real_cfg).context(grid, (80, 80))
    assert {"land_cover", "slope_deg", "accessibility", "latlon"} <= set(ctx)
    assert isinstance(ctx["land_cover"], str)


def test_missing_raster_raises_clearly(tmp_path):
    """
    Scenario: construct RasterTerrain pointing at a non-existent file.
    Why it matters: it must fail loudly so the caller can fall back to SyntheticTerrain,
    not silently produce an empty/garbage map.
    """
    with pytest.raises(FileNotFoundError):
        RasterTerrain(BrainConfig(), dem_path=tmp_path / "nope.tif")
