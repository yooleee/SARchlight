# =============================================================================
# test_showcase.py
# -----------------------------------------------------------------------------
# Responsible for: The real-terrain showcase acceptance — that the closed loop, run
#                  on the REAL DEM + WorldCover rasters with the NW subject placement
#                  and the thermal floor, actually LOCATES the planted subject. This
#                  is the regression that closes docs/brain_followups.md B5.
# Role in project: Mirrors test_geo_integration's synthetic locate test, but on
#                  RasterTerrain + SHOWCASE_CONFIG. Skipped when the big rasters are
#                  not present (same pattern as test_terrain_raster.py), so CI without
#                  the data still passes.
# =============================================================================

import math

import numpy as np
import pytest

from src.search.terrain_raster import _DEFAULT_DEM, _DEFAULT_WORLDCOVER
from src.demo.showcase import SHOWCASE_CONFIG, _hillshade_from_dem, drive_loop


# --- Hillshade (pure function, no rasters needed) ---------------------------------

def test_hillshade_lights_slopes_facing_the_sun():
    """
    Scenario: two constant-slope DEMs under the default NW sun (azimuth 315) — one ramp
    rising to the EAST (a west-facing slope, which faces the NW sun) and one rising to the
    WEST (an east-facing slope, turned away).
    Why it matters: a hillshade is only meaningful if slopes facing the light are brighter
    than slopes turned away — that's the whole shaded-relief effect. The west-facing ramp
    must come out brighter than the east-facing one, and all shading stays in [0, 1].
    """
    n = 20
    cols = np.arange(n)[None, :] * np.ones((n, 1))  # column index broadcast to (n, n)
    east_facing = _hillshade_from_dem(-2.0 * cols, cell_size_m=50.0)  # elevation rises westward
    west_facing = _hillshade_from_dem(2.0 * cols, cell_size_m=50.0)   # elevation rises eastward

    assert 0.0 <= east_facing.min() and west_facing.max() <= 1.0
    assert west_facing.mean() > east_facing.mean()  # the sun-facing slope is brighter


# --- RasterTerrain acceptance (skipped when the data isn't present) ----------------



# The acceptance run needs the real rasters; skip cleanly when they aren't present.
_HAVE_RASTERS = _DEFAULT_DEM.exists() and _DEFAULT_WORLDCOVER.exists()
_real = pytest.mark.skipif(not _HAVE_RASTERS, reason="real terrain rasters not present")


@_real
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_real_terrain_loop_locates_the_subject(seed):
    """
    Scenario: the full real-terrain chain — scripted NW flight, simulated detector with
    the thermal canopy floor, real GeoReferencer, real brain — run end to end on the DEM +
    WorldCover rasters, for a few seeds.
    Why it matters: closes B5. On real Marin terrain (~72% canopy, subject at ~average
    prior) the loop previously did NOT locate; with the NW placement + thermal floor + the
    recalibrated concentration gate it must now locate the planted subject at (or within a
    cell of) its true cell, robustly across seeds — and the located cell must EMERGE from
    the projection chain, not be hand-placed.
    """
    result = drive_loop(cfg=SHOWCASE_CONFIG, seed=seed)

    assert result.located_event is not None, f"real-terrain loop failed to locate (seed {seed})"
    err_cells = math.hypot(result.located_event.cell[0] - result.subject[0],
                           result.located_event.cell[1] - result.subject[1])
    assert err_cells <= 1.5, (
        f"located {result.located_event.cell} too far from true subject {result.subject} "
        f"(seed {seed})"
    )


@_real
def test_real_terrain_mass_invariant_holds():
    """
    Scenario: inspect the final recorded frame of a real-terrain run.
    Why it matters: the loop's core invariant — Σ posterior + p_out == 1 — must survive the
    whole real run (clearing, capped detection boosts, renormalization), not just synthetic.
    """
    result = drive_loop(cfg=SHOWCASE_CONFIG, seed=0)
    last = result.frames[-1]
    assert last.posterior.sum() + last.p_out == pytest.approx(1.0)


@_real
def test_located_context_is_real_landcover():
    """
    Scenario: the LocatedEvent from the real run.
    Why it matters: the spoken/broadcast message draws on terrain_context; on real terrain it
    must carry an actual WorldCover label for the subject's cell (tree cover, by construction),
    not a synthetic guess.
    """
    result = drive_loop(cfg=SHOWCASE_CONFIG, seed=0)
    assert result.located_event is not None
    ctx = result.located_event.terrain_context
    assert "land_cover" in ctx and isinstance(ctx["land_cover"], str)
