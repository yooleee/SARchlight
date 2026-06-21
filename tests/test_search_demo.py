# =============================================================================
# test_search_demo.py
# -----------------------------------------------------------------------------
# Responsible for: The C1 closed-loop search demo — that the planner-driven flight
#                  (no scripted path) actually LOCATES the subject on real terrain,
#                  and that the path EMERGES from the planner (starts at the LKP, the
#                  drone is sent to sectors). Skipped when the rasters aren't present.
# Role in project: The acceptance regression for the single-drone closed loop (the
#                  multi-drone acceptance is added in Unit 4).
# =============================================================================

import math

import pytest

from src.search.terrain_raster import _DEFAULT_DEM, _DEFAULT_WORLDCOVER
from src.demo.search_demo import PLANNER_CONFIG, run_multi_drone, run_single_drone

_HAVE_RASTERS = _DEFAULT_DEM.exists() and _DEFAULT_WORLDCOVER.exists()
_real = pytest.mark.skipif(not _HAVE_RASTERS, reason="real terrain rasters not present")


@_real
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_closed_loop_locates_with_an_emergent_path(seed):
    """
    Scenario: the single-drone closed loop on real terrain — the planner ranks sectors, the
    drone sweeps the top one with thermal, the map updates, until it locates.
    Why it matters: this is C1's core claim — that the flight path EMERGES from the planner
    (not a script) and still finds the subject. Assert it locates within a cell of the truth
    AND that the path is real (starts at the LKP, the drone actually flew).
    """
    result = run_single_drone(cfg=PLANNER_CONFIG, seed=seed)

    assert result.located_event is not None, f"closed loop failed to locate (seed {seed})"
    err = math.hypot(result.located_event.cell[0] - result.subject[0],
                     result.located_event.cell[1] - result.subject[1])
    assert err <= 1.5, f"located {result.located_event.cell} too far from {result.subject} (seed {seed})"

    path = result.frames[-1].drones[0].path
    assert len(path) > 0                                   # the drone actually flew
    lkp = result.grid.latlon_to_cell(*PLANNER_CONFIG.lkp_latlon)
    # The path begins near the LKP (the planner sends the drone out FROM there, not from the find).
    assert math.hypot(path[0][0] - lkp[0], path[0][1] - lkp[1]) <= result.planner.sectors.sector_size_cells * 2


@_real
def test_closed_loop_preserves_the_mass_invariant():
    """
    Scenario: inspect the final frame of a closed-loop run.
    Why it matters: the planner drives the brain through many real updates; the core invariant
    Σ posterior + p_out == 1 must survive all of them (the planner never writes belief — it only
    reads it to choose where to fly).
    """
    result = run_single_drone(cfg=PLANNER_CONFIG, seed=0)
    last = result.frames[-1]
    assert last.posterior.sum() + last.p_out == pytest.approx(1.0)


# --- Multi-drone coordination ----------------------------------------------------

@_real
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multi_drone_locates_with_no_overlap(seed):
    """
    Scenario: three drones, coordinated by the planner, sweep disjoint sectors on real terrain.
    Why it matters: C1's headline — coordinated, NON-OVERLAPPING coverage that still locates. At
    EVERY tick no two drones may be assigned the same sector (the no-overlap guarantee), and the
    loop must still find the subject. All N streams feed the one brain (single writer).
    """
    result = run_multi_drone(cfg=PLANNER_CONFIG, seed=seed, n_drones=3)

    assert result.located_event is not None, f"multi-drone loop failed to locate (seed {seed})"
    err = math.hypot(result.located_event.cell[0] - result.subject[0],
                     result.located_event.cell[1] - result.subject[1])
    assert err <= 1.5

    # No two drones ever share a sector in the same frame.
    for fr in result.frames:
        sectors = [dv.target_sector for dv in fr.drones if dv.target_sector is not None]
        assert len(sectors) == len(set(sectors)), f"two drones shared a sector (seed {seed})"


@_real
def test_multi_drone_uses_all_drones():
    """
    Scenario: a 3-drone run.
    Why it matters: coordination only means something if the drones actually fan out — each of the
    three should have flown (a non-empty path), not sat idle while one drone did all the work.
    """
    result = run_multi_drone(cfg=PLANNER_CONFIG, seed=0, n_drones=3)
    final_views = result.frames[-1].drones
    assert len(final_views) == 3
    assert all(len(dv.path) > 0 for dv in final_views)   # every drone contributed
