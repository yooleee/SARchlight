# =============================================================================
# test_planner.py
# -----------------------------------------------------------------------------
# Responsible for: The search director (C1) — the SectorGrid overlay, per-sector
#                  POA/priority aggregation, and ranking. These are pure functions on
#                  posterior/coverage arrays, so they test fast with tiny synthetic maps.
# Role in project: Pins the tasking layer's math (cell<->sector mapping, ragged edges,
#                  POA = Σ posterior, priority = POA x (1 - mean coverage), ranking) before
#                  the single/multi-drone planners build on it.
# =============================================================================

import numpy as np

from src.common.config import BrainConfig
from src.common.grid import GridSpec
from src.search.brain import SearchBrain
from src.search.terrain import SyntheticTerrain
from src.search.planner import (
    SectorGrid,
    SectorPlanner,
    boustrophedon_waypoints,
    rank_sectors,
    sector_poa,
)


def _grid(n_rows, n_cols, sector_size_cells):
    """A small GridSpec + its SectorGrid overlay for a given sector size."""
    cfg = BrainConfig(n_rows=n_rows, n_cols=n_cols, sector_size_cells=sector_size_cells)
    grid = GridSpec.from_config(cfg)
    return grid, SectorGrid.from_config(grid, cfg)


# --- SectorGrid geometry ---------------------------------------------------------

def test_sector_mapping_and_names_even_grid():
    """
    Scenario: a 20x20 grid with 10-cell sectors -> a clean 2x2 sector grid.
    Why it matters: the cell->sector mapping and the human names are the legibility layer;
    A1 must be the SW sector, columns letter west->east, rows number south->north.
    """
    _, sectors = _grid(20, 20, 10)
    assert (sectors.n_sector_rows, sectors.n_sector_cols) == (2, 2)
    assert sectors.sector_of((5, 5)) == (0, 0) and sectors.name((0, 0)) == "A1"   # SW
    assert sectors.sector_of((15, 5)) == (1, 0) and sectors.name((1, 0)) == "A2"  # north of A1
    assert sectors.sector_of((5, 15)) == (0, 1) and sectors.name((0, 1)) == "B1"  # east of A1
    assert sectors.bounds((1, 1)) == (10, 20, 10, 20)


def test_ragged_last_sector_is_clipped():
    """
    Scenario: a 25x25 grid with 10-cell sectors -> a 3x3 sector grid whose last row/col is
    only 5 cells wide.
    Why it matters: a grid size that isn't a multiple of the sector size must not index past the
    edge or invent cells — the ragged sector's bounds clip to the grid (5x5 = 25 cells).
    """
    _, sectors = _grid(25, 25, 10)
    assert (sectors.n_sector_rows, sectors.n_sector_cols) == (3, 3)
    r0, r1, c0, c1 = sectors.bounds((2, 2))
    assert (r0, r1, c0, c1) == (20, 25, 20, 25)
    assert (r1 - r0) * (c1 - c0) == 25
    assert sectors.center_cell((0, 0)) == (4, 4)  # center of the 0..10 block


# --- POA / priority aggregation --------------------------------------------------

def test_poa_sums_posterior_per_sector():
    """
    Scenario: a 20x20 grid (2x2 sectors) with all probability mass placed inside sector (1,1).
    Why it matters: POA must be exactly the probability mass a sector holds (standard SAR
    probability-of-area), so the sector with the mass scores high and the empty ones score 0.
    """
    grid, sectors = _grid(20, 20, 10)
    posterior = np.zeros((20, 20))
    posterior[12:15, 13:16] = 0.1            # 9 cells * 0.1 = 0.9, all in sector (1,1)
    coverage = np.zeros((20, 20))

    scores = sector_poa(posterior, coverage, sectors)
    assert abs(scores.poa[1, 1] - 0.9) < 1e-9
    assert scores.poa[0, 0] == 0.0 and scores.poa[0, 1] == 0.0 and scores.poa[1, 0] == 0.0
    # With no coverage, priority == POA.
    assert abs(scores.priority[1, 1] - 0.9) < 1e-9


def test_coverage_discounts_priority_so_swept_sectors_rank_lower():
    """
    Scenario: two sectors hold mass — a high-POA sector that is fully covered (already swept) and
    a lower-POA sector that is untouched.
    Why it matters: priority = POA x (1 - mean coverage) is what makes the search MOVE ON: an
    exhausted sector must rank below an unsearched one even if it once held more probability.
    """
    grid, sectors = _grid(20, 20, 10)
    posterior = np.zeros((20, 20))
    posterior[0:10, 0:10] = 0.05    # sector (0,0): POA 5.0, but we mark it fully swept
    posterior[0:10, 10:20] = 0.02   # sector (0,1): POA 2.0, untouched
    coverage = np.zeros((20, 20))
    coverage[0:10, 0:10] = 1.0       # (0,0) fully cleared

    scores = sector_poa(posterior, coverage, sectors)
    assert scores.poa[0, 0] > scores.poa[0, 1]            # (0,0) still holds more mass
    assert scores.priority[0, 0] < scores.priority[0, 1]  # ...but ranks LOWER once swept

    ranked = rank_sectors(scores, sectors)
    assert ranked[0].sector == (0, 1)   # the untouched sector is now the top search target
    assert ranked[0].name == "B1"


def test_rank_sectors_orders_by_priority_and_respects_k():
    """
    Scenario: distinct mass in three sectors; rank them.
    Why it matters: the ranked list is the planner's backbone — it must be priority-descending,
    carry names + POA, and honor a top-k cut (multi-drone walks the top-k to hand out sectors).
    """
    grid, sectors = _grid(20, 20, 10)
    posterior = np.zeros((20, 20))
    posterior[0:10, 0:10] = 0.01    # (0,0) POA 1.0
    posterior[10:20, 0:10] = 0.03   # (1,0) POA 3.0  <- highest
    posterior[0:10, 10:20] = 0.02   # (0,1) POA 2.0
    scores = sector_poa(posterior, np.zeros((20, 20)), sectors)

    top2 = rank_sectors(scores, sectors, k=2)
    assert len(top2) == 2
    assert [rs.sector for rs in top2] == [(1, 0), (0, 1)]            # 3.0 then 2.0
    assert top2[0].priority >= top2[1].priority


# --- Boustrophedon sweep + single-drone plan -------------------------------------

def test_boustrophedon_covers_sector_edges_and_serpentines():
    """
    Scenario: lay a sweep over a 10x10 sector with 4-cell swath spacing.
    Why it matters: the passes must reach the sector EDGES (rows/cols 0 and 9), not just the
    interior — a subject on a sector boundary must still be covered — and the path must
    SERPENTINE (each pass reverses) for gap-free coverage with no wasted turns.
    """
    wps = boustrophedon_waypoints((0, 10, 0, 10), swath_spacing=4)
    rows = [r for r, _ in wps]
    cols = [c for _, c in wps]
    assert all(0 <= r < 10 and 0 <= c < 10 for r, c in wps)
    assert min(rows) == 0 and max(rows) == 9 and min(cols) == 0 and max(cols) == 9  # edges covered
    # First pass runs left-to-right (col 0 -> 9), the next reverses (serpentine).
    assert wps[0] == (0, 0) and wps[1][1] > wps[0][1]
    second_pass_first_col = next(c for r, c in wps if r != wps[0][0])
    assert second_pass_first_col == 9  # the second pass enters from the far side


def test_boustrophedon_starts_near_the_entry_corner():
    """
    Scenario: the drone arrives from the NE corner (high row, high col).
    Why it matters: starting the sweep at the nearest corner avoids flying across the sector to
    begin — important when chaining sector to sector in the closed loop.
    """
    wps = boustrophedon_waypoints((0, 10, 0, 10), swath_spacing=4, entry_corner=(9, 9))
    assert wps[0] == (9, 9)   # starts at the corner nearest the drone (NE)


def test_plan_single_targets_the_top_sector_and_heads_to_first_waypoint():
    """
    Scenario: a 20x20 map with all mass in sector (1,1), nothing covered.
    Why it matters: plan_single must choose the highest-priority sector, return a sweep over it,
    and set next_target to the first waypoint so the "go here" arrow and the flown path agree.
    """
    cfg = BrainConfig(n_rows=20, n_cols=20, sector_size_cells=10, swath_spacing_cells=4)
    grid = GridSpec.from_config(cfg)
    planner = SectorPlanner(grid, cfg)
    posterior = np.zeros((20, 20))
    posterior[12:15, 13:16] = 0.1   # mass in sector (1,1)

    plan = planner.plan_single(posterior, np.zeros((20, 20)))
    assert plan is not None
    assert plan.target_sector == (1, 1) and plan.target_name == "B2"
    assert plan.next_target == plan.waypoints[0]
    assert all(10 <= r < 20 and 10 <= c < 20 for r, c in plan.waypoints)


def test_plan_single_returns_none_on_a_swept_out_map():
    """
    Scenario: no probability mass anywhere (all priority 0).
    Why it matters: the planner must say "nothing worth searching" (None) rather than invent a
    target — the closed-loop driver uses None to stop.
    """
    cfg = BrainConfig(n_rows=20, n_cols=20, sector_size_cells=10)
    grid = GridSpec.from_config(cfg)
    planner = SectorPlanner(grid, cfg)
    assert planner.plan_single(np.zeros((20, 20)), np.zeros((20, 20))) is None


# --- Brain integration: MapState publishes + serializes search_path --------------

def test_brain_publishes_search_path_and_it_serializes():
    """
    Scenario: a fresh brain on synthetic terrain publishes its MapState.
    Why it matters: the brain must emit the ratified search_path (a non-empty sweep over the top
    sector) AND to_dict must serialize it (the dashboard/operator read it over the seam). The
    legacy next_target stays present (backward-compatible — its behavior is unchanged).
    """
    cfg = BrainConfig()
    brain = SearchBrain(cfg, SyntheticTerrain(cfg))
    ms = brain.map_state()

    assert ms.search_path is not None and len(ms.search_path) > 0
    assert ms.next_target is not None                      # legacy field still emitted
    d = ms.to_dict()
    assert d["search_path"] == [list(cell) for cell in ms.search_path]


# --- Multi-drone assignment (disjoint sectors) -----------------------------------

def test_assign_multi_gives_drones_disjoint_top_sectors():
    """
    Scenario: mass in four sectors, three drones starting at one corner.
    Why it matters: C1's no-overlap guarantee — assign_multi must hand each drone a DISTINCT
    sector (no two drones sent to the same ground) and pick the highest-priority ones first.
    """
    cfg = BrainConfig(n_rows=40, n_cols=40, sector_size_cells=10, swath_spacing_cells=4)
    grid = GridSpec.from_config(cfg)
    planner = SectorPlanner(grid, cfg)
    posterior = np.zeros((40, 40))
    posterior[0:10, 0:10] = 0.04     # sector (0,0) POA 4 (highest)
    posterior[0:10, 10:20] = 0.03    # (0,1) POA 3
    posterior[10:20, 0:10] = 0.02    # (1,0) POA 2
    posterior[30:40, 30:40] = 0.01   # (3,3) POA 1 (lowest)
    coverage = np.zeros((40, 40))

    plans = planner.assign_multi(posterior, coverage, [(0, 0), (0, 0), (0, 0)])
    sectors = [p.target_sector for p in plans if p is not None]
    assert len(sectors) == 3 and len(set(sectors)) == 3              # three DISTINCT sectors
    assert set(sectors) == {(0, 0), (0, 1), (1, 0)}                  # the three highest-POA


def test_assign_multi_respects_locked_sectors_and_runs_out():
    """
    Scenario: a sector is already locked to another drone (mid-sweep), and there are more drones
    than remaining sectors with mass.
    Why it matters: a locked sector must never be reassigned (no overlap), and a drone with no
    sector left gets None (it idles rather than doubling up).
    """
    cfg = BrainConfig(n_rows=40, n_cols=40, sector_size_cells=10)
    grid = GridSpec.from_config(cfg)
    planner = SectorPlanner(grid, cfg)
    posterior = np.zeros((40, 40))
    posterior[0:10, 0:10] = 0.04     # sector (0,0)
    posterior[0:10, 10:20] = 0.03    # sector (0,1)

    # (0,0) is locked to someone else; two free drones, only (0,1) left worth searching.
    plans = planner.assign_multi(posterior, np.zeros((40, 40)), [(0, 0), (0, 0)], exclude={(0, 0)})
    assigned = [p.target_sector for p in plans if p is not None]
    assert (0, 0) not in assigned                 # the locked sector is never handed out
    assert assigned == [(0, 1)]                   # exactly one drone gets the one free sector
    assert plans.count(None) == 1                 # the other idles (no double-up)
