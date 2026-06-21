# =============================================================================
# tests/test_guide.py
# -----------------------------------------------------------------------------
# Responsible for: Verifying the guidance simulation (src/search/guide.py) — the drone
#                  leads the subject home while staying within sight, the subject arrives,
#                  and pace responds to terrain.
# Role in project: Guards the behavioral claims of the guide-home showcase. The headline
#                  property — the drone is NEVER out of sight of the subject — is asserted on
#                  every tick, because that's the assumption the whole "follow me" idea rests on.
# Assumptions: A FAKE terrain with a known accessibility array, so pace/route behavior is
#              deterministic and independent of the real rasters.
# =============================================================================

from __future__ import annotations

import numpy as np

from src.common.config import BrainConfig
from src.common.grid import GridSpec
from src.search.guide import simulate_guidance
from src.search.return_path import plan_return_path
from src.search.terrain import TerrainLayers


class _FakeTerrain:
    """A TerrainProvider over a hand-built accessibility array. Why: deterministic pace/route."""

    def __init__(self, accessibility: np.ndarray) -> None:
        self._a = accessibility

    def layers(self, grid: GridSpec) -> TerrainLayers:
        return TerrainLayers(accessibility=self._a, corridor=np.ones_like(self._a))

    def visibility(self, grid: GridSpec) -> np.ndarray:
        return np.ones_like(self._a)

    def context(self, grid: GridSpec, cell):
        return {}


def _grid(n: int = 30) -> GridSpec:
    return GridSpec.from_config(BrainConfig(n_rows=n, n_cols=n, cell_size_m=50.0))


def test_drone_stays_within_sight_every_tick():
    """
    Scenario: guide the subject along a route, checking the lead each tick.
    Why it matters: "the drone stays within sight as a thing to follow" is the core promise.
    Assert drone_s >= subject_s (it leads) and the gap never exceeds sight_distance on ANY tick.
    """
    grid = _grid()
    terrain = _FakeTerrain(np.ones((grid.n_rows, grid.n_cols)))
    path = plan_return_path(grid, terrain, (5, 5), (5, 25))
    sight = 120.0
    result = simulate_guidance(grid, terrain, path, sight_distance_m=sight, dt_s=4.0)

    for st in result.states:
        assert st.drone_s >= st.subject_s - 1e-9, "drone fell behind the subject"
        assert st.drone_s - st.subject_s <= sight + 1e-6, f"drone out of sight: gap {st.drone_s - st.subject_s:.1f}m"


def test_subject_reaches_home():
    """
    Scenario: run the sim to completion on a clear route.
    Why it matters: the point is to get the subject HOME; assert it arrives, the final subject
    position is the home cell, and the drone has also converged to home (lead clamps at the end).
    """
    grid = _grid()
    terrain = _FakeTerrain(np.ones((grid.n_rows, grid.n_cols)))
    path = plan_return_path(grid, terrain, (3, 3), (20, 22))
    result = simulate_guidance(grid, terrain, path)

    assert result.arrived
    assert result.states[-1].subject_cell == result.home
    assert result.states[-1].drone_cell == result.home  # drone waits at home as subject arrives


def test_positions_lie_on_the_route():
    """
    Scenario: check every interpolated position sits on the route polyline.
    Why it matters: the subject follows the planned (walkable) route — a position off the route
    could be on terrain we deliberately avoided. Each position should be within ~1 cell of some
    route vertex (it's a linear interpolation between adjacent route cells).
    """
    grid = _grid()
    terrain = _FakeTerrain(np.ones((grid.n_rows, grid.n_cols)))
    path = plan_return_path(grid, terrain, (4, 4), (4, 26))
    result = simulate_guidance(grid, terrain, path)

    route = np.array(path, dtype=float)
    for st in result.states:
        for pos in (st.subject_pos, st.drone_pos):
            dists = np.hypot(route[:, 0] - pos[0], route[:, 1] - pos[1])
            assert dists.min() <= 1.0 + 1e-6, f"position {pos} strayed off the route"


def test_pace_is_slower_over_hard_ground():
    """
    Scenario: two identical straight routes, one over easy ground, one over hard-but-passable
    ground (low accessibility).
    Why it matters: terrain-aware pace is what makes the guidance honest — crossing hard ground
    should take MORE ticks than easy ground for the same distance.
    """
    grid = _grid()
    easy = _FakeTerrain(np.ones((grid.n_rows, grid.n_cols)))
    hard_access = np.ones((grid.n_rows, grid.n_cols))
    hard_access[10, :] = 0.3  # the whole route row is hard ground
    hard = _FakeTerrain(hard_access)

    path = [(10, c) for c in range(3, 27)]  # a straight horizontal route along row 10
    easy_ticks = len(simulate_guidance(grid, easy, path).states)
    hard_ticks = len(simulate_guidance(grid, hard, path).states)
    assert hard_ticks > easy_ticks, "hard ground did not slow the subject down"


def test_already_home_is_handled():
    """
    Scenario: the subject is already at home (a singleton route).
    Why it matters: the degenerate case must not divide-by-zero or loop; it should arrive
    immediately with the subject at home.
    """
    grid = _grid()
    terrain = _FakeTerrain(np.ones((grid.n_rows, grid.n_cols)))
    result = simulate_guidance(grid, terrain, [(7, 7)])
    assert result.arrived
    assert result.total_length_m == 0.0
    assert result.states[0].subject_cell == (7, 7)
