# =============================================================================
# guide.py
# -----------------------------------------------------------------------------
# Responsible for: Simulating the "drone leads, subject follows" phase after a locate —
#                  the subject walks the return route at a terrain-scaled pace while the
#                  drone rides a bounded lead ahead, staying within sight, until home.
# Role in project: The guidance half of the drone-as-guide feature. Consumes a return path
#                  (from return_path.plan_return_path) and produces a time series of
#                  (drone, subject) positions the animation and the live dashboard replay.
# Role/pattern: A lightweight 1-D-along-path kinematic sim. The route is parameterized by
#               cumulative distance s; the subject advances by pace*dt, and the drone is
#               pinned at s_drone = min(s_subject + sight, L) — a leader/follower with a
#               hard "within sight" coupling, by construction.
# Assumptions (the two feature premises): (1) the subject is mobile and tracks the drone
#              beacon (perfect tracking — a feasibility simplification); (2) the drone knows
#              home and its route, so it always leads toward home. Terrain modulates the
#              subject's walking SPEED via the same accessibility layer the route used.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.common.config import BrainConfig
from src.common.contracts import Cell
from src.common.grid import GridSpec
from src.search.return_path import path_length_m
from src.search.terrain import TerrainProvider

# Default tunables (a seam, not buried magic — overridable per call). A walking pace and a
# "within sight" lead that read as plausible for a low-altitude guide drone over foot terrain.
_SUBJECT_SPEED_MPS = 1.1      # ~4 km/h on easy ground (scaled DOWN by accessibility)
_SIGHT_DISTANCE_M = 120.0     # how far ahead the drone leads while staying visible
_DT_S = 4.0                   # simulation tick
# Pace floor: even on hard ground the subject doesn't crawl to a near-stop (the route already
# avoids the worst terrain). Clamp the accessibility used for PACE into [floor, 1].
_PACE_ACCESS_FLOOR = 0.3
_MAX_TICKS = 100_000          # safety bound so a degenerate route can't loop forever


@dataclass(frozen=True)
class GuideState:
    """
    One tick of the guidance phase — presentation-free positions along the route.

    Args (fields):
        t: Elapsed sim time in seconds.
        subject_s / drone_s: Arc length (meters from the route start) of the subject and the
            drone. drone_s >= subject_s and drone_s - subject_s <= sight_distance always.
        subject_pos / drone_pos: Interpolated (row, col) FLOAT positions on the route polyline
            (float so the animation moves smoothly between cells).

    Why:
        Storing arc length AND the interpolated point lets consumers both reason about progress
        (s / L) and draw smooth motion, while the invariant lives in the s values where it's
        cheap to assert.
    """

    t: float
    subject_s: float
    drone_s: float
    subject_pos: Tuple[float, float]
    drone_pos: Tuple[float, float]

    @property
    def subject_cell(self) -> Cell:
        """Nearest cell to the subject. Why: the projection/animation place markers on cells."""
        return (int(round(self.subject_pos[0])), int(round(self.subject_pos[1])))

    @property
    def drone_cell(self) -> Cell:
        """Nearest cell to the drone. Why: same — cell-based markers + the live dronePos field."""
        return (int(round(self.drone_pos[0])), int(round(self.drone_pos[1])))


@dataclass(frozen=True)
class GuidanceResult:
    """
    The full recorded guidance phase.

    Args (fields):
        path: The return route (cells), subject -> home.
        home: The home/operator cell (path[-1]).
        total_length_m: Route length in meters (L).
        states: Ordered GuideStates, states[0] = t0 (subject at the start).
        arrived: Whether the subject reached home.
        sight_distance_m: The lead/visibility bound used (carried for assertions + captions).

    Why:
        A single value the animation and the live server both replay, so "what we simulated"
        and "what we show" can never drift.
    """

    path: List[Cell]
    home: Cell
    total_length_m: float
    states: List[GuideState]
    arrived: bool
    sight_distance_m: float


def _arc_lengths(path: List[Cell], cell_size_m: float) -> List[float]:
    """
    Cumulative ground distance (meters) to each vertex of the route polyline.

    Args:
        path: The route cells.
        cell_size_m: Grid cell edge length.

    Returns:
        A list `cum` with cum[0] = 0 and cum[k] = distance along the route to path[k].

    Why:
        Parameterizing by cumulative distance is what lets the subject and drone advance in
        real meters (and the drone hold a meters-based lead), independent of how many cells a
        segment spans.
    """
    cum = [0.0]
    for a, b in zip(path, path[1:]):
        cum.append(cum[-1] + math.hypot(a[0] - b[0], a[1] - b[1]) * cell_size_m)
    return cum


def _point_at(path: List[Cell], cum: List[float], s: float) -> Tuple[float, float]:
    """
    Interpolate the (row, col) point at arc length s along the route polyline.

    Args:
        path: The route cells.
        cum: Cumulative arc lengths from _arc_lengths.
        s: Arc length in meters, clamped into [0, total].

    Returns:
        The interpolated (row, col) as floats.

    Why:
        Smooth motion: the drone/subject sit BETWEEN cell centers as they move, so the
        animation glides instead of hopping cell-to-cell.
    """
    total = cum[-1]
    s = min(max(s, 0.0), total)
    for k in range(len(cum) - 1):
        if s <= cum[k + 1]:
            seg = cum[k + 1] - cum[k]
            frac = 0.0 if seg == 0.0 else (s - cum[k]) / seg
            ar, ac = path[k]
            br, bc = path[k + 1]
            return (ar + frac * (br - ar), ac + frac * (bc - ac))
    return (float(path[-1][0]), float(path[-1][1]))


def simulate_guidance(
    grid: GridSpec,
    terrain: TerrainProvider,
    path: List[Cell],
    *,
    subject_speed_mps: float = _SUBJECT_SPEED_MPS,
    sight_distance_m: float = _SIGHT_DISTANCE_M,
    dt_s: float = _DT_S,
    cfg: Optional[BrainConfig] = None,
) -> GuidanceResult:
    """
    Simulate the drone leading the subject home along the return route.

    Args:
        grid: The shared GridSpec.
        terrain: TerrainProvider, for the accessibility that scales walking pace.
        path: The return route (subject -> home) from plan_return_path.
        subject_speed_mps: Base walking speed on easy ground.
        sight_distance_m: The drone's lead / visibility bound — it never gets farther ahead.
        dt_s: Simulation tick length in seconds.
        cfg: Unused for now (a seam if pace/sight move into config); accepted for symmetry.

    Returns:
        A GuidanceResult with one GuideState per tick (t0 .. arrival).

    Why:
        This encodes the feature's two premises directly: the subject MOVES (advances pace*dt
        each tick, slower on hard ground), and the drone is a beacon that LEADS within sight
        (`drone_s = min(subject_s + sight, L)`). The within-sight coupling is structural, so
        the drone is provably never out of sight — the property the showcase claims.
    """
    home = path[-1]
    cum = _arc_lengths(path, grid.cell_size_m)
    total = cum[-1]
    accessibility = terrain.layers(grid).accessibility

    def point(s: float) -> Tuple[float, float]:
        return _point_at(path, cum, s)

    def state_at(t: float, s_subj: float, s_drone: float) -> GuideState:
        return GuideState(t=t, subject_s=s_subj, drone_s=s_drone,
                          subject_pos=point(s_subj), drone_pos=point(s_drone))

    # t0: subject at the start, drone already a sight-distance ahead (clamped to the route end).
    subject_s = 0.0
    drone_s = min(sight_distance_m, total)
    states: List[GuideState] = [state_at(0.0, subject_s, drone_s)]

    t = 0.0
    ticks = 0
    while subject_s < total and ticks < _MAX_TICKS:
        # Pace scales with how walkable the subject's CURRENT cell is (clamped so it never
        # crawls): easy ground -> full speed, hard ground -> slower, exactly the terrain
        # reasoning the route itself used.
        sr, sc = point(subject_s)
        cell = (int(round(sr)), int(round(sc)))
        access = float(accessibility[cell]) if grid.in_bounds(*cell) else 1.0
        pace = subject_speed_mps * min(max(access, _PACE_ACCESS_FLOOR), 1.0)

        subject_s = min(subject_s + pace * dt_s, total)
        drone_s = min(subject_s + sight_distance_m, total)  # leads, within sight, clamps at home
        t += dt_s
        ticks += 1
        states.append(state_at(t, subject_s, drone_s))

    arrived = subject_s >= total
    return GuidanceResult(
        path=path,
        home=home,
        total_length_m=path_length_m(path, grid),
        states=states,
        arrived=arrived,
        sight_distance_m=sight_distance_m,
    )
