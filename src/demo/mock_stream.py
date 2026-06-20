# =============================================================================
# mock_stream.py
# -----------------------------------------------------------------------------
# Responsible for: A scripted, deterministic Observation stream standing in for
#                  the real GeoReferencer — a SW sweep of clean non-detections
#                  down the corridor, then color detections on the subject cell,
#                  then a thermal corroboration (docs/demo_scenario.md §5 & §7).
# Role in project: Lets the brain run the full loop today. When the real
#                  GeoReferencer exists, it produces the same Observation contract
#                  and this module is simply unplugged.
# Assumptions: Subject is ~1.8 km SW of the LKP, near the drainage (demo §4). All
#              cells are placed relative to the LKP cell so the stream tracks the
#              grid, not hard-coded indices.
# =============================================================================

from __future__ import annotations

from typing import List, Tuple

from src.common.config import BrainConfig
from src.common.contracts import (
    CellCoverage,
    GroundDetection,
    Observation,
    SensorType,
)
from src.common.grid import GridSpec

# Cells SW of the LKP to the subject: ~1.8 km / 50 m ≈ 36 cells along the diagonal.
_SUBJECT_OFFSET_ROWS = -26  # south of the LKP
_SUBJECT_OFFSET_COLS = -23  # west of the LKP
_FRAME_DT_S = 9.0           # sim seconds between frames


def subject_cell(grid: GridSpec, cfg: BrainConfig) -> Tuple[int, int]:
    """
    The subject's true cell, ~1.8 km SW of the LKP near the drainage.

    Args:
        grid: The shared GridSpec.
        cfg: Brain config (for the LKP).

    Returns:
        (row, col) of the planted subject.

    Why:
        Chosen (per demo §4) so the prior gives this area MODERATE — not top —
        probability: the update has to do real work to lock on, which is what makes
        the demo's map evolution meaningful rather than trivial.
    """
    lkp_r, lkp_c = grid.latlon_to_cell(*cfg.lkp_latlon)
    return (lkp_r + _SUBJECT_OFFSET_ROWS, lkp_c + _SUBJECT_OFFSET_COLS)


def _footprint(grid: GridSpec, center: Tuple[int, int], visibility: float, half: int = 1) -> List[CellCoverage]:
    """
    A square footprint of fully-covered cells around a center.

    Args:
        grid: GridSpec, for bounds.
        center: (row, col) center of the look.
        visibility: visibility_weight for every cell in this look (canopy lowers it).
        half: half-width; half=1 gives a 3x3 footprint (~6 cells at this altitude).

    Returns:
        A list of CellCoverage, clipped to the grid.

    Why:
        A ~3x3 ground footprint at 100 m / 70° FOV over a 50 m grid (demo §5.2) is the
        right amount of overlap for coverage to accumulate smoothly across the sweep.
    """
    cells: List[CellCoverage] = []
    r0, c0 = center
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            cell = (r0 + dr, c0 + dc)
            if grid.in_bounds(*cell):
                cells.append(CellCoverage(cell=cell, coverage_fraction=1.0, visibility_weight=visibility))
    return cells


def build_demo_stream(grid: GridSpec, cfg: BrainConfig) -> Tuple[Tuple[int, int], List[Observation]]:
    """
    Build the scripted Observation sequence for the demo.

    Args:
        grid: The shared GridSpec.
        cfg: Brain config (LKP, sensor types).

    Returns:
        (subject, observations): the planted subject cell and the ordered frames —
        a clean SW sweep, then 3 color detections, then 2 thermal corroborations.

    Why:
        Mirrors demo §7's detection/non-detection sequence: the sweep dims the
        high-prior corridor (coverage rises, mass redistributes), color hits brighten
        the subject but stay `searching` (persistence not met), and the thermal pass
        meets persistence AND crosses P_located → `located`.
    """
    lkp_r, lkp_c = grid.latlon_to_cell(*cfg.lkp_latlon)
    subject = subject_cell(grid, cfg)

    observations: List[Observation] = []
    t = 0.0
    frame_no = 1

    # --- Sweep: a boustrophedon ("lawnmower") over the high-prior corridor band ---
    # The band spans from just SW of the LKP (over the prior peak) down toward — but
    # stopping SHORT of — the subject, so the subject stays an unsearched surprise.
    # 5x5 footprints stepping by 4-5 cells give full, overlapping coverage of the band.
    row_top = lkp_r + 2
    row_bottom = subject[0] + 6        # stay clear of the subject's neighborhood
    col_left = subject[1] - 2
    col_right = lkp_c + 4
    row_centers = list(range(row_top, row_bottom - 1, -4))   # marching south
    serpentine = True
    for row_c in row_centers:
        cols = list(range(col_left, col_right + 1, 5))
        if not serpentine:
            cols = cols[::-1]            # reverse alternate passes (the lawnmower turn)
        serpentine = not serpentine
        for col_c in cols:
            footprint = _footprint(grid, (row_c, col_c), visibility=0.7, half=2)
            observations.append(
                Observation(
                    frame_id=f"F{frame_no:04d}",
                    timestamp=t,
                    footprint=footprint,
                    detections_ground=[],   # clean look -> non-detection (clears area)
                    sensor_type=SensorType.COLOR,
                )
            )
            frame_no += 1
            t += _FRAME_DT_S

    # --- Detections on the subject: color first (partial canopy), conf ~0.6 ---
    for conf in (0.62, 0.60, 0.64):
        observations.append(
            Observation(
                frame_id=f"F{frame_no:04d}",
                timestamp=t,
                footprint=[CellCoverage(cell=subject, coverage_fraction=1.0, visibility_weight=0.6)],
                detections_ground=[GroundDetection(cell=subject, confidence=conf)],
                sensor_type=SensorType.COLOR,
            )
        )
        frame_no += 1
        t += _FRAME_DT_S

    # --- Thermal corroboration: stronger evidence (conf ~0.85+) under low light ---
    for conf in (0.85, 0.88):
        observations.append(
            Observation(
                frame_id=f"F{frame_no:04d}",
                timestamp=t,
                footprint=[CellCoverage(cell=subject, coverage_fraction=1.0, visibility_weight=0.85)],
                detections_ground=[GroundDetection(cell=subject, confidence=conf)],
                sensor_type=SensorType.THERMAL,
            )
        )
        frame_no += 1
        t += _FRAME_DT_S

    return subject, observations
