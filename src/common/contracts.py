# =============================================================================
# contracts.py
# -----------------------------------------------------------------------------
# Responsible for: The ratified data contracts at the brain's boundaries —
#                  Observation (in), MapState + LocatedEvent (out) — plus the
#                  small value types they carry (CellCoverage, GroundDetection)
#                  and the Status / SensorType enums.
# Role in project: interfaces.md §4 and §6. These are the *shared* contract, so
#                  the field sets here must stay byte-identical with teammates'
#                  copies; only the representation (plain dataclasses) is ours.
# Assumptions: posterior/coverage are NumPy float arrays of shape (n_rows,
#              n_cols). MapState.to_dict() is the serializable seam — it turns
#              arrays into lists so the state can cross a socket/Redis boundary
#              later without a contract change.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grid import GridSpec

# A cell is addressed as (row, col) everywhere in the system.
Cell = Tuple[int, int]


class SensorType(str, Enum):
    """
    The sensor that produced a frame; modulates detection probability downstream.

    Why:
        d_i's recall term depends on the sensor (thermal beats color at dusk).
        Subclassing str keeps it JSON-friendly (the value serializes as "color").
    """

    COLOR = "color"
    THERMAL = "thermal"


class Status(str, Enum):
    """
    The search status carried in MapState.

    Why:
        Consumers (dashboard, voice) branch on this. Declared `extensible` in the
        contract, so new states can be added without breaking the enum's str values.
    """

    SEARCHING = "searching"
    LOCATED = "located"


# --- Observation and its parts (interfaces.md §4: geo -> map) ---


@dataclass(frozen=True)
class CellCoverage:
    """
    One ground cell a frame covered, with how well it was seen.

    Args (fields):
        cell: (row, col) the frame overlapped.
        coverage_fraction: Fraction of the cell inside the footprint, in [0, 1].
        visibility_weight: How observable a person would be here this look, in
            [0, 1] (canopy lowers it; a clear nadir look raises it).

    Why:
        These two numbers are exactly what the non-detection update needs: they
        feed d_i, so a partial or canopy-occluded look barely clears a cell while a
        full clear look clears it strongly.
    """

    cell: Cell
    coverage_fraction: float
    visibility_weight: float


@dataclass(frozen=True)
class GroundDetection:
    """
    A single detection projected onto the ground.

    Args (fields):
        cell: (row, col) where the detection's box center projects on the ground.
        confidence: Raw detector score in [0, 1], consumed as evidence strength
            (not a probability of presence).

    Why:
        The detection update needs only where the detection landed and how strong
        the evidence is; everything pixel-space stayed behind the GeoReferencer.
    """

    cell: Cell
    confidence: float


@dataclass(frozen=True)
class Observation:
    """
    Everything the map consumes for one frame: which cells were seen, how well,
    and where (if anywhere) a person was detected.

    Args (fields):
        frame_id: Stable handle for provenance back to the source frame.
        timestamp: Sim-clock time the frame was captured.
        footprint: The covered cells (drives the non-detection update). An empty
            footprint means the frame saw no ground cells on our grid.
        detections_ground: Zero or more ground detections. An EMPTY list is
            meaningful — it is the non-detection signal over the footprint.
        sensor_type: COLOR or THERMAL, modulating recall in d_i.

    Why:
        This is the *only* input contract the brain depends on, which is what lets
        the brain run against a stubbed GeoReferencer (mock Observations) today and
        a real one later with zero change here.
    """

    frame_id: str
    timestamp: float
    footprint: List[CellCoverage]
    detections_ground: List[GroundDetection]
    sensor_type: SensorType


# --- Outputs (interfaces.md §6: map -> consumers) ---


@dataclass
class MapState:
    """
    The read model and single source of truth, written ONLY by the brain.

    Args (fields):
        grid_spec: The frame of reference for every array below.
        update_count: Monotonic counter; lets consumers cheaply detect new state.
        timestamp: Time of the last update.
        posterior: (n_rows, n_cols) float array of p_i over the grid (the in-region
            mass; p_out is tracked separately by the brain and reported here).
        coverage: (n_rows, n_cols) float array of accumulated cleared_i in [0, 1].
        top_cells: Ranked [(cell, p_i), ...], highest probability first — the
            search targets.
        next_target: The cell the map directs the drone toward next (or None).
        status: SEARCHING or LOCATED.
        detections_log: Confirmed detection events for the dashboard timeline.
        p_out: Reserved out-of-region mass; posterior.sum() + p_out == 1.

    Why:
        Not frozen because the brain republishes a fresh snapshot each step. Plain
        dataclass + to_dict() gives the "serializable seam" the contract requires
        without pulling in a serialization framework.
    """

    grid_spec: GridSpec
    update_count: int
    timestamp: float
    posterior: np.ndarray
    coverage: np.ndarray
    top_cells: List[Tuple[Cell, float]]
    next_target: Optional[Cell]
    status: Status
    detections_log: List[Dict[str, Any]] = field(default_factory=list)
    p_out: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain JSON-serializable dict (arrays -> nested lists).

        Returns:
            A dict mirroring the fields, with posterior/coverage as lists, the
            grid_spec flattened, and the status as its string value.

        Why:
            This is the seam that lets MapState cross a process/socket boundary
            (a web dashboard, or Redis if we go multi-process) without changing the
            contract. We build the seam now and defer the transport choice.
        """
        return {
            "grid_spec": {
                "crs": self.grid_spec.crs,
                "origin": list(self.grid_spec.origin),
                "cell_size_m": self.grid_spec.cell_size_m,
                "n_rows": self.grid_spec.n_rows,
                "n_cols": self.grid_spec.n_cols,
            },
            "update_count": self.update_count,
            "timestamp": self.timestamp,
            "posterior": self.posterior.tolist(),
            "coverage": self.coverage.tolist(),
            "top_cells": [[list(cell), float(p)] for cell, p in self.top_cells],
            "next_target": list(self.next_target) if self.next_target else None,
            "status": self.status.value,
            "detections_log": self.detections_log,
            "p_out": self.p_out,
        }


@dataclass(frozen=True)
class LocatedEvent:
    """
    Emitted once on a confident, persistent find. Triggers the subject broadcast
    and the operator alert.

    Args (fields):
        cell: (row, col) of the located subject (the trigger's peak cell).
        latlon: (lat, lon) of that cell's center, for routing ground teams.
        confidence: Aggregate confidence at trigger time.
        terrain_context: Land cover / slope / nearby trail or water at the
            location — for the spoken message and ground-team routing.
        timestamp: When the trigger fired.

    Why:
        Carries everything the message composer needs so the voice layer never has
        to reach back into the brain's internals — keeps the brain authoritative
        and the consumers decoupled.
    """

    cell: Cell
    latlon: Tuple[float, float]
    confidence: float
    terrain_context: Dict[str, Any]
    timestamp: float
