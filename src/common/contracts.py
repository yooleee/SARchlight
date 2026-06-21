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


# --- Detector & telemetry inputs (interfaces.md §2-§3: detector/telemetry -> geo) ---
#
# These mirror the ratified interfaces.md field sets. They are the EXPECTED shape, not a
# hard guarantee: the detector is a separate track, so a thin adapter at the geo boundary
# (Geo Unit 2) normalizes whatever it actually emits into these types. Geo never depends on
# the detector's internals — only on this normalized form.


@dataclass(frozen=True)
class Detection:
    """
    One pixel-space detection from the detector (geography-blind).

    Args (fields):
        bbox_xywh: Pixel box as (x, y, w, h) — top-left corner + size, in image pixels.
        confidence: Raw detector score in [0, 1]. Evidence strength, NOT a probability
            of presence (the brain treats it as bucketed/uncalibrated, interfaces §5.3).
        class_name: The detected class; "person" for the weekend. (Named class_name
            because `class` is a Python keyword; maps to the contract's `class` field.)

    Why:
        The detector never sees geography, which is what keeps its backend swappable; all
        it reports is a box, a score, and a label. Geo turns the box into a ground cell.
    """

    bbox_xywh: Tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"


@dataclass(frozen=True)
class DetectorOutput:
    """
    The detector's output for one frame (interfaces §2). Pixel-space, no lat/lon, no grid.

    Args (fields):
        frame_id: Stable handle for this frame; the join key to its CameraPose.
        timestamp: When the frame was captured (sim clock is fine).
        detections: Zero or more Detection. An EMPTY list is meaningful — it is the
            non-detection signal that lets the map lower probability over the covered area.
        model_id: Which backend produced this (e.g. "yolo11s-pretrained"); provenance and
            A/B of a fine-tuned swap.
        sensor_type: COLOR or THERMAL; affects detection probability downstream (§5).
        inference_ms: Optional latency, for the dashboard / observability.

    Why:
        This is boundary A — the seam where the teammate's detector plugs in. Geo consumes
        exactly this (after the adapter normalizes it), so swapping simulator -> recorded
        trace -> live detector is wiring, not redesign.
    """

    frame_id: str
    timestamp: float
    detections: List[Detection]
    model_id: str
    sensor_type: SensorType
    inference_ms: Optional[float] = None


@dataclass(frozen=True)
class CameraPose:
    """
    Per-frame camera telemetry (interfaces §3), supplied alongside each frame.

    Args (fields):
        frame_id: Join key to DetectorOutput.
        drone_latlon: (lat, lon) camera position.
        altitude_agl_m: Height above ground level; sets the ground sample distance.
        heading_deg: Compass heading of the camera (0 = north, 90 = east).
        gimbal_pitch_deg: Downward tilt; 90 = straight down (nadir).
        fov_deg: Field of view as (horizontal, vertical) in degrees. (interfaces §3 allows
            a single float or (h, v); we use the structured (h, v) form geo needs.)
        image_size_px: Frame dimensions as (width, height) in pixels, to map pixels->ground.

    Why:
        The detector is geography-blind; this telemetry is what lets geo place a pixel box
        on the real ground. The Detector never sees this — it is joined to DetectorOutput
        by frame_id at the geo boundary.
    """

    frame_id: str
    drone_latlon: Tuple[float, float]
    altitude_agl_m: float
    heading_deg: float
    gimbal_pitch_deg: float
    fov_deg: Tuple[float, float]
    image_size_px: Tuple[int, int]


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
        search_path: The recommended single-drone sweep — an ordered list of cells the
            search director (the sector planner) lays over the top-priority sector. None
            until a planner is wired in. The ratified §6.1 `next_target` / `search_path`
            field; added additively so existing consumers are unaffected.
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
    search_path: Optional[List[Cell]] = None
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
            "search_path": [list(cell) for cell in self.search_path] if self.search_path else None,
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
