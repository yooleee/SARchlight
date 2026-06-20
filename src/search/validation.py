# =============================================================================
# validation.py
# -----------------------------------------------------------------------------
# Responsible for: Defending the brain against malformed Observations. A real
#                  GeoReferencer/detector can emit NaN (projection divide-by-zero,
#                  missing telemetry) or out-of-range values; a single NaN in one
#                  cell otherwise poisons the ENTIRE posterior (mass -> NaN). This
#                  sanitizes at the boundary: clamp ranges, neutralize NaN, drop
#                  off-grid or untrustworthy items, and report what was fixed.
# Role in project: The brain is the single consumer of Observation and the
#                  integration point, so it owns this input defense. Kept as a
#                  standalone pure function so the future GeoReferencer can reuse
#                  the same guard and so it is testable in isolation.
# Policy: sanitize-and-log (chosen) — the demo degrades gracefully on a bad frame
#         rather than crashing; the report lets us SEE that it happened.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass

from src.common.contracts import (
    CellCoverage,
    GroundDetection,
    Observation,
)
from src.common.grid import GridSpec


@dataclass(frozen=True)
class SanitationReport:
    """
    A tally of what sanitize_observation had to fix in one frame.

    Args (fields):
        dropped_cells: Footprint cells dropped (off-grid or non-finite coordinates).
        dropped_detections: Ground detections dropped (off-grid cell or NaN confidence).
        clamped: Values pulled back into [0, 1] (coverage/visibility/confidence).
        nan_fixed: Non-finite values neutralized (NaN/inf coverage or visibility -> 0).

    Why:
        Returning a report (rather than silently cleaning) is what makes the
        sanitize-and-log policy honest: the brain can log/counter exactly what was
        wrong so a real upstream bug is visible, not hidden.
    """

    dropped_cells: int = 0
    dropped_detections: int = 0
    clamped: int = 0
    nan_fixed: int = 0

    @property
    def clean(self) -> bool:
        """True if nothing needed fixing."""
        return (self.dropped_cells == 0 and self.dropped_detections == 0
                and self.clamped == 0 and self.nan_fixed == 0)

    def summary(self) -> str:
        """One-line human-readable summary for logging."""
        return (f"dropped_cells={self.dropped_cells} "
                f"dropped_detections={self.dropped_detections} "
                f"clamped={self.clamped} nan_fixed={self.nan_fixed}")


def _clean_unit_interval(value: float, nan_to: float, counters: dict) -> float:
    """
    Coerce a value into [0, 1]: non-finite -> nan_to, otherwise clamp.

    Args:
        value: The raw value (coverage_fraction, visibility_weight, or confidence).
        nan_to: What to substitute for a NaN/inf value (the safe default).
        counters: A mutable dict accumulating "clamped" and "nan_fixed" counts.

    Returns:
        A finite float in [0, 1].

    Why:
        Centralizes the clamp/NaN logic so coverage, visibility, and confidence all
        get identical, auditable treatment (DRY). NaN is mapped to a caller-chosen
        SAFE default rather than blindly clamped, because max(NaN, 0.0) is
        order-dependent in Python and would otherwise leak NaN downstream.
    """
    if not math.isfinite(value):
        counters["nan_fixed"] += 1
        return nan_to
    if value < 0.0:
        counters["clamped"] += 1
        return 0.0
    if value > 1.0:
        counters["clamped"] += 1
        return 1.0
    return value


def _valid_cell(cell, grid: GridSpec) -> bool:
    """
    Whether a (row, col) is finite, integer-valued, and inside the grid.

    Args:
        cell: A (row, col) tuple from an Observation.
        grid: GridSpec, for bounds.

    Returns:
        True if the cell can be safely used to index the posterior.

    Why:
        Off-grid indices would wrap NumPy's negative indexing silently (touching the
        wrong cell); non-finite/non-integer coordinates would raise on indexing. We
        filter both here so the brain can index without guards.
    """
    try:
        r, c = cell
    except (TypeError, ValueError):
        return False
    if not (isinstance(r, int) and isinstance(c, int)):
        # Reject floats/NaN coordinates; the contract is integer (row, col).
        return False
    return grid.in_bounds(r, c)


def sanitize_observation(obs: Observation, grid: GridSpec) -> tuple[Observation, SanitationReport]:
    """
    Return a cleaned copy of an Observation plus a report of what was fixed.

    Args:
        obs: The raw Observation (possibly containing NaN/out-of-range/off-grid data).
        grid: The shared GridSpec, for bounds checking.

    Returns:
        (clean_obs, report): a new Observation safe to feed the update, and a
        SanitationReport tallying the fixes.

    Why:
        One NaN coverage value otherwise turns the whole posterior to NaN on the next
        renormalization. Sanitizing here means the brain's math can assume finite,
        in-range, in-bounds inputs — the single most important real-data hardening,
        since the producers (detector, GeoReferencer) are exactly where messy values
        originate. NaN coverage/visibility map to 0.0 (don't clear unknown ground);
        a detection with NaN confidence is dropped (untrustworthy, not silently inert).
    """
    counters = {"clamped": 0, "nan_fixed": 0}
    dropped_cells = 0
    dropped_detections = 0

    clean_footprint = []
    for cc in obs.footprint:
        if not _valid_cell(cc.cell, grid):
            dropped_cells += 1
            continue
        # NaN coverage/visibility -> 0.0: an unknown look should NOT clear the cell.
        coverage = _clean_unit_interval(cc.coverage_fraction, nan_to=0.0, counters=counters)
        visibility = _clean_unit_interval(cc.visibility_weight, nan_to=0.0, counters=counters)
        clean_footprint.append(
            CellCoverage(cell=cc.cell, coverage_fraction=coverage, visibility_weight=visibility)
        )

    clean_detections = []
    for det in obs.detections_ground:
        if not _valid_cell(det.cell, grid):
            dropped_detections += 1
            continue
        if not math.isfinite(det.confidence):
            # A detection we can't trust the confidence of is dropped, not kept as LR=1.
            dropped_detections += 1
            continue
        confidence = _clean_unit_interval(det.confidence, nan_to=0.0, counters=counters)
        clean_detections.append(GroundDetection(cell=det.cell, confidence=confidence))

    report = SanitationReport(
        dropped_cells=dropped_cells,
        dropped_detections=dropped_detections,
        clamped=counters["clamped"],
        nan_fixed=counters["nan_fixed"],
    )
    clean_obs = Observation(
        frame_id=obs.frame_id,
        timestamp=obs.timestamp,
        footprint=clean_footprint,
        detections_ground=clean_detections,
        sensor_type=obs.sensor_type,
    )
    return clean_obs, report
