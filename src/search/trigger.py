# =============================================================================
# trigger.py
# -----------------------------------------------------------------------------
# Responsible for: The located trigger (interfaces.md §5.4) and the detection-
#                  evidence cap. Tracks lightweight detection "blobs" across
#                  frames, decides the *effective* LR to apply per blob (so a run
#                  of correlated hits on the same person is ONE accumulating piece
#                  of evidence, capped at lr_max — never re-multiplied per frame),
#                  counts persistence, and fires a LocatedEvent only when both the
#                  windowed posterior mass and the persistence gate are met.
# Role in project: Sits beside the brain. The brain asks the tracker which boosts
#                  to apply (it remains the single writer that applies them), then
#                  asks whether the trigger has tripped.
# Assumptions: A cell is (row, col). Persistence counts DISTINCT frames, not
#              detections, so multiple boxes in one frame don't inflate it.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from src.common.config import BrainConfig
from src.common.contracts import Cell, GroundDetection, LocatedEvent
from src.common.grid import GridSpec
from src.search.terrain import TerrainProvider
from src.search.update import lr_for_confidence


@dataclass
class Blob:
    """
    A tracked cluster of detections believed to be the same subject across frames.

    Args (fields):
        peak_cell: Representative (row, col) — where boosts and the window center.
        frame_ids: Distinct frames that hit this blob (len == persistence).
        max_lr: Highest LR bucket any of its detections has reached.
        applied_lr: LR already multiplied into the posterior for this blob.
        max_confidence: Strongest raw detector confidence seen (for logging).
        fired: Whether this blob has already emitted a LocatedEvent.

    Why:
        Holding applied_lr separately from max_lr is what implements the cap: we only
        ever apply the *difference*, so cumulative evidence tops out at lr_max no
        matter how many correlated frames arrive. Persistence (len(frame_ids)) is
        confirmation logic ON TOP of the posterior, not more multiplications into it.
    """

    peak_cell: Cell
    frame_ids: Set[str] = field(default_factory=set)
    max_lr: float = 1.0
    applied_lr: float = 1.0
    max_confidence: float = 0.0
    fired: bool = False


def _window(posterior: np.ndarray, center: Cell, halfwidth: int) -> np.ndarray:
    """
    Return the square sub-array around a center cell, clamped to the grid bounds.

    Args:
        posterior: (n_rows, n_cols) probability array.
        center: (row, col) center of the window.
        halfwidth: Window half-width in cells (a value of 2 -> a 5x5 window).

    Returns:
        A view of the posterior over the (clamped) window.

    Why:
        Both windowed_mass and concentration_ratio need the same clamped window;
        factoring it here keeps them consistent and avoids duplicating the bounds
        math (DRY). Clamping (not wrapping) matters near the grid edge.
    """
    r, c = center
    n_rows, n_cols = posterior.shape
    r0, r1 = max(0, r - halfwidth), min(n_rows, r + halfwidth + 1)
    c0, c1 = max(0, c - halfwidth), min(n_cols, c + halfwidth + 1)
    return posterior[r0:r1, c0:c1]


def windowed_mass(posterior: np.ndarray, center: Cell, halfwidth: int) -> float:
    """
    Sum the posterior over a square window around a center cell (clamped to bounds).

    Args:
        posterior: (n_rows, n_cols) probability array.
        center: (row, col) center of the window.
        halfwidth: Window half-width in cells (a value of 2 -> a 5x5 window).

    Returns:
        The summed probability mass in the window.

    Why:
        The trigger tests aggregated mass over a window, not one cell's p_i, so it is
        invariant to cell_size_m — refining the grid splits a peak across more cells
        but the windowed sum is unchanged. (interfaces §5.4 criterion 1, absolute gate.)
    """
    return float(_window(posterior, center, halfwidth).sum())


def concentration_ratio(posterior: np.ndarray, center: Cell, halfwidth: int) -> float:
    """
    How many times denser the window around `center` is than the map's average cell.

    Args:
        posterior: (n_rows, n_cols) probability array.
        center: (row, col) center of the window.
        halfwidth: Window half-width in cells.

    Returns:
        density_in_window / density_over_whole_map. 1.0 = no concentration (uniform);
        large values = belief is piled into this window. 0.0 on a zero-mass map.

    Why:
        This is the RELATIVE trigger signal (interfaces §5.4's relative criterion). An
        absolute windowed-mass threshold is brittle because a fixed window holds a much
        smaller absolute share of a larger grid; comparing window density to the map's
        average density largely cancels that out, so the threshold transfers FAR better
        across regions/resolutions than a raw mass would (it is much more grid-stable,
        though not perfectly invariant — an extreme peak holding most of the mass still
        scales with grid size). It answers "is the find region anomalously dense?"
        rather than "did mass cross a magic number we tuned to one grid?".
    """
    window = _window(posterior, center, halfwidth)
    total = float(posterior.sum())
    if total <= 0.0 or window.size == 0:
        return 0.0
    density_window = float(window.sum()) / window.size
    density_average = total / posterior.size
    return density_window / density_average


class BlobTracker:
    """
    Tracks detection blobs, supplies capped per-blob boosts, and tests the located
    condition.

    Why:
        Separating the persistence/cap logic from the Bayesian math (update.py) keeps
        each piece single-purpose: update.py never double-counts because it is only
        ever handed an already-capped effective LR; this class never touches the
        posterior, so the brain stays the single writer.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        self._cfg = cfg
        self.blobs: List[Blob] = []

    def _find_blob(self, cell: Cell) -> Optional[Blob]:
        """
        Find the nearest existing blob within blob_radius of a cell.

        Args:
            cell: The detection's (row, col).

        Returns:
            The closest blob within blob_radius_cells, or None to start a new one.

        Why:
            At altitude a few pixels of error is tens of meters, so detections of one
            person scatter across nearby cells; grouping them within a radius is the
            weekend-appropriate stand-in for a full detection track.
        """
        best: Optional[Blob] = None
        best_dist = float(self._cfg.blob_radius_cells)
        for blob in self.blobs:
            dist = math.hypot(cell[0] - blob.peak_cell[0], cell[1] - blob.peak_cell[1])
            if dist <= self._cfg.blob_radius_cells and (best is None or dist < best_dist):
                best, best_dist = blob, dist
        return best

    def register(self, detections: List[GroundDetection], frame_id: str) -> List[Tuple[Cell, float]]:
        """
        Ingest a frame's detections; return the capped boosts the brain should apply.

        Args:
            detections: This frame's ground detections.
            frame_id: The frame's id (so persistence counts distinct frames).

        Returns:
            A list of (center_cell, effective_lr) boosts. effective_lr is the
            INCREMENTAL factor (>= 1) that brings a blob's applied evidence up to its
            best bucket — 1.0 (omitted) when the bucket was already applied.

        Why:
            This is the cap in action: the first detection of a blob applies its LR;
            later frames at the same or lower bucket add nothing (effective == 1, so
            no boost is emitted); a later, stronger frame applies only the difference.
            Net applied evidence per blob never exceeds lr_max.
        """
        boosts: List[Tuple[Cell, float]] = []
        for det in detections:
            blob = self._find_blob(det.cell)
            if blob is None:
                blob = Blob(peak_cell=det.cell)
                self.blobs.append(blob)
            blob.frame_ids.add(frame_id)
            blob.max_confidence = max(blob.max_confidence, det.confidence)
            lr = lr_for_confidence(det.confidence, self._cfg)
            blob.max_lr = max(blob.max_lr, lr)
            # Apply only the increment beyond what's already in the posterior.
            effective_lr = blob.max_lr / blob.applied_lr
            if effective_lr > 1.0 + 1e-12:
                boosts.append((blob.peak_cell, effective_lr))
                blob.applied_lr = blob.max_lr
        return boosts

    def check_located(
        self,
        posterior: np.ndarray,
        grid: GridSpec,
        terrain: TerrainProvider,
        timestamp: float,
    ) -> Optional[LocatedEvent]:
        """
        Test every blob for the located condition; emit a LocatedEvent for the first
        that newly trips (persistence AND windowed mass).

        Args:
            posterior: The current (already renormalized) posterior.
            grid: GridSpec, for the cell-center lat/lon.
            terrain: TerrainProvider, for the event's terrain_context.
            timestamp: The current frame time, stamped on the event.

        Returns:
            A LocatedEvent if a blob just satisfied the gates, else None.

        Why:
            The persistence gate plus a CONCENTRATION gate is what protects the
            broadcast from a single false positive: a strong but one-off detection
            lacks persistence; a persistent-but-diffuse blob lacks concentration. The
            concentration gate (relative) is grid-invariant where the absolute mass
            gate is not, so they are OR'd — either suffices once persistence is met.
            fired is latched so the event emits exactly once.
        """
        halfwidth = self._cfg.window_halfwidth_cells
        for blob in self.blobs:
            if blob.fired:
                continue
            # Persistence gate: detections across N distinct overlapping frames.
            if len(blob.frame_ids) < self._cfg.persistence_n:
                continue
            # Concentration gate (two OR'd forms around the peak):
            #   - absolute: windowed posterior mass crosses p_located, and/or
            #   - relative: window is located_concentration_ratio x denser than average.
            mass = windowed_mass(posterior, blob.peak_cell, halfwidth)
            concentration = concentration_ratio(posterior, blob.peak_cell, halfwidth)
            if mass >= self._cfg.p_located or concentration >= self._cfg.located_concentration_ratio:
                blob.fired = True
                lat, lon = grid.cell_to_latlon(*blob.peak_cell)
                return LocatedEvent(
                    cell=blob.peak_cell,
                    latlon=(lat, lon),
                    confidence=float(mass),  # windowed posterior mass at trigger time
                    terrain_context=terrain.context(grid, blob.peak_cell),
                    timestamp=timestamp,
                )
        return None
