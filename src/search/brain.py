# =============================================================================
# brain.py
# -----------------------------------------------------------------------------
# Responsible for: The search loop and the SINGLE WRITER of MapState. Holds the
#                  posterior + p_out + coverage, applies one Observation per step
#                  through the predict()->update() seam, drives the BlobTracker,
#                  and publishes a fresh MapState snapshot plus any LocatedEvent.
# Role in project: interfaces.md §5/§6. This is the convergence point of the loop:
#                  Observation in -> MapState/LocatedEvent out. Every other surface
#                  reads MapState; none writes it.
# Assumptions: Stationary subject for now, so predict() is a no-op (the seam for a
#              moving-target diffusion step is left explicit). Invariant held every
#              step: posterior.sum() + p_out == 1.
# =============================================================================

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.common.config import BrainConfig
from src.common.contracts import (
    LocatedEvent,
    MapState,
    Observation,
    SensorType,
    Status,
)
from src.common.grid import GridSpec
from src.search.planner import SectorPlanner
from src.search.prior import build_prior, rank_top_cells
from src.search.terrain import TerrainProvider
from src.search.trigger import BlobTracker
from src.search.update import (
    apply_coverage_and_nondetection,
    apply_detection_boost,
    renormalize,
)
from src.search.validation import sanitize_observation

# Module logger: warnings about malformed input go here (not print), so the demo
# and a future dashboard can route them without the brain knowing the sink.
logger = logging.getLogger(__name__)


class SearchBrain:
    """
    The probability-map brain: the only writer of search state.

    Why:
        Centralizing the posterior, the update orchestration, and the trigger in one
        object enforces the single-writer invariant by construction — consumers get
        immutable MapState snapshots, never a handle on the live arrays.
    """

    def __init__(self, cfg: BrainConfig, terrain: TerrainProvider) -> None:
        """
        Initialize the brain with the prior built from terrain + behavior.

        Args:
            cfg: All tunables.
            terrain: A TerrainProvider supplying the prior's A/C layers and the
                LocatedEvent terrain context.

        Why:
            The prior is the map's t0 state; building it here means the brain is
            "ready to read" the moment it exists, before any Observation arrives.
        """
        self._cfg = cfg
        self._terrain = terrain
        self.grid: GridSpec = GridSpec.from_config(cfg)
        # The search director (C1): a PURE, read-only sector planner. The brain composes it
        # to publish the recommended single-drone sweep in MapState.search_path; it never lets
        # the planner write state, so the single-writer invariant holds.
        self.planner = SectorPlanner(self.grid, cfg)

        # State the brain owns and is the sole writer of.
        self.posterior, self.p_out = build_prior(self.grid, cfg, terrain)
        self.coverage = np.zeros((self.grid.n_rows, self.grid.n_cols), dtype=float)
        # Per-sensor cumulative clearance (capped, interfaces §5.5). Kept separate per
        # sensor because correlated looks are within a sensor; the displayed `coverage`
        # is their combination. One array per SensorType value.
        self._cleared_by_sensor = {
            s.value: np.zeros((self.grid.n_rows, self.grid.n_cols), dtype=float)
            for s in SensorType
        }
        self.tracker = BlobTracker(cfg)
        self.update_count = 0
        self.timestamp = 0.0
        self.status = Status.SEARCHING
        self.detections_log: List[dict] = []
        self.last_event: Optional[LocatedEvent] = None
        # How many frames arrived malformed and had to be sanitized (real-data health).
        self.input_warning_count = 0

    # --- the loop seam: predict() then update() (interfaces §5.5) ---

    def predict(self) -> None:
        """
        The predict half of a Bayes filter — a no-op for now (stationary subject).

        Why:
            A proper moving-target search would spread the posterior by a diffusion/
            transition kernel here before each measurement. We leave the call in the
            loop so adding that later is purely additive (build seams, not futures);
            for a found-in-place demo, omitting the spread is defensible.
        """
        return None

    def step(self, observation: Observation) -> Optional[LocatedEvent]:
        """
        Advance the loop by one Observation (predict, then update).

        Args:
            observation: One frame's footprint + ground detections.

        Returns:
            A LocatedEvent if this step tripped the trigger, else None.

        Why:
            This is the single public entry point the pipeline calls per frame. Keeping
            predict()->update() ordering here (not in the caller) means the seam can't
            be wired wrong by a consumer.
        """
        self.predict()
        return self.update(observation)

    def update(self, observation: Observation) -> Optional[LocatedEvent]:
        """
        The measurement step: apply one Observation to the map (interfaces §5).

        Args:
            observation: The frame's footprint + ground detections + sensor type.

        Returns:
            A LocatedEvent if the trigger fired on this update, else None.

        Why:
            Orchestrates the §5 pieces in the one correct order — clear/cover, then
            apply capped detection boosts, then a SINGLE renormalization over cells and
            p_out, then test the trigger. Doing it here (not spread across callers)
            keeps the brain the single writer and the math atomic per frame.
        """
        # Sanitize at the boundary FIRST: a real detector/GeoReferencer can emit NaN
        # (projection divide-by-zero, missing telemetry) or out-of-range/off-grid
        # values, and a single NaN would otherwise poison the whole posterior on the
        # next renormalization. Degrade gracefully (sanitize-and-log), don't crash.
        observation, report = sanitize_observation(observation, self.grid)
        if not report.clean:
            self.input_warning_count += 1
            logger.warning("Sanitized frame %s: %s", observation.frame_id, report.summary())

        # After sanitization, detections are already finite and in-bounds.
        detections = observation.detections_ground
        exclude_cells = {d.cell for d in detections}
        recall = self._cfg.recall_for(observation.sensor_type.value)

        # 1) Non-detection clearing, capped per sensor (correlated looks — §5.5), into
        #    THIS sensor's clearance array; then derive the displayed coverage from all
        #    sensors' clearance.
        cleared = self._cleared_by_sensor[observation.sensor_type.value]
        apply_coverage_and_nondetection(
            self.posterior, cleared, observation.footprint, recall, exclude_cells,
            self.grid, self._cfg.clearance_cap_per_sensor,
        )
        self.coverage = self._combined_coverage()

        # 2) Capped detection boosts (the tracker decides the effective LR per blob,
        #    so correlated frames of one subject are not re-multiplied — §5.4).
        boosts = self.tracker.register(detections, observation.frame_id)
        for center_cell, effective_lr in boosts:
            apply_detection_boost(self.posterior, center_cell, effective_lr, self.grid, self._cfg)

        # 3) One renormalization over all cells AND p_out.
        self.posterior, self.p_out = renormalize(self.posterior, self.p_out)

        # 4) Bookkeeping + provenance for the dashboard timeline.
        self.update_count += 1
        self.timestamp = observation.timestamp
        for d in detections:
            self.detections_log.append(
                {
                    "frame_id": observation.frame_id,
                    "cell": list(d.cell),
                    "confidence": d.confidence,
                    "sensor_type": observation.sensor_type.value,
                    "timestamp": observation.timestamp,
                }
            )

        # 5) Trigger test (persistence AND windowed mass).
        event = self.tracker.check_located(self.posterior, self.grid, self._terrain, observation.timestamp)
        if event is not None:
            self.status = Status.LOCATED
            self.last_event = event
        return event

    # --- the read model (single source of truth, published as snapshots) ---

    def _combined_coverage(self) -> np.ndarray:
        """
        Combine the per-sensor clearance arrays into one "how cleared" coverage layer.

        Returns:
            (n_rows, n_cols) coverage in [0, 1]: 1 - Π_sensors (1 - cleared_sensor).

        Why:
            Different sensors clear independently (thermal sees through canopy that
            blocks color), so a cell's overall cleared-ness combines them as independent.
            Deriving coverage from the capped per-sensor arrays keeps the displayed
            "where have we cleared" consistent with what actually moved the posterior.
        """
        remaining = np.ones_like(self.posterior)
        for arr in self._cleared_by_sensor.values():
            remaining = remaining * (1.0 - arr)
        return 1.0 - remaining

    def _next_target(self) -> Optional[tuple]:
        """
        Choose where to direct the search next.

        Returns:
            The (row, col) of the most probable cell we have NOT yet cleared, or None
            on a degenerate map.

        Why:
            argmax(posterior * (1 - coverage)) is a one-line, sensible director: it
            prefers high-probability ground we still haven't searched, so as the sweep
            clears the corridor the target naturally moves on. A simple seam — a real
            path planner would replace this without touching MapState.
        """
        score = self.posterior * (1.0 - self.coverage)
        if score.max() <= 0.0:
            return None
        idx = int(np.argmax(score))
        return (idx // self.grid.n_cols, idx % self.grid.n_cols)

    def map_state(self) -> MapState:
        """
        Publish an immutable snapshot of the current search state.

        Returns:
            A MapState with copies of the posterior/coverage arrays (so a consumer
            never sees a half-written array on the next update), ranked top cells, the
            next target, and the status.

        Why:
            MapState is the read model every other surface consumes. Copying the arrays
            is what makes "single writer, many readers" safe in-process and keeps the
            snapshot serializable for a later socket/Redis boundary unchanged.
        """
        top_cells = rank_top_cells(self.posterior, k=5)
        # The recommended single-drone sweep over the top-priority sector (C1). Computed from
        # the published belief, so consumers (dashboard/operator) see where the director would
        # send one drone next. The multi-drone assignment is an orchestration layer above this.
        plan = self.planner.plan_single(self.posterior, self.coverage)
        search_path = plan.waypoints if plan is not None else None
        return MapState(
            grid_spec=self.grid,
            update_count=self.update_count,
            timestamp=self.timestamp,
            posterior=self.posterior.copy(),
            coverage=self.coverage.copy(),
            top_cells=top_cells,
            next_target=self._next_target(),
            status=self.status,
            search_path=search_path,
            detections_log=list(self.detections_log),
            p_out=self.p_out,
        )
