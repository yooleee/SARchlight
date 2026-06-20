# =============================================================================
# test_brain.py
# -----------------------------------------------------------------------------
# Responsible for: Integration of the whole loop through the single writer —
#                  that a non-detection clears and drains into p_out, that a
#                  persistent detection flips status to LOCATED with a
#                  LocatedEvent, and that the mass invariant holds every step.
# Role in project: Proves the wiring (prior + update + trigger) before the demo
#                  drives it on the scripted scenario.
# =============================================================================

from typing import Any, Dict, Tuple

import numpy as np
import pytest

from src.common.config import BrainConfig
from src.common.contracts import (
    CellCoverage,
    GroundDetection,
    Observation,
    SensorType,
    Status,
)
from src.common.grid import GridSpec
from src.search.brain import SearchBrain
from src.search.terrain import TerrainLayers


class UniformTerrain:
    """Flat, fully-passable terrain so the test exercises the loop, not the prior shape."""

    def layers(self, grid: GridSpec) -> TerrainLayers:
        n = (grid.n_rows, grid.n_cols)
        return TerrainLayers(accessibility=np.ones(n), corridor=np.ones(n))

    def context(self, grid: GridSpec, cell: Tuple[int, int]) -> Dict[str, Any]:
        return {"land_cover": "test", "near_drainage": False}


@pytest.fixture
def cfg() -> BrainConfig:
    # Small grid so a single boost concentrates enough windowed mass to cross P_located.
    return BrainConfig(n_rows=7, n_cols=7)


def _nondetection_frame(frame_id, cells, t):
    """A clean look over `cells` (full coverage/visibility), no detections."""
    fp = [CellCoverage(cell=rc, coverage_fraction=1.0, visibility_weight=1.0) for rc in cells]
    return Observation(frame_id=frame_id, timestamp=t, footprint=fp,
                       detections_ground=[], sensor_type=SensorType.COLOR)


def _detection_frame(frame_id, cell, conf, t, sensor=SensorType.COLOR):
    """A frame with a detection at `cell` (and that cell covered)."""
    fp = [CellCoverage(cell=cell, coverage_fraction=1.0, visibility_weight=1.0)]
    det = [GroundDetection(cell=cell, confidence=conf)]
    return Observation(frame_id=frame_id, timestamp=t, footprint=fp,
                       detections_ground=det, sensor_type=sensor)


def test_initial_state_is_searching_and_normalized(cfg):
    """
    Scenario: a freshly built brain, before any observation.
    Why it matters: the t0 read model must already be valid — status searching,
    mass == 1, nothing covered.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    ms = brain.map_state()
    assert ms.status == Status.SEARCHING
    assert ms.update_count == 0
    assert ms.posterior.sum() + ms.p_out == pytest.approx(1.0)
    assert ms.coverage.sum() == 0.0


def test_nondetection_clears_and_raises_pout(cfg):
    """
    Scenario: one clean sweep over a block of cells.
    Why it matters: the loop's clearing behavior end-to-end — covered cells dim,
    coverage rises, p_out rises, mass stays 1.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    p_out_before = brain.p_out
    covered = [(r, c) for r in range(0, 3) for c in range(0, 3)]
    before = brain.posterior[0, 0]

    brain.step(_nondetection_frame("F1", covered, t=1.0))
    ms = brain.map_state()

    assert ms.posterior[0, 0] < before          # covered cell dimmed
    assert ms.coverage[0, 0] > 0.0              # and is now marked looked-at
    assert ms.p_out > p_out_before              # 'they left' grew
    assert ms.posterior.sum() + ms.p_out == pytest.approx(1.0)
    assert ms.update_count == 1


def test_persistent_detection_locates_and_emits_event(cfg):
    """
    Scenario: three frames of detections on the same cell (color, then thermal).
    Why it matters: the headline behavior — status stays searching until persistence
    is met, then flips to LOCATED with a well-formed LocatedEvent; mass stays 1.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    subject = (3, 3)

    e1 = brain.step(_detection_frame("F1", subject, conf=0.7, t=1.0))
    assert e1 is None and brain.status == Status.SEARCHING  # 1 frame: no persistence

    e2 = brain.step(_detection_frame("F2", subject, conf=0.72, t=2.0))
    assert e2 is None and brain.status == Status.SEARCHING  # 2 frames: still short

    e3 = brain.step(_detection_frame("F3", subject, conf=0.85, t=3.0, sensor=SensorType.THERMAL))
    assert e3 is not None                                   # 3rd frame trips it
    assert brain.status == Status.LOCATED
    assert e3.cell == subject
    assert e3.latlon == brain.grid.cell_to_latlon(*subject)

    ms = brain.map_state()
    assert ms.status == Status.LOCATED
    assert ms.posterior.sum() + ms.p_out == pytest.approx(1.0)  # invariant survived locating
    assert len(ms.detections_log) == 3                          # all three detections logged


def test_map_state_is_a_snapshot_not_a_live_handle(cfg):
    """
    Scenario: grab a MapState, then advance the brain.
    Why it matters: consumers must see a stable snapshot — the single writer mutating
    its arrays afterward must not retro-change a MapState someone already holds.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    snap = brain.map_state()
    snap_sum = snap.posterior.sum()
    brain.step(_nondetection_frame("F1", [(0, 0), (0, 1)], t=1.0))
    # the earlier snapshot is unchanged by the later update
    assert snap.posterior.sum() == pytest.approx(snap_sum)
    assert snap.update_count == 0
