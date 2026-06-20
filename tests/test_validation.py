# =============================================================================
# test_validation.py
# -----------------------------------------------------------------------------
# Responsible for: Proving the Observation-boundary sanitizer (Unit 1) — the fix
#                  for the confirmed bug where one NaN poisons the whole posterior.
# Role in project: Tests both the pure sanitizer AND the brain wired to it, so the
#                  "sanitize-and-log, never crash, never poison" policy is locked.
# =============================================================================

import logging

import numpy as np
import pytest

from src.common.config import BrainConfig
from src.common.contracts import (
    CellCoverage,
    GroundDetection,
    Observation,
    SensorType,
)
from src.common.grid import GridSpec
from src.search.brain import SearchBrain
from src.search.terrain import SyntheticTerrain
from src.search.validation import sanitize_observation


@pytest.fixture
def cfg() -> BrainConfig:
    return BrainConfig(n_rows=20, n_cols=20)


@pytest.fixture
def grid(cfg: BrainConfig) -> GridSpec:
    return GridSpec.from_config(cfg)


def _obs(footprint, detections, sensor=SensorType.COLOR):
    return Observation("F1", 1.0, footprint, detections, sensor)


# --- the pure sanitizer ---


def test_nan_coverage_becomes_zero_and_is_reported(grid):
    """
    Scenario: a footprint cell with NaN coverage_fraction.
    Why it matters: NaN must map to 0.0 (an unknown look should not clear the cell)
    and be tallied, not silently leaked downstream where it poisons renormalization.
    """
    obs = _obs([CellCoverage((5, 5), coverage_fraction=float("nan"), visibility_weight=1.0)], [])
    clean, report = sanitize_observation(obs, grid)
    assert clean.footprint[0].coverage_fraction == 0.0
    assert report.nan_fixed == 1
    assert not report.clean


def test_out_of_range_values_are_clamped(grid):
    """
    Scenario: coverage 2.0, visibility -0.5, confidence 5.0.
    Why it matters: values must be pulled into [0,1] so the math's assumptions hold,
    and the clamps must be counted.
    """
    obs = _obs(
        [CellCoverage((5, 5), coverage_fraction=2.0, visibility_weight=-0.5)],
        [GroundDetection((6, 6), confidence=5.0)],
    )
    clean, report = sanitize_observation(obs, grid)
    assert clean.footprint[0].coverage_fraction == 1.0
    assert clean.footprint[0].visibility_weight == 0.0
    assert clean.detections_ground[0].confidence == 1.0
    assert report.clamped == 3


def test_offgrid_cells_and_detections_are_dropped(grid):
    """
    Scenario: a footprint cell and a detection both off the grid.
    Why it matters: off-grid indices would wrap NumPy's negative indexing silently;
    they must be dropped (and counted), leaving only valid cells.
    """
    obs = _obs(
        [CellCoverage((-1, 5), 1.0, 1.0), CellCoverage((5, 5), 1.0, 1.0)],
        [GroundDetection((999, 999), 0.8)],
    )
    clean, report = sanitize_observation(obs, grid)
    assert len(clean.footprint) == 1 and clean.footprint[0].cell == (5, 5)
    assert clean.detections_ground == []
    assert report.dropped_cells == 1
    assert report.dropped_detections == 1


def test_nan_confidence_detection_is_dropped(grid):
    """
    Scenario: a detection with NaN confidence.
    Why it matters: an untrustworthy detection must be dropped explicitly, not kept
    and silently treated as inert evidence (LR=1) — dropping is visible in the report.
    """
    obs = _obs([], [GroundDetection((5, 5), confidence=float("nan"))])
    clean, report = sanitize_observation(obs, grid)
    assert clean.detections_ground == []
    assert report.dropped_detections == 1


def test_malformed_cell_coordinates_are_dropped(grid):
    """
    Scenario: footprint/detection cells with non-integer or non-unpackable coordinates
    (a float-valued cell from a buggy projection, or a malformed tuple).
    Why it matters: such cells would raise or silently mis-index on the posterior; they
    must be dropped, not trusted. (The contract is integer (row, col).)
    """
    obs = _obs(
        [
            CellCoverage(cell=(1.5, 2.0), coverage_fraction=1.0, visibility_weight=1.0),  # float coords
            CellCoverage(cell=(5, 5), coverage_fraction=1.0, visibility_weight=1.0),       # valid
        ],
        [GroundDetection(cell=5, confidence=0.8)],  # not even a pair -> unpack fails
    )
    clean, report = sanitize_observation(obs, grid)
    assert [cc.cell for cc in clean.footprint] == [(5, 5)]
    assert clean.detections_ground == []
    assert report.dropped_cells == 1
    assert report.dropped_detections == 1


def test_clean_observation_passes_through_unchanged(grid):
    """
    Scenario: a perfectly valid Observation.
    Why it matters: the common case must be a no-op with a `clean` report, so logging
    only fires on genuine problems.
    """
    obs = _obs(
        [CellCoverage((5, 5), 0.8, 0.7)],
        [GroundDetection((5, 5), 0.6)],
    )
    clean, report = sanitize_observation(obs, grid)
    assert report.clean
    assert clean.footprint[0].coverage_fraction == 0.8
    assert clean.detections_ground[0].confidence == 0.6


# --- the brain wired to the sanitizer (the bug is closed end to end) ---


def test_nan_frame_does_not_poison_the_posterior(cfg):
    """
    Scenario: PROBE 1 reproduced — a NaN coverage value reaches the brain.
    Why it matters: this is the confirmed bug. After the fix the posterior must stay
    finite, mass must remain 1, and the warning counter must record the bad frame.
    """
    brain = SearchBrain(cfg, SyntheticTerrain(cfg))
    bad = _obs([CellCoverage((10, 10), coverage_fraction=float("nan"), visibility_weight=1.0)], [])
    brain.step(bad)
    assert not np.isnan(brain.posterior).any()
    assert brain.posterior.sum() + brain.p_out == pytest.approx(1.0)
    assert brain.input_warning_count == 1


def test_brain_logs_a_warning_on_bad_input(cfg, caplog):
    """
    Scenario: a malformed frame reaches the brain.
    Why it matters: the sanitize-and-log policy must actually log (so a real upstream
    bug is visible in the demo), not silently swallow the problem.
    """
    brain = SearchBrain(cfg, SyntheticTerrain(cfg))
    with caplog.at_level(logging.WARNING):
        brain.step(_obs([CellCoverage((10, 10), float("nan"), 1.0)], []))
    assert any("Sanitized frame" in rec.message for rec in caplog.records)


def test_clean_frames_do_not_increment_warnings(cfg):
    """
    Scenario: only valid frames.
    Why it matters: the health counter must stay at 0 on clean input, so it's a
    meaningful signal of real-data trouble rather than noise.
    """
    brain = SearchBrain(cfg, SyntheticTerrain(cfg))
    brain.step(_obs([CellCoverage((5, 5), 1.0, 1.0)], [GroundDetection((5, 5), 0.6)]))
    assert brain.input_warning_count == 0
