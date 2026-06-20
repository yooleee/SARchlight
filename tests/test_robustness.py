# =============================================================================
# test_robustness.py
# -----------------------------------------------------------------------------
# Responsible for: Whole-brain behavior under MESSY / adversarial Observation
#                  streams — the gap our happy-path tests left. Each test here was
#                  first found by a manual probe; this file makes them permanent
#                  regression tests so the proven behavior can't silently break.
# Role in project: integration-level (drives SearchBrain end to end). Unit-level
#                  pieces live in test_validation / test_trigger / test_update;
#                  here we assert the system-level invariants and dynamics.
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
from src.search.trigger import concentration_ratio, windowed_mass


class UniformTerrain:
    """Flat, fully-passable terrain so these tests exercise the loop, not the prior shape."""

    def layers(self, grid: GridSpec) -> TerrainLayers:
        n = (grid.n_rows, grid.n_cols)
        return TerrainLayers(accessibility=np.ones(n), corridor=np.ones(n))

    def context(self, grid: GridSpec, cell: Tuple[int, int]) -> Dict[str, Any]:
        return {"land_cover": "test"}


@pytest.fixture
def cfg() -> BrainConfig:
    # A small grid keeps these tests fast. The relative gate's threshold is LOWERED here
    # (an isolated detection on a tiny uniform grid only reaches ~4.5x, vs ~10x in the
    # demo where clearing + a structured prior amplify it) — because these tests exercise
    # the blob/persistence/edge LOGIC, not the production threshold calibration (that is
    # covered by the demo and test_trigger). Persistence still gates every locate.
    return BrainConfig(n_rows=20, n_cols=20, located_concentration_ratio=3.0)


def _detection(frame_id, cell, conf, t, sensor=SensorType.COLOR):
    fp = [CellCoverage(cell=cell, coverage_fraction=1.0, visibility_weight=0.85)]
    return Observation(frame_id, t, fp, [GroundDetection(cell, conf)], sensor)


def _clean(frame_id, cells, t):
    fp = [CellCoverage(cell=rc, coverage_fraction=1.0, visibility_weight=1.0) for rc in cells]
    return Observation(frame_id, t, fp, [], SensorType.COLOR)


def _invariants_ok(brain: SearchBrain) -> bool:
    """Mass==1, no NaN, no negatives — the conditions every step must preserve."""
    p = brain.posterior
    return (not np.isnan(p).any()) and (p >= 0).all() and abs(p.sum() + brain.p_out - 1.0) < 1e-6


# --- transient false positive: recovers, never declares located ---


def test_transient_false_positive_recovers_and_never_locates(cfg):
    """
    Scenario: one strong but spurious detection (conf 0.9), then several clean passes
    over that cell.
    Why it matters: a lone false positive must not declare a find. Two things protect
    us, and this checks both: persistence (1 frame < N) keeps status `searching`, and
    the subsequent non-detections drain the spurious spike back down.
    (Limitation, by design: a detector that fires on the SAME rock every frame is a
    detector problem, not the brain's — the LR cap bounds the damage; see followups.)
    """
    brain = SearchBrain(cfg, UniformTerrain())
    fp_cell = (10, 10)

    event = brain.step(_detection("FP", fp_cell, conf=0.9, t=0.0))
    spiked = brain.posterior[fp_cell]
    assert event is None and brain.status == Status.SEARCHING

    for k in range(8):
        brain.step(_clean(f"C{k}", [fp_cell], t=1.0 + k))

    assert brain.status == Status.SEARCHING            # never located on a transient FP
    assert brain.posterior[fp_cell] < spiked           # the spike was drained back down
    assert _invariants_ok(brain)


# --- two real subjects, tracked and located independently ---


def test_two_separate_subjects_locate_independently(cfg):
    """
    Scenario: two well-separated cells each get a persistent run of strong detections.
    Why it matters: real footage can contain more than one person; the brain must track
    disjoint blobs and emit a LocatedEvent for each, not merge or miss one.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    a, b = (4, 4), (15, 16)
    events = []
    for k in range(3):
        events.append(brain.step(_detection(f"A{k}", a, 0.88, t=float(k), sensor=SensorType.THERMAL)))
    for k in range(3):
        events.append(brain.step(_detection(f"B{k}", b, 0.88, t=10.0 + k, sensor=SensorType.THERMAL)))

    fired = [e for e in events if e is not None]
    assert len(brain.tracker.blobs) == 2                  # two disjoint blobs tracked
    assert {e.cell for e in fired} == {a, b}             # both located, at their own cells


# --- intermittent (flickering) detections still accrue persistence ---


def test_intermittent_detections_still_reach_persistence(cfg):
    """
    Scenario: detections on the same blob appear on frames 1, 3, 5 — with the drone
    looking elsewhere on frames 2 and 4 (no clean look over the blob).
    Why it matters: a real detector flickers; persistence counts DISTINCT detection
    frames, so non-consecutive hits must still accumulate and eventually locate.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    blob = (9, 9)
    elsewhere = [(1, 1), (1, 2)]
    events = [
        brain.step(_detection("F1", blob, 0.86, 1.0, SensorType.THERMAL)),
        brain.step(_clean("F2", elsewhere, 2.0)),
        brain.step(_detection("F3", blob, 0.86, 3.0, SensorType.THERMAL)),
        brain.step(_clean("F4", elsewhere, 4.0)),
        brain.step(_detection("F5", blob, 0.86, 5.0, SensorType.THERMAL)),
    ]
    assert len(brain.tracker.blobs[0].frame_ids) == 3     # 3 distinct detection frames
    assert any(e is not None and e.cell == blob for e in events)


# --- off-grid data flows through the brain without error ---


def test_offgrid_observation_does_not_crash_and_is_counted(cfg):
    """
    Scenario: an Observation with off-grid footprint cells and an off-grid detection.
    Why it matters: a real GeoReferencer can project a box just off the region edge;
    the brain must drop those safely (no NumPy negative-index wrap, no crash), keep the
    mass invariant, and record that the frame needed sanitizing.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    obs = Observation(
        "EDGE", 1.0,
        footprint=[CellCoverage((-3, 5), 1.0, 1.0), CellCoverage((5, 5), 1.0, 1.0)],
        detections_ground=[GroundDetection((999, 999), 0.8)],
        sensor_type=SensorType.COLOR,
    )
    brain.step(obs)
    assert _invariants_ok(brain)
    assert brain.input_warning_count == 1


# --- subject at the grid corner still locates ---


def test_subject_at_grid_corner_still_locates(cfg):
    """
    Scenario: the subject sits at cell (0, 0), so the trigger window clamps to a corner.
    Why it matters: edge subjects must not be un-findable. The window is smaller
    (clamped to bounds), but the trigger still evaluates on the clamped window and fires.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    corner = (0, 0)
    located = None
    for k in range(4):
        sensor = SensorType.THERMAL if k >= 2 else SensorType.COLOR
        conf = 0.88 if k >= 2 else 0.62
        ev = brain.step(_detection(f"E{k}", corner, conf, float(k), sensor))
        located = located or ev
    assert located is not None and located.cell == corner
    assert _invariants_ok(brain)


# --- soak: invariants hold over a long random stream ---


def test_soak_invariants_hold_over_long_random_stream(cfg):
    """
    Scenario: 400 random but VALID frames (random covered cell, random coverage/
    visibility, occasional random detection), seeded for reproducibility.
    Why it matters: many renormalizations could accumulate float error or drift; this
    pins numerical stability and that status only ever advances searching -> located.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    rng = np.random.default_rng(0)
    seen_located = False
    for i in range(400):
        r, c = int(rng.integers(0, 20)), int(rng.integers(0, 20))
        fp = [CellCoverage((r, c), float(rng.random()), float(rng.random()))]
        dets = [GroundDetection((r, c), float(rng.random()))] if rng.random() < 0.1 else []
        brain.step(Observation(f"S{i}", float(i), fp, dets, SensorType.COLOR))
        assert _invariants_ok(brain), f"invariant broke at frame {i}"
        if brain.status == Status.LOCATED:
            seen_located = True
        else:
            assert not seen_located, "status regressed from located back to searching"


# --- fuzz: malformed frames interleaved, invariants still hold ---


def test_soak_with_malformed_frames_never_poisons_or_crashes(cfg):
    """
    Scenario: 300 frames where ~15% inject a NaN or out-of-range value (the kind a real
    detector/geo can emit), interleaved with valid frames.
    Why it matters: the sanitize-and-log defense must hold under sustained bad input —
    no NaN ever reaches the posterior, mass stays 1, and the health counter records the
    bad frames. This stresses the Unit-1 guard and the loop together.
    """
    brain = SearchBrain(cfg, UniformTerrain())
    rng = np.random.default_rng(7)
    bad_frames = 0
    for i in range(300):
        r, c = int(rng.integers(0, 20)), int(rng.integers(0, 20))
        cov, vis = float(rng.random()), float(rng.random())
        roll = rng.random()
        if roll < 0.15:
            bad_frames += 1
            # inject one of several malformations
            choice = rng.integers(0, 3)
            if choice == 0:
                cov = float("nan")
            elif choice == 1:
                vis = 5.0          # out of range
            else:
                r = -1             # off grid
        fp = [CellCoverage((r, c), cov, vis)]
        brain.step(Observation(f"Z{i}", float(i), fp, [], SensorType.COLOR))
        assert _invariants_ok(brain), f"invariant broke at frame {i}"
    assert brain.input_warning_count >= 1   # bad frames were detected and handled
    assert bad_frames >= 1                  # the test actually exercised the bad path
