# =============================================================================
# test_trigger.py
# -----------------------------------------------------------------------------
# Responsible for: Pinning the located trigger's two hardest guarantees — (1) the
#                  detection-evidence cap (correlated frames are NOT re-multiplied,
#                  net evidence tops out at lr_max), and (2) firing requires BOTH
#                  persistence and windowed mass, so a single frame never trips it.
# Role in project: These are the behaviors that protect the broadcast from a false
#                  positive; each test states the scenario and why it matters.
# =============================================================================

from typing import Any, Dict, Tuple

import numpy as np
import pytest

from src.common.config import BrainConfig
from src.common.contracts import GroundDetection
from src.common.grid import GridSpec
from src.search.terrain import TerrainLayers
from src.search.trigger import Blob, BlobTracker, concentration_ratio, windowed_mass


@pytest.fixture
def cfg() -> BrainConfig:
    # persistence_n=3, p_located default 0.5; small grid for readable windows.
    return BrainConfig(n_rows=11, n_cols=11)


@pytest.fixture
def grid(cfg: BrainConfig) -> GridSpec:
    return GridSpec.from_config(cfg)


class FakeTerrain:
    """A no-op terrain so check_located can build terrain_context without the stub."""

    def layers(self, grid: GridSpec) -> TerrainLayers:
        n = (grid.n_rows, grid.n_cols)
        return TerrainLayers(accessibility=np.ones(n), corridor=np.ones(n))

    def context(self, grid: GridSpec, cell: Tuple[int, int]) -> Dict[str, Any]:
        return {"land_cover": "test"}


def det(cell, confidence):
    """Tiny helper to build a GroundDetection."""
    return GroundDetection(cell=cell, confidence=confidence)


# --- the evidence cap: correlated frames are not re-multiplied ---


def test_same_bucket_repeats_apply_no_extra_boost(cfg):
    """
    Scenario: three frames of the same blob, all in the modest bucket (LR 5).
    Why it matters: the core anti-double-count rule — only the FIRST frame may move
    the posterior; later same-bucket frames add persistence but no new multiplication.
    """
    tracker = BlobTracker(cfg)
    b1 = tracker.register([det((5, 5), 0.6)], frame_id="F1")
    b2 = tracker.register([det((5, 5), 0.6)], frame_id="F2")
    b3 = tracker.register([det((5, 5), 0.65)], frame_id="F3")  # still the 5.0 bucket
    assert b1 == [((5, 5), 5.0)]   # first frame applies LR 5
    assert b2 == []                # second frame: no boost
    assert b3 == []                # third frame: no boost
    # persistence still accrued across all three frames
    assert len(tracker.blobs[0].frame_ids) == 3


def test_stronger_bucket_applies_only_the_increment_and_caps_at_lr_max(cfg):
    """
    Scenario: a modest color hit (LR 5) then a strong thermal hit (LR 20).
    Why it matters: the upgrade must apply only 20/5 = 4 more, so the NET multiplier
    at the center is exactly lr_max (20), not 5*20 — capped, not compounded.
    """
    tracker = BlobTracker(cfg)
    b1 = tracker.register([det((5, 5), 0.6)], frame_id="F1")    # LR 5
    b2 = tracker.register([det((5, 5), 0.85)], frame_id="F2")   # LR 20 -> increment 4
    assert b1 == [((5, 5), 5.0)]
    assert b2 == [((5, 5), pytest.approx(4.0))]
    # net center multiplier = product of applied effective LRs == lr_max
    net = b1[0][1] * b2[0][1]
    assert net == pytest.approx(cfg.lr_max)
    assert tracker.blobs[0].applied_lr == pytest.approx(cfg.lr_max)


def test_subthreshold_detection_yields_no_boost(cfg):
    """
    Scenario: a weak detection below the LR floor (conf 0.3 -> LR 1).
    Why it matters: noise must not move the map; an LR of 1 means no boost emitted.
    """
    tracker = BlobTracker(cfg)
    assert tracker.register([det((5, 5), 0.3)], frame_id="F1") == []


# --- blob assignment ---


def test_nearby_detections_join_one_blob_far_ones_split(cfg):
    """
    Scenario: two detections within blob_radius, then one far away.
    Why it matters: scattered hits on one person must be a single blob (so persistence
    accrues), while a genuinely separate detection must start its own.
    """
    tracker = BlobTracker(cfg)
    tracker.register([det((5, 5), 0.6)], frame_id="F1")
    tracker.register([det((5, 7), 0.6)], frame_id="F2")  # within radius 3 of (5,5)
    assert len(tracker.blobs) == 1
    tracker.register([det((0, 0), 0.6)], frame_id="F3")  # far -> new blob
    assert len(tracker.blobs) == 2


def test_persistence_counts_distinct_frames_not_detections(cfg):
    """
    Scenario: two detections of one blob within a SINGLE frame.
    Why it matters: persistence is about repeated *looks*, so two boxes in one frame
    must count as one frame of evidence, not two.
    """
    tracker = BlobTracker(cfg)
    tracker.register([det((5, 5), 0.6), det((5, 6), 0.6)], frame_id="F1")
    assert len(tracker.blobs) == 1
    assert len(tracker.blobs[0].frame_ids) == 1


# --- the located gates: BOTH required ---


def _peaked_posterior(cfg, center, mass_in_window):
    """Build a posterior with `mass_in_window` concentrated at `center` (rest spread)."""
    p = np.full((cfg.n_rows, cfg.n_cols), 1e-6)
    p[center] = mass_in_window
    return p


def test_single_strong_frame_does_not_fire(cfg, grid):
    """
    Scenario: one very confident frame, with high windowed mass present.
    Why it matters: confidence alone must NOT declare a find — persistence (N frames)
    is the guard against a single false positive triggering the broadcast.
    """
    tracker = BlobTracker(cfg)
    tracker.register([det((5, 5), 0.95)], frame_id="F1")
    posterior = _peaked_posterior(cfg, (5, 5), 0.9)  # mass is there...
    event = tracker.check_located(posterior, grid, FakeTerrain(), timestamp=1.0)
    assert event is None  # ...but persistence (1 < 3) is not met


def test_persistent_but_diffuse_does_not_fire(cfg, grid):
    """
    Scenario: N frames of persistence, but the windowed mass stays below P_located.
    Why it matters: the other half of the AND — a persistent but unconcentrated blob
    is not a confident localization.
    """
    tracker = BlobTracker(cfg)
    for f in ("F1", "F2", "F3"):
        tracker.register([det((5, 5), 0.6)], frame_id=f)
    posterior = _peaked_posterior(cfg, (5, 5), 0.2)  # below P_located 0.5
    assert tracker.check_located(posterior, grid, FakeTerrain(), timestamp=1.0) is None


def test_fires_when_both_gates_met_and_only_once(cfg, grid):
    """
    Scenario: N frames of persistence AND windowed mass over P_located.
    Why it matters: the trigger must fire exactly once (latched), with a well-formed
    LocatedEvent carrying the cell, lat/lon, and terrain context.
    """
    tracker = BlobTracker(cfg)
    for f in ("F1", "F2", "F3"):
        tracker.register([det((5, 5), 0.6)], frame_id=f)
    posterior = _peaked_posterior(cfg, (5, 5), 0.7)  # over P_located
    event = tracker.check_located(posterior, grid, FakeTerrain(), timestamp=2.0)
    assert event is not None
    assert event.cell == (5, 5)
    assert event.confidence == pytest.approx(windowed_mass(posterior, (5, 5), cfg.window_halfwidth_cells))
    assert event.terrain_context == {"land_cover": "test"}
    assert event.timestamp == 2.0
    # second check does not re-fire
    assert tracker.check_located(posterior, grid, FakeTerrain(), timestamp=3.0) is None


# --- windowed_mass edge handling ---


def test_windowed_mass_clamps_at_grid_edge(cfg):
    """
    Scenario: a window centered in the corner (part of it off-grid).
    Why it matters: the window must clamp to bounds, not wrap NumPy's negative indices
    (which would sum the wrong cells).
    """
    p = np.ones((11, 11))
    # corner window of halfwidth 2 covers a 3x3 block (rows/cols 0..2) = 9 cells
    assert windowed_mass(p, (0, 0), halfwidth=2) == pytest.approx(9.0)


# --- concentration ratio (the grid-invariant relative gate) ---


def test_concentration_ratio_is_one_for_a_uniform_map():
    """
    Scenario: a perfectly uniform posterior.
    Why it matters: with no concentration anywhere, the ratio must be 1.0 — the
    baseline that means "no better than average," so it can never false-trigger.
    """
    p = np.full((50, 50), 1.0 / 2500)
    assert concentration_ratio(p, (25, 25), halfwidth=2) == pytest.approx(1.0)


def test_concentration_ratio_rises_as_mass_piles_into_the_window():
    """
    Scenario: progressively more mass concentrated at one cell.
    Why it matters: the ratio must increase as belief piles up, which is what lets it
    stand in for "the find region is anomalously dense."
    """
    base = np.full((50, 50), 1e-4)
    low = base.copy(); low[25, 25] = 1e-2
    high = base.copy(); high[25, 25] = 1e-1
    r_low = concentration_ratio(low, (25, 25), halfwidth=2)
    r_high = concentration_ratio(high, (25, 25), halfwidth=2)
    assert r_high > r_low > 1.0


def test_concentration_ratio_matches_hand_computed_value():
    """
    Scenario: a 3x3 peak holding 0.9 of the mass, the rest spread over the background.
    Why it matters: pins the exact formula (window density / map-average density) so a
    refactor can't silently change what the gate means. By hand: window (halfwidth 1 =
    9 cells) density = 0.9/9 = 0.1; map average = 1.0/100 = 0.01; ratio = 10.0.
    """
    p = np.full((10, 10), 0.1 / 91)        # 91 background cells share 0.1
    p[4:7, 4:7] = 0.9 / 9                   # 9 peak cells share 0.9
    assert concentration_ratio(p, (5, 5), halfwidth=1) == pytest.approx(10.0, rel=1e-6)


def test_concentration_ratio_is_far_more_grid_stable_than_absolute_mass():
    """
    Scenario: the SAME local peak-over-background contrast embedded in a 40x40 and a
    120x120 grid (the far field filled with representative background, not zeros).
    Why it matters: the honest, useful claim — a fixed window's ABSOLUTE mass collapses
    on the bigger grid, while the concentration RATIO stays the same order. It is much
    more transferable than p_located, even though it is not perfectly invariant.
    """
    def stats_for(n):
        p = np.full((n, n), 1.0)
        p[n // 2 - 1:n // 2 + 2, n // 2 - 1:n // 2 + 2] = 50.0  # 3x3 peak, 50x background
        p /= p.sum()
        center = (n // 2, n // 2)
        return windowed_mass(p, center, 2), concentration_ratio(p, center, 2)

    m_small, r_small = stats_for(40)
    m_large, r_large = stats_for(120)
    assert m_small / m_large > 5.0          # absolute windowed mass differs ~7x
    assert 0.5 < r_small / r_large < 2.0    # the ratio stays within ~2x (far steadier)


def test_concentration_ratio_zero_on_empty_map():
    """
    Scenario: an all-zero posterior.
    Why it matters: must return 0.0, not divide-by-zero, so the gate is simply not met.
    """
    assert concentration_ratio(np.zeros((10, 10)), (5, 5), halfwidth=2) == 0.0


def test_relative_gate_fires_on_large_grid_where_absolute_would_not():
    """
    Scenario: a large grid where a concentrated, persistent blob has tiny absolute
    windowed mass (well below p_located) but a high concentration ratio.
    Why it matters: this is the core Unit-2 win — on a realistic grid the relative gate
    must carry `located` even though the absolute mass gate never would.
    """
    big_cfg = BrainConfig(n_rows=120, n_cols=120, p_located=0.9, located_concentration_ratio=7.0)
    tracker = BlobTracker(big_cfg)
    peak = (60, 60)
    for f in ("F1", "F2", "F3"):
        tracker.register([GroundDetection(cell=peak, confidence=0.85)], frame_id=f)
    # a posterior concentrated at the peak but with small absolute windowed mass
    posterior = np.full((120, 120), 1e-5)
    posterior[58:63, 58:63] = 0.02  # a modest local pile, far below p_located=0.9
    posterior /= posterior.sum()    # normalize (no p_out here, just for the gate test)

    mass = windowed_mass(posterior, peak, big_cfg.window_halfwidth_cells)
    conc = concentration_ratio(posterior, peak, big_cfg.window_halfwidth_cells)
    assert mass < big_cfg.p_located          # absolute gate would NOT fire
    assert conc >= big_cfg.located_concentration_ratio  # but the relative gate does

    event = tracker.check_located(posterior, grid_for(big_cfg), FakeTerrain(), timestamp=1.0)
    assert event is not None and event.cell == peak


def grid_for(cfg):
    """Helper: a GridSpec matching a config (used by the large-grid trigger test)."""
    return GridSpec.from_config(cfg)
