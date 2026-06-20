# =============================================================================
# test_update.py
# -----------------------------------------------------------------------------
# Responsible for: Proving the Bayesian per-cell update (src/search/update.py)
#                  behaves correctly BEFORE any of it is wired into the loop — the
#                  highest-risk core, so it gets the most scrutiny.
# Role in project: Each test states the search-theory behavior it pins down and
#                  why that behavior matters for the map's reasoning.
# =============================================================================

import numpy as np
import pytest

from src.common.config import BrainConfig
from src.common.contracts import CellCoverage
from src.common.grid import GridSpec
from src.search.update import (
    apply_coverage_and_nondetection,
    apply_detection_boost,
    detection_probability,
    gaussian_weight_kernel,
    lr_for_confidence,
    renormalize,
)


@pytest.fixture
def cfg() -> BrainConfig:
    """A small grid keeps the arrays readable; everything else is the real default."""
    # 11x11 so a detection at (5,5) has room for the kernel radius on every side.
    return BrainConfig(n_rows=11, n_cols=11)


@pytest.fixture
def grid(cfg: BrainConfig) -> GridSpec:
    return GridSpec.from_config(cfg)


# --- LR(c): the bucketed, clipped likelihood ratio ---


def test_lr_buckets_pick_the_highest_threshold_met(cfg):
    """
    Scenario: confidences spanning the bucket boundaries.
    Why it matters: confidence is uncalibrated, so the step function (not a smooth
    curve) is the contract — and a sub-threshold detection must be inert (LR == 1).
    """
    assert lr_for_confidence(0.30, cfg) == 1.0   # below the 0.4 floor -> ignored
    assert lr_for_confidence(0.40, cfg) == 5.0   # exactly on the modest bucket
    assert lr_for_confidence(0.65, cfg) == 5.0   # still modest
    assert lr_for_confidence(0.70, cfg) == 20.0  # strong bucket (== lr_max)
    assert lr_for_confidence(0.99, cfg) == 20.0


def test_lr_is_clipped_to_lr_max():
    """
    Scenario: a bucket value above lr_max.
    Why it matters: lr_max is the hard ceiling that stops one frame collapsing the
    map; a mis-set bucket must not be able to exceed it.
    """
    cfg = BrainConfig(lr_max=10.0, lr_buckets=((0.0, 1.0), (0.5, 50.0)))
    assert lr_for_confidence(0.9, cfg) == 10.0


# --- d_i: detection probability ---


def test_detection_probability_is_product_clamped():
    """
    Scenario: a full clear look vs a partial canopy look.
    Why it matters: d_i is what makes coverage principled — a weak look must yield a
    small d_i so 'saw nothing' barely clears the cell.
    """
    assert detection_probability(1.0, 1.0, 0.8) == pytest.approx(0.8)
    assert detection_probability(0.5, 0.4, 0.6) == pytest.approx(0.12)
    # clamps even if inputs are slightly out of range
    assert detection_probability(2.0, 1.0, 1.0) == 1.0


# --- the Gaussian kernel ---


def test_kernel_peaks_at_one_and_decays(cfg):
    """
    Scenario: inspect the raw weight kernel.
    Why it matters: peak == 1 at the center is what lets the boost apply the full LR
    at the detection and only a fraction of the boost to neighbors (never < 1).
    """
    k = gaussian_weight_kernel(sigma_cells=1.0, radius_cells=2)
    assert k.shape == (5, 5)
    assert k[2, 2] == pytest.approx(1.0)               # center is exactly 1
    assert k[2, 2] > k[2, 3] > k[2, 4]                 # decays with distance
    assert k[2, 3] == pytest.approx(k[3, 2])           # radially symmetric


# --- non-detection clears covered cells and drains mass into p_out ---


def test_nondetection_lowers_covered_and_raises_pout(cfg, grid):
    """
    Scenario: a uniform map, then a clean look over a few cells (no detection).
    Why it matters: the defining behavior of the map — searching and seeing nothing
    must LOWER the covered cells, RAISE p_out (they might have left), and RAISE the
    untouched cells, all while mass stays at 1.
    """
    n = cfg.n_rows * cfg.n_cols
    posterior = np.full((cfg.n_rows, cfg.n_cols), (1.0 - 0.07) / n)
    coverage = np.zeros((cfg.n_rows, cfg.n_cols))
    p_out = 0.07

    covered = [(5, 5), (5, 6), (4, 5)]
    footprint = [CellCoverage(cell=rc, coverage_fraction=1.0, visibility_weight=1.0) for rc in covered]
    p_before_covered = posterior[5, 5]
    p_before_other = posterior[0, 0]

    apply_coverage_and_nondetection(posterior, coverage, footprint, recall=0.8, exclude_cells=set(), grid=grid)
    posterior, p_out = renormalize(posterior, p_out)

    assert posterior[5, 5] < p_before_covered          # covered cell dropped
    assert posterior[0, 0] > p_before_other            # untouched cell rose
    assert p_out > 0.07                                 # 'they left' became likelier
    assert posterior.sum() + p_out == pytest.approx(1.0)   # mass conserved
    # coverage accrued exactly d_i on the first look (1*1*0.8)
    assert coverage[5, 5] == pytest.approx(0.8)


def test_coverage_compounds_over_repeated_looks(cfg, grid):
    """
    Scenario: the same cell looked at twice with d_i = 0.5 each.
    Why it matters: cleared_i = 1 - Prod(1 - d_i) — two 0.5 looks give 0.75, not 1.0;
    repeated weak looks must not falsely report a cell as fully cleared.
    """
    posterior = np.full((cfg.n_rows, cfg.n_cols), 0.01)
    coverage = np.zeros((cfg.n_rows, cfg.n_cols))
    fp = [CellCoverage(cell=(5, 5), coverage_fraction=1.0, visibility_weight=1.0)]
    # recall 0.5 -> d_i = 0.5 per look
    apply_coverage_and_nondetection(posterior, coverage, fp, recall=0.5, exclude_cells=set(), grid=grid)
    apply_coverage_and_nondetection(posterior, coverage, fp, recall=0.5, exclude_cells=set(), grid=grid)
    assert coverage[5, 5] == pytest.approx(0.75)


def test_detection_cell_accrues_coverage_but_is_not_cleared(cfg, grid):
    """
    Scenario: a covered cell that also holds a detection (in exclude_cells).
    Why it matters: a cell with a detection was still *looked at* (coverage rises),
    but 'saw nothing there' is false, so its probability must NOT be cleared.
    """
    posterior = np.full((cfg.n_rows, cfg.n_cols), 0.01)
    coverage = np.zeros((cfg.n_rows, cfg.n_cols))
    fp = [CellCoverage(cell=(5, 5), coverage_fraction=1.0, visibility_weight=1.0)]
    before = posterior[5, 5]
    apply_coverage_and_nondetection(posterior, coverage, fp, recall=0.8, exclude_cells={(5, 5)}, grid=grid)
    assert posterior[5, 5] == before          # not cleared
    assert coverage[5, 5] == pytest.approx(0.8)  # but coverage still accrued


# --- detection boost sharpens the peak without lowering neighbors ---


def test_detection_boost_sharpens_peak_and_never_lowers(cfg, grid):
    """
    Scenario: a uniform map, then one detection boost at (5,5).
    Why it matters: the center must rise the most, neighbors rise less, far cells are
    untouched pre-renorm, and crucially NO cell is lowered by the boost itself (the
    bug the 1 + (LR-1)*w form avoids).
    """
    posterior = np.full((cfg.n_rows, cfg.n_cols), 0.01)
    before = posterior.copy()
    apply_detection_boost(posterior, center_cell=(5, 5), effective_lr=20.0, grid=grid, cfg=cfg)

    assert posterior[5, 5] == pytest.approx(before[5, 5] * 20.0)   # center gets full LR
    assert posterior[5, 6] > before[5, 6]                          # neighbor rose
    assert posterior[5, 6] < posterior[5, 5]                       # but less than center
    assert np.all(posterior >= before - 1e-12)                     # nothing was lowered
    # a cell outside the kernel radius is untouched
    assert posterior[0, 0] == pytest.approx(before[0, 0])


def test_boost_with_unit_lr_is_a_noop(cfg, grid):
    """
    Scenario: effective_lr == 1.0 (e.g. the blob's evidence was already applied).
    Why it matters: the trigger passes lr==1 to mean 'no new evidence' — it must not
    perturb the map at all (this is how we avoid double-counting correlated frames).
    """
    posterior = np.full((cfg.n_rows, cfg.n_cols), 0.01)
    before = posterior.copy()
    apply_detection_boost(posterior, center_cell=(5, 5), effective_lr=1.0, grid=grid, cfg=cfg)
    assert np.array_equal(posterior, before)


def test_full_frame_conserves_mass(cfg, grid):
    """
    Scenario: a frame with both a non-detection footprint and a detection boost.
    Why it matters: the whole-frame invariant posterior.sum() + p_out == 1 must hold
    after the single renormalization the brain performs each step.
    """
    n = cfg.n_rows * cfg.n_cols
    posterior = np.full((cfg.n_rows, cfg.n_cols), (1.0 - 0.07) / n)
    coverage = np.zeros((cfg.n_rows, cfg.n_cols))
    p_out = 0.07
    fp = [CellCoverage(cell=(3, 3), coverage_fraction=0.8, visibility_weight=0.5)]
    apply_coverage_and_nondetection(posterior, coverage, fp, recall=0.6, exclude_cells={(5, 5)}, grid=grid)
    apply_detection_boost(posterior, center_cell=(5, 5), effective_lr=5.0, grid=grid, cfg=cfg)
    posterior, p_out = renormalize(posterior, p_out)
    assert posterior.sum() + p_out == pytest.approx(1.0)
    assert 0.0 < p_out < 1.0


def test_renormalize_rejects_zero_mass(cfg):
    """
    Scenario: a degenerate all-zero posterior with no p_out.
    Why it matters: dividing by zero would silently produce NaNs that corrupt every
    downstream consumer; we fail loudly instead.
    """
    posterior = np.zeros((cfg.n_rows, cfg.n_cols))
    with pytest.raises(ValueError):
        renormalize(posterior, p_out=0.0)
