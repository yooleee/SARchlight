# =============================================================================
# update.py
# -----------------------------------------------------------------------------
# Responsible for: The Bayesian per-cell update math (interfaces.md §5) — the
#                  detection probability d_i, the non-detection clearing, the
#                  detection boost (clipped/bucketed LR + Gaussian kernel), the
#                  accumulated-coverage compounding, and the renormalization over
#                  all cells AND the reserved out-of-region mass p_out.
# Role in project: The brain's core. brain.py orchestrates these primitives once
#                  per Observation; trigger.py decides the *effective* LR per blob
#                  (the cap that stops correlated frames from over-concentrating).
# Assumptions: posterior is a float ndarray (n_rows, n_cols); p_out is a scalar;
#              the invariant posterior.sum() + p_out == 1 holds after renormalize.
#              Functions MUTATE the passed arrays in place (the brain is the single
#              writer) and also return them for convenient chaining.
# =============================================================================

from __future__ import annotations

import math
from typing import Iterable, Set, Tuple

import numpy as np

from src.common.config import BrainConfig
from src.common.contracts import Cell, CellCoverage
from src.common.grid import GridSpec


# --- Likelihood ratio LR(c): bucketed and clipped (interfaces §5.3) ---


def lr_for_confidence(confidence: float, cfg: BrainConfig) -> float:
    """
    Map a raw detector confidence to a clipped, bucketed likelihood ratio LR(c).

    Args:
        confidence: Raw detector score in [0, 1] (evidence strength, NOT a
            probability of presence).
        cfg: Config holding the (threshold, LR) buckets and lr_max.

    Returns:
        The LR for that confidence, clipped to [1, lr_max].

    Why:
        Raw detector confidence is uncalibrated (typically overconfident), so a
        step function — ignore below threshold -> modest LR -> lr_max — is more
        honest than a smooth curve implying continuous calibration. Clipping to
        lr_max means one frame can never collapse the map onto a false positive.
    """
    lr = 1.0
    # buckets are ascending by threshold; the highest one whose threshold is met wins.
    for threshold, bucket_lr in cfg.lr_buckets:
        if confidence >= threshold:
            lr = bucket_lr
    return float(min(max(lr, 1.0), cfg.lr_max))


# --- Detection probability d_i = coverage * visibility * recall (interfaces §5.1) ---


def detection_probability(coverage_fraction: float, visibility_weight: float, recall: float) -> float:
    """
    The chance we'd have detected a present person in a cell on this look.

    Args:
        coverage_fraction: Fraction of the cell inside the frame footprint, [0, 1].
        visibility_weight: How observable a person is here this look, [0, 1].
        recall: Detector true-positive rate for this sensor/altitude, [0, 1].

    Returns:
        d_i in [0, 1], clamped.

    Why:
        d_i is the single knob that makes coverage principled: a partial or canopy-
        occluded look gives a small d_i (seeing nothing barely clears the cell); a
        full clear look gives a large d_i (seeing nothing clears it strongly).
    """
    d = coverage_fraction * visibility_weight * recall
    return float(min(max(d, 0.0), 1.0))


# --- Detection kernel: a peak-1 Gaussian over nearby cells (interfaces §5.3) ---


def gaussian_weight_kernel(sigma_cells: float, radius_cells: int) -> np.ndarray:
    """
    A square Gaussian weight kernel, value 1.0 at the center, decaying outward.

    Args:
        sigma_cells: Gaussian standard deviation, in cells.
        radius_cells: Half-width; the kernel is (2*radius+1) square, truncated here.

    Returns:
        A (2*radius+1, 2*radius+1) float array; center == 1.0, monotic decay.

    Why:
        At altitude a few pixels of box-center error is tens of meters on the
        ground, so a single-cell update is brittle. Peaking at 1 (not summing to 1)
        is deliberate: it lets us apply the *full* LR at the center and a fraction
        of the boost to neighbors, never suppressing them (see apply_detection_boost).
    """
    size = 2 * radius_cells + 1
    kernel = np.zeros((size, size), dtype=float)
    two_sigma_sq = 2.0 * sigma_cells * sigma_cells
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            # distance-squared in cell units; center (0,0) -> exp(0) = 1.0
            kernel[dr + radius_cells, dc + radius_cells] = math.exp(-(dr * dr + dc * dc) / two_sigma_sq)
    return kernel


# --- The two update halves (no renormalization here — the brain renorms once) ---


def apply_coverage_and_nondetection(
    posterior: np.ndarray,
    coverage: np.ndarray,
    footprint: Iterable[CellCoverage],
    recall: float,
    exclude_cells: Set[Cell],
    grid: GridSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the non-detection clearing and accumulate coverage over a footprint.

    Args:
        posterior: (n_rows, n_cols) current p_i; mutated in place.
        coverage: (n_rows, n_cols) accumulated cleared_i; mutated in place.
        footprint: The frame's covered cells (each with coverage_fraction and
            visibility_weight).
        recall: The detector recall for this frame's sensor/altitude.
        exclude_cells: Cells that had a detection — they accrue coverage but are
            NOT cleared (a detection landed there, so "nothing seen" is false).
        grid: GridSpec, for bounds checking.

    Returns:
        (posterior, coverage), both mutated and returned for chaining. NOT yet
        renormalized — the brain renormalizes once after detection boosts too.

    Why:
        Non-detection: p_i <- p_i*(1 - d_i) for each covered cell with no detection
        (interfaces §5.2). Coverage compounds independently as
        cleared_i = 1 - Prod_k (1 - d_i^k), so a cell looked at many times with good
        d_i approaches fully cleared while one weak glance does not.
    """
    for cc in footprint:
        r, c = cc.cell
        if not grid.in_bounds(r, c):
            continue
        d_i = detection_probability(cc.coverage_fraction, cc.visibility_weight, recall)
        # Coverage ("where have we looked") accrues on every look, detection or not.
        # TODO(post-integration): cap cumulative clearance per (cell, sensor). This
        # product assumes independent looks, but repeated passes from the same angle/
        # sensor miss identically under canopy, so it OVERSTATES clearance (interfaces
        # §5.5). Tune against real coverage patterns once geo/detector data flows — see
        # docs/brain_followups.md. (Seam left here; behavior intentionally simple now.)
        coverage[r, c] = 1.0 - (1.0 - coverage[r, c]) * (1.0 - d_i)
        # Probability is cleared only where nothing was detected.
        if (r, c) not in exclude_cells:
            posterior[r, c] *= (1.0 - d_i)
    return posterior, coverage


def apply_detection_boost(
    posterior: np.ndarray,
    center_cell: Cell,
    effective_lr: float,
    grid: GridSpec,
    cfg: BrainConfig,
) -> np.ndarray:
    """
    Boost a small neighborhood around a detection by an effective likelihood ratio.

    Args:
        posterior: (n_rows, n_cols) current p_i; mutated in place.
        center_cell: (row, col) where the detection projected.
        effective_lr: The LR to apply at the center (>= 1). The trigger supplies
            this — capped per blob so correlated frames don't re-multiply (§5.4).
        grid: GridSpec, for bounds checking.
        cfg: Config holding kernel sigma/radius.

    Returns:
        posterior, mutated and returned. NOT renormalized here.

    Why:
        Realizes interfaces §5.3's "p_k <- p_k * LR(c) * kernel(k,i)" with the factor
        1 + (LR-1)*w(k): the center gets the full LR, neighbors get a fraction of the
        boost, and the factor is never < 1 — so a detection never *lowers* a nearby
        cell (the latent bug in the literal LR*kernel form). Clip+renorm (done by the
        brain) then concentrates mass without zeroing the rest, so a lone false
        positive stays recoverable by later non-detections.
    """
    if effective_lr <= 1.0:
        # Nothing to add (e.g. the blob's LR was already applied — see trigger.py).
        return posterior
    rad = cfg.kernel_radius_cells
    kernel = gaussian_weight_kernel(cfg.kernel_sigma_cells, rad)  # peak 1.0 at center
    r0, c0 = center_cell
    for dr in range(-rad, rad + 1):
        for dc in range(-rad, rad + 1):
            r, c = r0 + dr, c0 + dc
            if not grid.in_bounds(r, c):
                continue
            w = kernel[dr + rad, dc + rad]  # in [0, 1], 1.0 at the center
            posterior[r, c] *= 1.0 + (effective_lr - 1.0) * w
    return posterior


# --- Renormalization over all cells AND p_out (interfaces §5) ---


def renormalize(posterior: np.ndarray, p_out: float) -> Tuple[np.ndarray, float]:
    """
    Rescale so that posterior.sum() + p_out == 1, treating p_out as a fixed bin.

    Args:
        posterior: (n_rows, n_cols) unnormalized p_i; mutated in place (divided).
        p_out: Unnormalized reserved out-of-region mass.

    Returns:
        (posterior, p_out) both rescaled so their total is exactly 1.

    Why:
        p_out is an *unsearched* bin (d=0): non-detections shrink covered cells but
        leave p_out, so after this renorm p_out's share RISES — the map can finally
        say "they're probably not here" instead of endlessly reshuffling mass
        between cells. Detections grow cells, so p_out's share falls. Including
        p_out in the denominator is what makes both directions correct.
    """
    total = float(posterior.sum()) + p_out
    if total <= 0.0:
        raise ValueError("Cannot renormalize: total probability mass is non-positive.")
    posterior /= total
    return posterior, p_out / total
