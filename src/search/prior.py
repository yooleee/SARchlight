# =============================================================================
# prior.py
# -----------------------------------------------------------------------------
# Responsible for: Building the initial probability map (the prior the judge sees
#                  at t0) per docs/prior_model.md: prior_i ∝ D(dist_i)·A_i·C_i,
#                  normalized alongside the reserved out-of-region mass p_out.
# Role in project: Produces p_i^(0) for the loop. interfaces.md §5's update then
#                  applies detection/non-detection evidence on top of this.
# Assumptions: Distance is Euclidean from the LKP in local meters (the baseline;
#              Tobler cost-distance is the upgrade behind dist_i). A and C come
#              from a TerrainProvider. Mass invariant: Σ p_i + p_out == 1.
# Modeling choice: the combination is MULTIPLICATIVE (non-compensatory) — any
#              near-zero factor (a cliff) kills the cell regardless of proximity.
# =============================================================================

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.common.config import BrainConfig
from src.common.grid import GridSpec
from src.search.terrain import TerrainProvider


def half_normal_decay(dist_m: np.ndarray, sigma_m: float) -> np.ndarray:
    """
    The distance-decay factor D(dist) as a 2-D half-normal.

    Args:
        dist_m: Array of distances from the LKP, in meters.
        sigma_m: Half-normal sigma in meters (set from Koester quantiles).

    Returns:
        D in (0, 1]: 1.0 at the LKP, decaying smoothly outward.

    Why:
        Koester's lost-person data gives distance-from-LKP quantiles; a half-normal
        whose 50% radius ≈ 1.177·σ reproduces them (σ ≈ 2.6 km for a Hiker). It does
        not hard-zero the far field, so the ~5% of subjects beyond the 95% radius are
        still represented (p_out and this non-zero tail cover them).
    """
    # exp(-dist^2 / (2 sigma^2)); operate on dist^2 to avoid a needless sqrt upstream.
    return np.exp(-(dist_m ** 2) / (2.0 * sigma_m * sigma_m))


def distance_from_lkp_m(grid: GridSpec, lkp_latlon: Tuple[float, float]) -> np.ndarray:
    """
    Euclidean distance (meters) from the LKP to every cell center.

    Args:
        grid: The shared GridSpec.
        lkp_latlon: (lat, lon) of the last-known position.

    Returns:
        (n_rows, n_cols) array of distances in meters.

    Why:
        The prior's D(dist) needs a per-cell distance. Working in the grid's local
        meter frame keeps it one vectorized expression; cost-distance later swaps
        only this function, leaving D/A/C/normalization untouched (same seam).
    """
    lkp_east_m, lkp_north_m = grid.latlon_to_local_m(*lkp_latlon)
    rows = np.arange(grid.n_rows)
    cols = np.arange(grid.n_cols)
    # cell-center coordinates in local meters (the +0.5 centers the cell).
    north_centers = (rows + 0.5) * grid.cell_size_m      # shape (n_rows,)
    east_centers = (cols + 0.5) * grid.cell_size_m       # shape (n_cols,)
    # broadcast to (n_rows, n_cols): dn varies by row, de varies by col.
    dn = north_centers[:, None] - lkp_north_m
    de = east_centers[None, :] - lkp_east_m
    return np.sqrt(dn * dn + de * de)


def combine(layers: List[np.ndarray], mode: str = "product") -> np.ndarray:
    """
    Combine prior layers into one unnormalized field.

    Args:
        layers: List of (n_rows, n_cols) arrays (e.g. [D, A, C]).
        mode: "product" (the only supported mode for now).

    Returns:
        The combined (n_rows, n_cols) array.

    Why:
        A single combine() seam means individual layers could switch to a
        compensatory (weighted-sum) rule later without a rewrite. Product is the
        load-bearing choice now: terrain is non-compensatory, so any ~0 factor must
        zero the cell — proximity cannot "buy back" a cliff. (build seams, not futures.)
    """
    if mode != "product":
        raise ValueError(f"Unsupported combine mode: {mode!r} (only 'product' for now).")
    result = np.ones_like(layers[0], dtype=float)
    for layer in layers:
        result = result * layer
    return result


def build_prior(
    grid: GridSpec, cfg: BrainConfig, terrain: TerrainProvider
) -> Tuple[np.ndarray, float]:
    """
    Build the normalized prior probability map and its reserved out-of-region mass.

    Args:
        grid: The shared GridSpec.
        cfg: Brain config (sigma, p_out, LKP, ...).
        terrain: A TerrainProvider supplying the A and C layers.

    Returns:
        (posterior, p_out): posterior is (n_rows, n_cols) with Σ posterior == 1 - p_out,
        so posterior.sum() + p_out == 1 — the invariant the loop maintains.

    Why:
        prior_i ∝ D·A·C, then P_i = (1 - p_out)·prior_i / Σ prior_j, with p_out held
        separately. This is the POC/containment prior: where-people-go (D) ×
        where-terrain-lets-them (A) × where-corridors-pull-them (C), the explainable
        starting hypothesis the demo opens on.
    """
    dist_m = distance_from_lkp_m(grid, cfg.lkp_latlon)
    decay = half_normal_decay(dist_m, cfg.prior_sigma_m)         # D
    terrain_layers = terrain.layers(grid)
    raw = combine([decay, terrain_layers.accessibility, terrain_layers.corridor], mode="product")

    total = float(raw.sum())
    if total <= 0.0:
        raise ValueError("Prior is degenerate: all cells zeroed (check terrain/σ).")
    # Distribute the in-region mass (1 - p_out) across cells; p_out held separately.
    posterior = (1.0 - cfg.p_out) * raw / total
    return posterior, cfg.p_out


def rank_top_cells(posterior: np.ndarray, k: int = 5) -> List[Tuple[Tuple[int, int], float]]:
    """
    Return the k highest-probability cells as [((row, col), p_i), ...].

    Args:
        posterior: (n_rows, n_cols) probability array.
        k: How many top cells to return.

    Returns:
        Ranked list, highest p_i first.

    Why:
        MapState.top_cells (the search targets) is just the argmax-k of the
        posterior; keeping it here lets both the prior preview and the loop reuse one
        implementation. (DRY.)
    """
    flat = posterior.ravel()
    k = min(k, flat.size)
    # argpartition for the top-k, then sort just those k descending (cheap).
    top_idx = np.argpartition(flat, -k)[-k:]
    top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]
    n_cols = posterior.shape[1]
    return [((int(i // n_cols), int(i % n_cols)), float(flat[i])) for i in top_idx]
