# =============================================================================
# terrain.py
# -----------------------------------------------------------------------------
# Responsible for: The terrain seam the prior reads — accessibility (A) and
#                  corridor (C) layers over the grid — plus a SYNTHETIC stub that
#                  encodes the demo's Steep-Ravine story so the brain runs today
#                  with no geo dependencies.
# Role in project: prior_model.md's A_i and C_i come from terrain. This module is
#                  the single place geography enters the prior. The real raster
#                  loader (DEM slope -> A, WorldCover -> A, OSM trails/drainages
#                  -> C) drops in later as another TerrainProvider WITHOUT touching
#                  prior.py — that is the whole point of the seam.
# Assumptions: A_i in [epsilon, 1] with impassable (cliff/deep water) -> 0;
#              C_i in [1, k] (attraction, never < 1). Both are (n_rows, n_cols).
# Pattern: TerrainProvider is a Protocol (structural typing) used as a Strategy —
#          any object with these methods is a valid terrain backend, so the prior
#          depends on the interface, not on the synthetic implementation.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np

from src.common.config import BrainConfig
from src.common.grid import GridSpec

# --- Synthetic-stub shaping constants -------------------------------------------------
# These describe the FAKE terrain only; they vanish when real rasters replace the
# stub, so they live here (not in BrainConfig, which holds true brain tunables).
# All positions are expressed as fractions of the grid so the stub scales to any
# GridSpec (the 160x160 demo grid or a tiny test grid).
_DRAINAGE_WIDTH_CELLS = 1.6     # perpendicular falloff of the corridor boost
_RIDGE_HALF_WIDTH_CELLS = 2.5   # how wide the low-accessibility ridge band is
_RIDGE_CORE_CELLS = 0.8         # cells this close to the ridge crest are impassable (A=0)


@dataclass(frozen=True)
class TerrainLayers:
    """
    The per-cell terrain layers the prior consumes.

    Args (fields):
        accessibility: (n_rows, n_cols) A_i in [0, 1]; 0 = impassable.
        corridor: (n_rows, n_cols) C_i in [1, k]; 1 = no corridor pull.

    Why:
        Bundles exactly the two arrays prior_model.md's product needs, so the
        TerrainProvider contract is a single return value and the prior never has
        to know how they were produced.
    """

    accessibility: np.ndarray
    corridor: np.ndarray


class TerrainProvider(Protocol):
    """
    The terrain backend interface (Strategy pattern): produce A/C layers for a grid
    and describe a cell's terrain context.

    Why:
        Lets a synthetic stub now and a raster-backed loader later be interchangeable
        from the prior's perspective — the swappable-backend invariant, applied to
        geography just like the detector.
    """

    def layers(self, grid: GridSpec) -> TerrainLayers:
        """Return the accessibility and corridor layers over the grid."""
        ...

    def context(self, grid: GridSpec, cell: Tuple[int, int]) -> Dict[str, Any]:
        """Return a small land-cover/slope/near-feature dict for a cell (LocatedEvent)."""
        ...


def _distance_to_polyline_cells(
    n_rows: int, n_cols: int, polyline: List[Tuple[float, float]]
) -> np.ndarray:
    """
    Vectorized perpendicular distance (in cell units) from every cell to a polyline.

    Args:
        n_rows: Grid rows.
        n_cols: Grid cols.
        polyline: Ordered [(row, col), ...] waypoints in (fractional) cell coords.

    Returns:
        (n_rows, n_cols) float array: each cell's distance to the nearest segment.

    Why:
        Both the drainage corridor and the ridge band are line features; their
        influence falls off with perpendicular distance. This is the standard
        point-to-segment distance (clamp the projection parameter t to [0, 1] so we
        measure to the segment, not the infinite line), broadcast over the grid.
    """
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rr = rr.astype(float)
    cc = cc.astype(float)
    best = np.full((n_rows, n_cols), np.inf)
    for (r1, c1), (r2, c2) in zip(polyline[:-1], polyline[1:]):
        vr, vc = r2 - r1, c2 - c1                 # segment direction vector
        seg_len_sq = vr * vr + vc * vc
        if seg_len_sq == 0.0:
            dist = np.hypot(rr - r1, cc - c1)     # degenerate segment = a point
        else:
            # t = projection of (P - p1) onto the segment, clamped to the segment.
            t = ((rr - r1) * vr + (cc - c1) * vc) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj_r = r1 + t * vr
            proj_c = c1 + t * vc
            dist = np.hypot(rr - proj_r, cc - proj_c)
        best = np.minimum(best, dist)
    return best


class SyntheticTerrain:
    """
    A hand-shaped terrain stub matching docs/demo_scenario.md: a drainage corridor
    running downhill (SW) from the LKP toward the coast, and a low-accessibility
    ridge band north of the LKP with an impassable crest.

    Why:
        Lets the prior bloom around the LKP, elongate SW down the canyon, and be
        suppressed on the ridge — the explainable starting hypothesis the judge sees
        — without ingesting any rasters. Implements TerrainProvider, so the real
        loader is a drop-in replacement.
    """

    def __init__(self, cfg: BrainConfig) -> None:
        """
        Args:
            cfg: Brain config; supplies corridor_k, accessibility_epsilon, and the
                LKP used to anchor the synthetic features.
        Why:
            The stub respects the same tunables (k, epsilon) the real terrain would,
            so swapping backends doesn't change the prior's calibration.
        """
        self._cfg = cfg

    def _feature_polylines(self, grid: GridSpec) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Build the drainage and ridge polylines in cell coords, anchored at the LKP.

        Returns:
            (drainage, ridge): two lists of (row, col) waypoints.

        Why:
            Geometry is derived from the LKP cell and grid size (not hard cell
            numbers) so the same stub works on the 160x160 demo grid and tiny test
            grids alike.
        """
        lkp_r, lkp_c = grid.latlon_to_cell(*self._cfg.lkp_latlon)
        nr, nc = grid.n_rows, grid.n_cols
        # Drainage: from the LKP heading SW (down + west), with a slight bend, ending
        # toward the SW quadrant (the coast). SW = lower row (south) and lower col (west).
        drainage = [
            (float(lkp_r), float(lkp_c)),
            (lkp_r - 0.16 * nr, lkp_c - 0.12 * nc),
            (lkp_r - 0.34 * nr, lkp_c - 0.30 * nc),
        ]
        # Ridge: an E-W band just north of the LKP (the canyon's upper wall/ridgeline).
        ridge_row = lkp_r + 0.12 * nr
        ridge = [
            (ridge_row, 0.10 * nc),
            (ridge_row + 0.03 * nr, 0.55 * nc),
            (ridge_row, 0.92 * nc),
        ]
        return drainage, ridge

    def layers(self, grid: GridSpec) -> TerrainLayers:
        """
        Produce the accessibility and corridor arrays for the grid.

        Args:
            grid: The shared GridSpec.

        Returns:
            TerrainLayers with A in [epsilon..1] (0 on the ridge crest) and C in
            [1..k] (peaking along the drainage).

        Why:
            A and C are the prior_model.md factors. Corridor uses a Gaussian boost
            falling to 1 within ~1-2 cells of the drainage; accessibility dips toward
            epsilon across the ridge band and hard-zeroes the impassable crest.
        """
        nr, nc = grid.n_rows, grid.n_cols
        k = self._cfg.corridor_k
        eps = self._cfg.accessibility_epsilon
        drainage, ridge = self._feature_polylines(grid)

        # --- Corridor C in [1, k]: Gaussian boost around the drainage line ---
        d_drain = _distance_to_polyline_cells(nr, nc, drainage)
        corridor = 1.0 + (k - 1.0) * np.exp(-(d_drain / _DRAINAGE_WIDTH_CELLS) ** 2)

        # --- Accessibility A in [eps, 1]: dip toward eps across the ridge band ---
        d_ridge = _distance_to_polyline_cells(nr, nc, ridge)
        # A smooth band: 0 far from the ridge -> 1 at the crest (how "ridge-like" a cell is).
        ridge_strength = np.exp(-(d_ridge / _RIDGE_HALF_WIDTH_CELLS) ** 2)
        accessibility = 1.0 - (1.0 - eps) * ridge_strength  # crest ~eps, away ~1
        # The crest core is an impassable cliff: hard zero (non-compensatory in the prior).
        accessibility[d_ridge <= _RIDGE_CORE_CELLS] = 0.0

        return TerrainLayers(accessibility=accessibility, corridor=corridor)

    def context(self, grid: GridSpec, cell: Tuple[int, int]) -> Dict[str, Any]:
        """
        Describe a cell's terrain for the LocatedEvent's terrain_context.

        Args:
            grid: The shared GridSpec.
            cell: (row, col) to describe.

        Returns:
            A dict with a coarse land_cover guess, a near_drainage flag, and the
            cell-center lat/lon — enough for the spoken message and ground routing.

        Why:
            LocatedEvent carries terrain_context so the voice layer never reaches into
            the brain. The stub derives it from the same synthetic features; the real
            provider would read it from the land-cover raster.
        """
        layers = self.layers(grid)
        r, c = cell
        near_drainage = bool(layers.corridor[r, c] > 1.0 + 0.4 * (self._cfg.corridor_k - 1.0))
        access = float(layers.accessibility[r, c])
        # Coarse, illustrative land-cover label from the synthetic accessibility.
        if access <= self._cfg.accessibility_epsilon + 1e-6:
            land_cover = "steep/cliff"
        elif near_drainage:
            land_cover = "forested canyon (drainage)"
        else:
            land_cover = "mixed scrub/forest"
        lat, lon = grid.cell_to_latlon(r, c)
        return {
            "land_cover": land_cover,
            "near_drainage": near_drainage,
            "accessibility": round(access, 3),
            "latlon": (round(lat, 5), round(lon, 5)),
        }
