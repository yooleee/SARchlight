# =============================================================================
# planner.py
# -----------------------------------------------------------------------------
# Responsible for: The SEARCH DIRECTOR (C1) — a coarse sector overlay on the fine
#                  probability grid, per-sector probability-of-area (POA) ranking,
#                  single-drone sweep planning, and multi-drone no-overlap sector
#                  assignment. PURE and READ-ONLY: it consumes posterior + coverage
#                  arrays and returns a plan; it never writes MapState.
# Role in project: docs/brain_followups.md C1. An ADDITIVE layer that sits on the
#                  search-director seam (it generalizes SearchBrain._next_target's
#                  one-line argmax into a sector-level planner). It does NOT touch the
#                  Bayesian update, the located trigger, or the single-writer MapState.
# Invariants it must preserve: sectors are a TASKING overlay only — the belief stays
#                  50 m-fine, never coarsened; the brain remains the single writer.
# =============================================================================

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.common.config import BrainConfig
from src.common.grid import GridSpec

# A fine cell and a coarse sector are both (row, col); alias for readability.
Cell = Tuple[int, int]
Sector = Tuple[int, int]


def _ceil_div(a: int, b: int) -> int:
    """Integer ceiling division (number of sectors along an axis, last one ragged)."""
    return -(-a // b)


@dataclass(frozen=True)
class SectorGrid:
    """
    A coarse tasking overlay: square blocks of `sector_size_cells` fine cells.

    Args (fields):
        grid: The shared fine GridSpec (50 m cells).
        sector_size_cells: Edge length of a sector in fine cells (e.g. 10 = 500 m).

    Why:
        SAR sectorization tasks search by named, assignable areas. Making this its own
        object (rather than scattering `// sector_size` math) keeps the cell<->sector
        mapping, the per-sector bounds, and the human names in one place — and keeps the
        belief grid untouched (this is a separate, coarser index used only for tasking).
    """

    grid: GridSpec
    sector_size_cells: int

    @classmethod
    def from_config(cls, grid: GridSpec, cfg: BrainConfig) -> "SectorGrid":
        """Build the sector overlay from config's `sector_size_cells`."""
        return cls(grid=grid, sector_size_cells=cfg.sector_size_cells)

    @property
    def n_sector_rows(self) -> int:
        """Number of sector rows (south->north); the last row may be ragged."""
        return _ceil_div(self.grid.n_rows, self.sector_size_cells)

    @property
    def n_sector_cols(self) -> int:
        """Number of sector columns (west->east); the last column may be ragged."""
        return _ceil_div(self.grid.n_cols, self.sector_size_cells)

    def sector_of(self, cell: Cell) -> Sector:
        """
        The sector (sr, sc) that contains a fine cell.

        Args:
            cell: (row, col) fine-grid cell.

        Returns:
            (sector_row, sector_col).

        Why:
            The inverse of `bounds` — used to label a detection's sector and to find which
            sector a drone currently sits in.
        """
        r, c = cell
        return (r // self.sector_size_cells, c // self.sector_size_cells)

    def bounds(self, sector: Sector) -> Tuple[int, int, int, int]:
        """
        The fine-cell half-open bounds of a sector: (r0, r1, c0, c1).

        Args:
            sector: (sector_row, sector_col).

        Returns:
            (r0, r1, c0, c1) with rows in [r0, r1) and cols in [c0, c1), clipped to the grid
            (so the ragged last row/col stops at the edge).

        Why:
            POA summation, the boustrophedon sweep, and the viz rectangle all need a sector's
            fine extent. Clipping here means callers never index past the grid.
        """
        sr, sc = sector
        sz = self.sector_size_cells
        r0, c0 = sr * sz, sc * sz
        r1, c1 = min(r0 + sz, self.grid.n_rows), min(c0 + sz, self.grid.n_cols)
        return r0, r1, c0, c1

    def center_cell(self, sector: Sector) -> Cell:
        """The fine cell at a sector's center (for travel distance + labels)."""
        r0, r1, c0, c1 = self.bounds(sector)
        return ((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2)

    def name(self, sector: Sector) -> str:
        """
        A human-legible sector name: column letter + row number, e.g. 'C4' (A1 at the SW).

        Args:
            sector: (sector_row, sector_col).

        Returns:
            A short name. Columns map to letters A, B, C... (west->east); rows to 1-based
            numbers (south->north, since row increases north from the SW origin).

        Why:
            Operator legibility is half of C1's point — "send drone 2 to sector C4" beats a
            cell index. Falls back to a numeric column tag beyond 26 columns (won't happen at
            the demo's size, but stays defined).
        """
        sr, sc = sector
        col = string.ascii_uppercase[sc] if sc < 26 else f"[{sc}]"
        return f"{col}{sr + 1}"

    def iter_sectors(self) -> List[Sector]:
        """All (sector_row, sector_col) pairs, row-major."""
        return [(sr, sc) for sr in range(self.n_sector_rows) for sc in range(self.n_sector_cols)]


@dataclass(frozen=True)
class SectorScores:
    """
    Per-sector scores over the coarse grid.

    Args (fields):
        poa: (n_sector_rows, n_sector_cols) probability-of-area = Σ posterior in the sector.
        mean_coverage: (n_sector_rows, n_sector_cols) mean cleared-fraction in the sector.
        priority: (n_sector_rows, n_sector_cols) search priority = poa * (1 - mean_coverage).

    Why:
        POA is the honest "how much probability mass is here" (what an operator reads);
        priority is what we RANK by, so an already-swept high-POA sector drops below an
        unsearched one. Keeping both separate avoids conflating "likely" with "worth flying to
        next."
    """

    poa: np.ndarray
    mean_coverage: np.ndarray
    priority: np.ndarray


def _block_reduce_sum(arr: np.ndarray, sz: int) -> np.ndarray:
    """
    Sum `arr` over sz x sz blocks (the last block per axis may be ragged).

    Args:
        arr: (n_rows, n_cols) array.
        sz: Block edge length in cells.

    Returns:
        (n_sector_rows, n_sector_cols) array of block sums.

    Why:
        np.add.reduceat sums the slices between the given start indices (and the final slice to
        the end), so it block-sums to the coarse grid in two vectorized passes AND handles a
        grid size that isn't a multiple of sz — no padding, no Python loop over 256 sectors.
    """
    row_starts = np.arange(0, arr.shape[0], sz)
    col_starts = np.arange(0, arr.shape[1], sz)
    return np.add.reduceat(np.add.reduceat(arr, row_starts, axis=0), col_starts, axis=1)


def sector_poa(posterior: np.ndarray, coverage: np.ndarray, sectors: SectorGrid) -> SectorScores:
    """
    Aggregate the fine posterior + coverage into per-sector POA and search priority.

    Args:
        posterior: (n_rows, n_cols) probability map (the belief; not modified).
        coverage: (n_rows, n_cols) accumulated cleared-fraction in [0, 1].
        sectors: The coarse SectorGrid overlay.

    Returns:
        A SectorScores with POA, mean coverage, and priority per sector.

    Why:
        Standard SAR "probability of area": a sector's worth-searching score is the probability
        mass it holds, discounted by how much of it we've already cleared. This is the ranking
        signal the single- and multi-drone planners consume — computed on the fine arrays so no
        information is thrown away (the coarse grid is only the aggregation target).
    """
    sz = sectors.sector_size_cells
    poa = _block_reduce_sum(posterior, sz)
    cov_sum = _block_reduce_sum(coverage, sz)
    counts = _block_reduce_sum(np.ones_like(coverage), sz)   # cells per sector (ragged-aware)
    mean_coverage = cov_sum / counts
    priority = poa * (1.0 - mean_coverage)
    return SectorScores(poa=poa, mean_coverage=mean_coverage, priority=priority)


@dataclass(frozen=True)
class RankedSector:
    """One ranked sector: its index, name, POA, and search priority."""

    sector: Sector
    name: str
    poa: float
    priority: float


def rank_sectors(
    scores: SectorScores, sectors: SectorGrid, k: Optional[int] = None
) -> List[RankedSector]:
    """
    Rank sectors by search priority (highest first).

    Args:
        scores: The per-sector SectorScores.
        sectors: The SectorGrid (for names).
        k: If given, return only the top-k; otherwise all sectors.

    Returns:
        A list of RankedSector, highest priority first.

    Why:
        The ordered list IS the search plan's backbone: the single drone takes the top sector,
        and multi-drone assignment walks down this list handing out disjoint sectors. Ranking
        by priority (not raw POA) means a swept sector won't keep being re-picked.
    """
    ranked = [
        RankedSector(
            sector=(sr, sc),
            name=sectors.name((sr, sc)),
            poa=float(scores.poa[sr, sc]),
            priority=float(scores.priority[sr, sc]),
        )
        for sr in range(sectors.n_sector_rows)
        for sc in range(sectors.n_sector_cols)
    ]
    ranked.sort(key=lambda rs: rs.priority, reverse=True)
    return ranked if k is None else ranked[:k]


# =============================================================================
# Single-drone sweep planning: pick the top sector, lay a boustrophedon over it.
# =============================================================================

def _sweep_passes(lo: int, hi: int, swath_spacing: int) -> List[int]:
    """
    Evenly-spaced sweep pass positions across [lo, hi) that INCLUDE both edges.

    Args:
        lo, hi: Half-open extent (rows or cols) of the sector.
        swath_spacing: Target cells between adjacent passes.

    Returns:
        A sorted list of pass indices from lo to hi-1, spaced <= swath_spacing.

    Why:
        Centering passes (lo + half, lo + half + sz, ...) leaves the sector EDGES under the
        footprint rim, so a subject on a sector boundary is only grazed by one marginal frame —
        which is exactly how a real find can slip through (the boundary is an artifact of where
        we drew the sectors, not the terrain). Spreading the passes edge-to-edge (and using one
        extra pass so spacing stays <= the footprint width) covers the whole sector, boundary
        cells included, so the subject is reliably seen wherever it sits.
    """
    length = hi - lo
    if length <= swath_spacing:
        return [(lo + hi - 1) // 2]                      # one central pass for a small/ragged sector
    n = math.ceil(length / swath_spacing) + 1            # +1 -> spacing <= swath and both edges hit
    return [lo + round(i * (hi - 1 - lo) / (n - 1)) for i in range(n)]


def boustrophedon_waypoints(
    bounds: Tuple[int, int, int, int], swath_spacing: int, entry_corner: Optional[Cell] = None
) -> List[Cell]:
    """
    A lawnmower (boustrophedon) path of sweep waypoints covering a sector's cells.

    Args:
        bounds: (r0, r1, c0, c1) half-open fine-cell bounds of the sector.
        swath_spacing: Cells between adjacent passes/waypoints (~the footprint width).
        entry_corner: The cell the drone is coming from; the path starts at the nearest
            corner so the drone doesn't fly across the sector to begin. None -> SW-ish start.

    Returns:
        Ordered [(row, col), ...] waypoints, alternating pass direction (the classic
        back-and-forth that covers an area with no wasted turns).

    Why:
        Boustrophedon (ox-turning, "as the ox plows") is the standard area-coverage pattern.
        Passes are spread edge-to-edge (`_sweep_passes`) so a footprint of ~swath_spacing tiles
        the sector INCLUDING its boundaries — a subject on a sector edge is still covered.
        Ordering by the entry corner keeps the path continuous when chaining sector to sector.
    """
    r0, r1, c0, c1 = bounds
    rows = _sweep_passes(r0, r1, swath_spacing)
    cols = _sweep_passes(c0, c1, swath_spacing)

    # Start from the corner nearest the drone, so we don't traverse the sector to begin.
    start_left = True
    if entry_corner is not None:
        er, ec = entry_corner
        if abs(er - rows[-1]) < abs(er - rows[0]):
            rows = rows[::-1]                       # enter from the near row band
        start_left = abs(ec - cols[0]) <= abs(ec - cols[-1])

    waypoints: List[Cell] = []
    left_to_right = start_left
    for r in rows:
        line = cols if left_to_right else cols[::-1]
        waypoints.extend((r, c) for c in line)
        left_to_right = not left_to_right           # serpentine: flip each pass
    return waypoints


@dataclass(frozen=True)
class SearchPlan:
    """
    A single drone's recommended next move: which sector and the sweep over it.

    Args (fields):
        target_sector: The (sector_row, sector_col) to search next.
        target_name: Its human name (e.g. 'C4').
        waypoints: The ordered boustrophedon sweep cells over the sector.
        next_target: The immediate cell to head to (the first waypoint).

    Why:
        Bundles exactly what a consumer needs to act: a named sector (legibility) and the
        concrete sweep path (`MapState.search_path`). `next_target` is the first waypoint so
        the dashboard's "go here next" arrow and the flown path agree.
    """

    target_sector: Sector
    target_name: str
    waypoints: List[Cell]
    next_target: Cell


class SectorPlanner:
    """
    The search director (C1): ranks sectors and plans sweeps, single- or multi-drone.

    Why:
        Holds the SectorGrid + tunables so callers (the brain's MapState publication, and the
        closed-loop demo) share one director. Pure and read-only — every method takes the
        posterior/coverage arrays and returns a plan; none mutates state, so the brain stays
        the single writer of MapState.
    """

    def __init__(self, grid: GridSpec, cfg: BrainConfig) -> None:
        """
        Args:
            grid: The shared fine GridSpec.
            cfg: Tunables (sector size, swath spacing).
        """
        self.sectors = SectorGrid.from_config(grid, cfg)
        self._cfg = cfg

    def rank(self, posterior: np.ndarray, coverage: np.ndarray, k: Optional[int] = None) -> List[RankedSector]:
        """Rank sectors by search priority (POA discounted by coverage)."""
        return rank_sectors(sector_poa(posterior, coverage, self.sectors), self.sectors, k)

    def plan_single(
        self,
        posterior: np.ndarray,
        coverage: np.ndarray,
        drone_pos: Optional[Cell] = None,
        exclude: Optional[set] = None,
        min_priority: float = 0.0,
    ) -> Optional[SearchPlan]:
        """
        Plan one drone's next sweep: the top-priority sector and its boustrophedon path.

        Args:
            posterior: The fine belief (read only).
            coverage: The fine cleared-fraction layer (read only).
            drone_pos: Where the drone is now (orders the sweep's entry); optional.
            exclude: Sectors already taken (multi-drone) or exhausted — skipped.
            min_priority: Skip sectors at/below this priority (a swept-out map).

        Returns:
            A SearchPlan for the chosen sector, or None if no sector is worth searching.

        Why:
            This generalizes the brain's old one-line `argmax(posterior * (1 - coverage))` into
            "go to the most-worth-searching SECTOR and sweep it" — the sector view an operator
            reasons about and the unit multi-drone assignment hands out.
        """
        exclude = exclude or set()
        for rs in self.rank(posterior, coverage):
            if rs.sector in exclude or rs.priority <= min_priority:
                continue
            waypoints = boustrophedon_waypoints(
                self.sectors.bounds(rs.sector), self._cfg.swath_spacing_cells, drone_pos
            )
            return SearchPlan(rs.sector, rs.name, waypoints, waypoints[0])
        return None

    def assign_multi(
        self,
        posterior: np.ndarray,
        coverage: np.ndarray,
        drone_positions: List[Cell],
        exclude: Optional[set] = None,
        min_priority: float = 0.0,
    ) -> List[Optional[SearchPlan]]:
        """
        Assign DISJOINT top-priority sectors to N drones — the multi-drone coordination.

        Args:
            posterior: The fine belief (read only).
            coverage: The fine cleared-fraction layer (read only).
            drone_positions: One (row, col) per drone to assign.
            exclude: Sectors already LOCKED to other drones (mid-sweep) — never reassigned.
            min_priority: Skip sectors at/below this priority.

        Returns:
            A list parallel to `drone_positions`: each entry a SearchPlan or None (no sector
            left worth searching for that drone). No two returned plans share a sector.

        Why:
            This is C1's real payoff — coordinated, NON-OVERLAPPING coverage. Walk the priority
            ranking and hand each sector to the NEAREST still-free drone (cutting travel), locking
            it so no other drone is sent to the same ground. Greedy nearest-assignment is the
            standard, legible heuristic; the hard guarantee that matters (disjoint sectors) comes
            from growing the locked set as we assign.
        """
        locked = set(exclude or set())
        plans: List[Optional[SearchPlan]] = [None] * len(drone_positions)
        free = set(range(len(drone_positions)))
        for rs in self.rank(posterior, coverage):
            if not free:
                break
            if rs.priority <= min_priority:
                break                                # nothing worth searching remains
            if rs.sector in locked:
                continue
            center = self.sectors.center_cell(rs.sector)
            # Nearest free drone to this sector (squared distance — no sqrt needed for argmin).
            d = min(free, key=lambda i: (drone_positions[i][0] - center[0]) ** 2
                    + (drone_positions[i][1] - center[1]) ** 2)
            waypoints = boustrophedon_waypoints(
                self.sectors.bounds(rs.sector), self._cfg.swath_spacing_cells, drone_positions[d]
            )
            plans[d] = SearchPlan(rs.sector, rs.name, waypoints, waypoints[0])
            locked.add(rs.sector)
            free.discard(d)
        return plans
