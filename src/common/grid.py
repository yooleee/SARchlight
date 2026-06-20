# =============================================================================
# grid.py
# -----------------------------------------------------------------------------
# Responsible for: GridSpec — the single shared frame of reference for every
#                  spatial component (the map, the georeferencer, the dashboard).
#                  Holds the region's geometry and the lat/lon <-> cell mapping.
# Role in project: interfaces.md §1 invariant. A cell is addressed as (row, col)
#                  everywhere. The brain, the prior, and all consumers index the
#                  same grid through this one object.
# Assumptions: For now lat/lon <-> cell uses a LOCAL EQUIRECTANGULAR (tangent-
#              plane) approximation, accurate to well under a cell over an ~8 km
#              box. The `crs` field tags the intended UTM zone; a pyproj-backed
#              exact projection is the upgrade behind these same two methods.
# Convention: origin = SW corner = cell (0, 0). row increases NORTH, col
#             increases EAST. (Plot with imshow(origin="lower") for north-up.)
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from .config import BrainConfig

# Meters per degree of latitude is nearly constant; longitude shrinks by cos(lat).
# These two constants are the whole local-tangent approximation.
_M_PER_DEG_LAT = 111_320.0


@dataclass(frozen=True)
class GridSpec:
    """
    The shared grid geometry: region anchor, cell size, and extent, plus the
    primitives to convert between (lat, lon) and (row, col).

    Args (fields):
        crs: Coordinate reference system tag (e.g. "EPSG:32610" for UTM 10N).
            Documentation/intent only while the math is local-tangent.
        origin: (lat, lon) of the SW corner — the anchor of cell (0, 0).
        cell_size_m: Square cell edge length in meters.
        n_rows: Number of rows (south -> north).
        n_cols: Number of columns (west -> east).

    Why:
        Every component must agree on one grid or detections and coverage land on
        different cells. Making it a frozen dataclass keeps that agreement
        immutable and trivially serializable so MapState can carry it across a
        process boundary later without a contract change.
    """

    crs: str
    origin: Tuple[float, float]  # (lat, lon) of SW corner
    cell_size_m: float
    n_rows: int
    n_cols: int

    @classmethod
    def from_config(cls, cfg: BrainConfig) -> "GridSpec":
        """
        Build the demo GridSpec from the geometry fields in BrainConfig.

        Args:
            cfg: The brain configuration holding cell_size_m, n_rows, n_cols,
                origin_latlon, and crs.

        Returns:
            A GridSpec instance for the demo region.

        Why:
            Keeps the grid's defining numbers in exactly one place (config) so the
            grid and the rest of the system can never disagree about cell size or
            extent. (DRY.)
        """
        return cls(
            crs=cfg.crs,
            origin=cfg.origin_latlon,
            cell_size_m=cfg.cell_size_m,
            n_rows=cfg.n_rows,
            n_cols=cfg.n_cols,
        )

    # --- local-tangent conversions (the only spatial primitives shared) ---

    def _m_per_deg_lon(self) -> float:
        """
        Meters per degree of longitude at the origin latitude.

        Returns:
            float meters/degree, shrunk from the latitude value by cos(origin_lat).

        Why:
            Longitude lines converge toward the poles; ignoring the cos factor would
            stretch the east-west axis (~21% error at this latitude) and distort
            every distance in the prior. Computing it once at the origin is exact
            enough across an 8 km box.
        """
        origin_lat_rad = math.radians(self.origin[0])
        return _M_PER_DEG_LAT * math.cos(origin_lat_rad)

    def latlon_to_local_m(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert a (lat, lon) to local meters (east, north) from the SW-corner origin.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            (east_m, north_m): meters east and north of the origin corner.

        Why:
            The prior's distance decay and all cell math are cleanest in a flat
            local meter frame. This is the single conversion the rest of the
            geometry builds on (cell indices are just these meters // cell_size).
        """
        east_m = (lon - self.origin[1]) * self._m_per_deg_lon()
        north_m = (lat - self.origin[0]) * _M_PER_DEG_LAT
        return east_m, north_m

    def latlon_to_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Map a (lat, lon) to its (row, col) cell index.

        Args:
            lat: Latitude in degrees.
            lon: Longitude in degrees.

        Returns:
            (row, col) integer cell index. May fall outside [0, n_rows) x
            [0, n_cols); use in_bounds() to check. Not clamped, so callers can
            detect points outside the region.

        Why:
            The georeferencer and the detection projection need a point's cell;
            keeping the conversion here (not duplicated per component) is the whole
            reason GridSpec exists.
        """
        east_m, north_m = self.latlon_to_local_m(lat, lon)
        # floor division so a point anywhere inside a cell maps to that cell's index.
        col = int(math.floor(east_m / self.cell_size_m))
        row = int(math.floor(north_m / self.cell_size_m))
        return row, col

    def cell_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """
        Map a (row, col) cell to the (lat, lon) of its CENTER.

        Args:
            row: Row index (south -> north).
            col: Column index (west -> east).

        Returns:
            (lat, lon) of the cell center in degrees.

        Why:
            LocatedEvent and the operator's spoken answers need a real-world
            coordinate for a cell. Using the center (the +0.5 offsets) is the least-
            biased single point to represent a cell.
        """
        east_m = (col + 0.5) * self.cell_size_m
        north_m = (row + 0.5) * self.cell_size_m
        lat = self.origin[0] + north_m / _M_PER_DEG_LAT
        lon = self.origin[1] + east_m / self._m_per_deg_lon()
        return lat, lon

    def in_bounds(self, row: int, col: int) -> bool:
        """
        Whether a (row, col) lies inside the grid.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if 0 <= row < n_rows and 0 <= col < n_cols.

        Why:
            Detections projected near the region edge (or a footprint cell off-grid)
            must be filtered before indexing the posterior array, or NumPy would
            wrap negative indices silently.
        """
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols
