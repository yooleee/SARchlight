# =============================================================================
# config.py
# -----------------------------------------------------------------------------
# Responsible for: One home for every tunable in the brain. Per the project
#                  conventions, tunables live in config, not scattered as magic
#                  numbers in logic, so they can be tuned live on Saturday.
# Role in project: Imported by the prior builder, the Bayesian update, the
#                  located trigger, the brain loop, and the demo. The grid
#                  geometry below also seeds the shared GridSpec (grid.py).
# Assumptions: Defaults are the demo-scenario values from docs/demo_scenario.md
#              and docs/prior_model.md. They are stage-appropriate, expected to
#              change; only their *home* (this file) is fixed.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class BrainConfig:
    """
    All tunable parameters for the search-map brain, with documented defaults.

    Why:
        The contracts and conventions name a fixed set of tunables (recall,
        LR_max, the LR buckets, P_located, persistence N, kernel width, p_out,
        cell_size_m, prior weights). Collecting them in one frozen dataclass
        makes "config not code" concrete: logic reads values from here, and a
        later YAML/TOML backing can load into this same object without touching
        the logic. Frozen because a config should not mutate mid-run.
    """

    # --- Grid geometry (seeds the shared GridSpec; docs/demo_scenario.md §1) ---
    cell_size_m: float = 50.0
    n_rows: int = 160
    n_cols: int = 160
    # SW corner of the grid = cell (0, 0). row increases north, col increases east.
    origin_latlon: Tuple[float, float] = (37.87, -122.66)
    # CRS tag only for now: the actual lat/lon<->cell math is a local-tangent
    # approximation (grid.py). pyproj-backed UTM is the upgrade behind that seam.
    crs: str = "EPSG:32610"  # UTM zone 10N

    # --- Incident / scenario (docs/demo_scenario.md §2) ---
    lkp_latlon: Tuple[float, float] = (37.905, -122.605)  # Pantoll trailhead

    # --- Search planner / sectorization (C1, docs/brain_followups.md C1) ---
    # The coarse TASKING grid: sectors are `sector_size_cells` x `sector_size_cells` fine
    # cells (10 x 50 m = 500 m, the SAR-typical sector size). Sectors are used ONLY to rank
    # and assign search (probability-of-area per sector, multi-drone no-overlap); the
    # posterior stays 50 m-fine — segmentation never coarsens the belief. Tunable: larger
    # sectors read more legibly for an operator; smaller ones task more finely.
    sector_size_cells: int = 10
    # Boustrophedon swath spacing (fine cells between adjacent sweep passes / waypoints).
    # ~= the sweep footprint cross-width at the demo altitude, so passes tile a sector with
    # neither gaps nor heavy overlap. Tunable with the real camera/altitude.
    swath_spacing_cells: int = 4

    # --- Non-detection clearing cap (interfaces §5.5: correlated looks) ---
    # Repeated passes from the SAME sensor over the same canopy/angle miss IDENTICALLY,
    # so the independent-looks product 1-Π(1-d_i) overstates how much one sensor can rule
    # out. Cap each sensor's CUMULATIVE clearance per cell: correlated looks can clear a
    # cell by at most this fraction. Different sensors clear independently (thermal sees
    # through canopy color can't), so the combined clearance can still exceed this.
    # Found necessary by the geo+brain feasibility test (loiter + canopy misses were
    # eroding the located subject); see docs/brain_followups.md A1.
    clearance_cap_per_sensor: float = 0.6

    # --- Prior construction (docs/prior_model.md §2-§3) ---
    prior_sigma_m: float = 2600.0        # half-normal sigma; Koester Hiker, Mtn/Temperate
    corridor_k: float = 3.0              # max corridor attraction C in [1, k]
    accessibility_epsilon: float = 0.05  # hard-but-passable floor for A (cliffs/water -> 0)
    p_out: float = 0.07                  # reserved "subject left the region" mass

    # --- Detection probability d_i = coverage*visibility*recall (interfaces §5.1) ---
    # recall(sensor, altitude): per-sensor true-positive rate. Thermal beats color
    # at dusk (the scenario's motivation). Stored as a tuple of pairs so the frozen
    # dataclass stays hashable; read via recall_for() in update.py.
    recall_by_sensor: Tuple[Tuple[str, float], ...] = (("color", 0.6), ("thermal", 0.8))

    # Sensor modulation of visibility (used by the GeoReferencer): thermal sees a person
    # better under canopy / low light, so it lifts the terrain's base visibility. Stored
    # as pairs (frozen-friendly); read via visibility_factor_for(). Clamped to [0,1] after.
    visibility_sensor_factor: Tuple[Tuple[str, float], ...] = (("color", 1.0), ("thermal", 1.4))

    # Thermal detection FLOOR for the detector simulator (detector_sim.py), THERMAL only.
    # Under dense canopy the base visibility is low (~0.40 tree cover), so even thermal's
    # visibility lift leaves detection marginal. A floor encodes the defensible dusk premise
    # that body heat penetrates canopy gaps: a thermal pass over a forested subject detects
    # with at least this probability. Default 0.0 = OFF (no floor) so synthetic-tuned demos
    # and all existing tests are unchanged; the real-terrain showcase raises it (~0.8) so the
    # canopy/thermal story actually resolves. Applied to thermal only — color stays gated by
    # canopy, which keeps the "thermal sees better under canopy" contrast honest.
    thermal_detection_floor: float = 0.0

    # --- Detection update: clipped, bucketed likelihood ratio (interfaces §5.3) ---
    lr_max: float = 20.0
    # Bucketed LR(c): (min_confidence, LR). The highest bucket whose threshold is
    # <= c wins. A step function is more honest than a smooth curve because raw
    # detector confidence is uncalibrated. Below 0.4 -> LR 1.0 (ignored).
    lr_buckets: Tuple[Tuple[float, float], ...] = (
        (0.0, 1.0),   # too weak to move the map
        (0.4, 5.0),   # modest evidence
        (0.7, 20.0),  # strong evidence (== lr_max)
    )
    # Detection spreads over a small gaussian neighborhood because a few pixels of
    # box-center error at altitude is tens of meters on the ground.
    kernel_sigma_cells: float = 1.0
    kernel_radius_cells: int = 2  # truncate the gaussian beyond this radius (cells)

    # --- located trigger (interfaces §5.4) ---
    # Two gates, OR'd, both behind the persistence gate:
    #   1. absolute: windowed posterior mass >= p_located (works on small/peaky maps)
    #   2. relative: concentration ratio >= located_concentration_ratio (grid-stable)
    # The relative gate is the robust one — it asks "is the find region N times denser
    # than the map's average?", which is FAR less grid-size dependent than an absolute
    # mass threshold. On a 160x160 grid a fixed window holds a tiny absolute share,
    # so p_located alone was brittle; the ratio carries the real grid. Default ~7 fires
    # on the demo's thermal corroboration (~10x) but not a lone color hit (~5x).
    # CALIBRATED TO THE DEMO — final tune is post-integration (see docs/brain_followups.md).
    p_located: float = 0.5                  # absolute windowed-mass gate
    located_concentration_ratio: float = 7.0  # relative gate: window density / map-average density
    persistence_n: int = 3                  # distinct frames hitting a blob to confirm
    window_halfwidth_cells: int = 2         # half-width of the aggregation window (5x5 at 2)
    blob_radius_cells: int = 3              # detections within this radius = the same blob

    def recall_for(self, sensor_type: str, altitude_agl_m: float | None = None) -> float:
        """
        Look up the detector recall for a sensor type (altitude is a seam, unused now).

        Args:
            sensor_type: "color" or "thermal" (the Observation's sensor_type value).
            altitude_agl_m: Height above ground; reserved for an altitude-dependent
                recall later. Ignored for v1 (a constant per sensor is enough).

        Returns:
            Recall in [0, 1] for that sensor; falls back to the color value if the
            sensor is unknown.

        Why:
            d_i = coverage * visibility * recall, and recall is the one piece that
            depends on the sensor (and, later, altitude). Keeping the table here and
            the lookup trivial means the d_i formula in update.py stays a one-liner.
        """
        table = dict(self.recall_by_sensor)
        return table.get(sensor_type, table.get("color", 0.6))

    def visibility_factor_for(self, sensor_type: str) -> float:
        """
        Look up the sensor's visibility multiplier (thermal helps under canopy).

        Args:
            sensor_type: "color" or "thermal".

        Returns:
            A multiplier applied to the terrain's base visibility; falls back to 1.0
            (color, no change) for an unknown sensor.

        Why:
            The GeoReferencer sets visibility_weight = base_visibility * this factor.
            Keeping the factor in config (not geo) means the sensor's advantage is a
            live-tunable, and geo stays pure geometry + a table lookup.
        """
        table = dict(self.visibility_sensor_factor)
        return table.get(sensor_type, 1.0)


# A module-level default instance for callers that don't thread a config through.
DEFAULT_CONFIG = BrainConfig()
