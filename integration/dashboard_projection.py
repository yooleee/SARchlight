# =============================================================================
# integration/dashboard_projection.py
# -----------------------------------------------------------------------------
# Responsible for: The ONE projection from the brain's grid MapState (interfaces §6.1)
#                  to the dashboard's UI view-model (dashboard_app/src/types.ts MapState).
# Role in project: The brain->dashboard seam. The two MapStates are deliberately different
#                  (grid posterior + p_out vs normalized heatBlobs + telemetry), so instead
#                  of unifying them we bridge them HERE, in exactly one place. All UI drift
#                  is isolated to this file — change the dashboard's shape, change only this.
# Assumptions: The brain MapState is canonical for the LOOP; types.ts is canonical for the UI.
#              A single MapState snapshot lacks a few UI fields (the flown path, the live
#              telemetry pose, the trend HISTORY), so those come in via ProjectionContext,
#              which the caller (the server) accumulates across steps. Output is plain
#              JSON-serializable Python (dicts/lists/floats) — ready for FastAPI to return.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.common.config import BrainConfig
from src.common.contracts import CameraPose, MapState, Status
from src.common.grid import GridSpec
from src.search.trigger import concentration_ratio

Cell = Tuple[int, int]

# The dashboard's layer toggles (view-only UI state). Mirrors mockState.ts so the sidebar
# renders the same controls; the brain doesn't own these, so they are a static list here.
_LAYERS: List[Dict[str, Any]] = [
    {"id": "prob", "label": "Probability Map", "enabled": True},
    {"id": "path", "label": "Flight Path", "enabled": True},
    {"id": "detections", "label": "Detections", "enabled": True},
    {"id": "searched", "label": "Searched (No Detection)", "enabled": True},
    {"id": "terrain", "label": "Terrain", "enabled": True},
    {"id": "trails", "label": "Trails", "enabled": True},
    {"id": "drainages", "label": "Drainages", "enabled": True},
    {"id": "vegetation", "label": "Vegetation", "enabled": False},
    {"id": "slope", "label": "Slope", "enabled": False},
]

# How many heat blobs / detection rows / waypoints to surface — matches the mock's density so
# the UI stays readable. These are display caps, not data limits.
_N_HEAT_BLOBS = 6
_N_DETECTION_ROWS = 6
_HEAT_SUPPRESS_CELLS = 8  # min separation between blob centers (non-max suppression)


@dataclass
class ProjectionContext:
    """
    The cross-snapshot state the UI needs that a single MapState does not carry.

    Args (fields):
        drone_path: Ordered (row, col) cells the drone has flown — for flightPath/dronePos
            and the searched-waypoint trail. (MapState has next_target/search_path, not the
            HISTORY of where we've been.)
        last_pose: The most recent CameraPose, for live telemetry (altitude, heading, speed).
            None at t0 (before any frame).
        trend: Accumulated (elapsed_seconds, mass_0_100) points for the probability-trend
            chart — a history the server appends to each step.
        located_cell: The located subject cell once declared (marks the primary detection).
            None while still searching.
        mission_name / region / started_at: Cosmetic mission metadata shown in the header.
        guidance_path: The walkable route home (cells, subject -> operators) during the
            guide-home phase. None while still searching.
        subject_cell: The (moving) subject's current cell during guidance — the follower marker.
        home_cell: The operators/home cell (the LKP). Shown as the home marker.
        guidance_status: "searching" | "guiding" | "arrived" — drives the Guide Home ribbon step.

    Why:
        Keeping these in a context object (rather than stuffing them into MapState) preserves
        the brain's read-model as purely the belief state, and keeps this projection a pure
        function of (snapshot, context) — trivially testable without a running server. The
        guidance fields are the additive seam for the drone-as-guide phase (all None/searching
        until the server enters guidance, so existing consumers are unaffected).
    """

    drone_path: List[Cell] = field(default_factory=list)
    last_pose: Optional[CameraPose] = None
    trend: List[Tuple[float, float]] = field(default_factory=list)
    located_cell: Optional[Cell] = None
    mission_name: str = "Lost Hiker — Marin"
    region: str = "Mt. Tamalpais, Marin CA"
    started_at: str = "—"
    guidance_path: Optional[List[Cell]] = None
    subject_cell: Optional[Cell] = None
    home_cell: Optional[Cell] = None
    guidance_status: str = "searching"


# --- small formatting / geometry helpers (pure) ---


def _fmt_clock(seconds: float) -> str:
    """Format elapsed sim-seconds as HH:MM:SS. Why: the UI timestamps are HH:MM:SS strings."""
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _fmt_hm(seconds: float) -> str:
    """Format elapsed sim-seconds as HH:MM (for the trend x-axis ticks)."""
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


def _cell_to_norm(cell: Cell, grid: GridSpec) -> Dict[str, float]:
    """
    Map a (row, col) cell CENTER to the UI's normalized {x, y} in [0, 1], x right / y down.

    Why:
        The dashboard draws everything in a normalized 0..1 box, now OVER a server-rendered base
        image (integration/map_render). That image is a matplotlib imshow whose cells span the
        extent [-0.5, n-0.5], so a cell's CENTER sits at fraction (c+0.5)/n_cols horizontally and
        (r+0.5)/n_rows vertically. We use that exact cell-center form here (not the old corner
        form c/(n_cols-1)) so the vector overlays line up with the base image pixel-for-pixel.
        Row 0 is SOUTH (the base renders origin='lower') and the UI's y points DOWN, so we flip y
        to keep north up. ONE helper for every overlay guarantees they share the base's frame.
    """
    r, c = cell
    x = (c + 0.5) / grid.n_cols
    y = 1.0 - (r + 0.5) / grid.n_rows
    return {"x": round(x, 4), "y": round(y, 4)}


def _heat_blobs(posterior, grid: GridSpec) -> List[Dict[str, float]]:
    """
    Downsample the posterior into a few spatially-separated heat blobs for the UI heatmap.

    Args:
        posterior: (n_rows, n_cols) probability array.
        grid: GridSpec for normalization.

    Returns:
        Up to _N_HEAT_BLOBS dicts {x, y, intensity in [0,1], radius}, brightest first.

    Why:
        The grid has tens of thousands of cells; the UI wants a handful of hot spots. We take
        the top cells under non-max suppression (skip any cell within _HEAT_SUPPRESS_CELLS of
        one already chosen) so the blobs spread across the high-probability region instead of
        all stacking on the single peak — a readable heatmap, not one dot. Intensity is
        normalized to the peak; radius grows slightly for dimmer (more diffuse) blobs.
    """
    import numpy as np

    peak = float(posterior.max())
    if peak <= 0.0:
        return []
    # Consider only the strongest candidates, then thin them by spatial suppression.
    flat = posterior.ravel()
    n_candidates = min(flat.size, 400)
    candidate_idx = np.argpartition(flat, -n_candidates)[-n_candidates:]
    candidate_idx = candidate_idx[np.argsort(flat[candidate_idx])[::-1]]  # strongest first

    chosen: List[Cell] = []
    blobs: List[Dict[str, float]] = []
    for idx in candidate_idx:
        if len(blobs) >= _N_HEAT_BLOBS:
            break
        r, c = int(idx // grid.n_cols), int(idx % grid.n_cols)
        # Non-max suppression: keep this cell only if it's far enough from all kept blobs.
        if any(max(abs(r - pr), abs(c - pc)) < _HEAT_SUPPRESS_CELLS for pr, pc in chosen):
            continue
        chosen.append((r, c))
        intensity = float(posterior[r, c]) / peak
        pos = _cell_to_norm((r, c), grid)
        blobs.append(
            {
                "x": pos["x"],
                "y": pos["y"],
                "intensity": round(intensity, 3),
                "radius": round(0.05 + 0.05 * (1.0 - intensity), 3),  # dimmer -> a bit larger
            }
        )
    return blobs


def _ground_distance_m(a: Cell, b: Cell, cfg: BrainConfig) -> float:
    """Euclidean ground distance between two cells, in meters. Why: detection 'distance from drone'."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) * cfg.cell_size_m


def _ui_detections(
    map_state: MapState, context: ProjectionContext, grid: GridSpec, cfg: BrainConfig
) -> List[Dict[str, Any]]:
    """
    Collapse the brain's per-frame detections_log into the UI's detection rows.

    Args:
        map_state: The snapshot (its detections_log holds per-frame detection events).
        context: For the drone position (distance) and the located cell (primary marker).
        grid, cfg: Geometry + tunables (persistence N for the seen/of badge).

    Returns:
        Up to _N_DETECTION_ROWS UI Detection dicts, most recent first.

    Why:
        The log is one entry per detection per frame — too granular for a UI list. We GROUP by
        projected cell so repeated hits on the same blob become one row whose persistence
        (seen/of) and confidence (max) summarize the track, matching how the brain actually
        reasons (a blob seen across N frames), and mark the located cell as the primary.
    """
    groups: Dict[Cell, Dict[str, Any]] = {}
    for entry in map_state.detections_log:
        cell = (int(entry["cell"][0]), int(entry["cell"][1]))
        g = groups.setdefault(cell, {"count": 0, "max_conf": 0.0, "last_ts": 0.0})
        g["count"] += 1
        g["max_conf"] = max(g["max_conf"], float(entry["confidence"]))
        g["last_ts"] = max(g["last_ts"], float(entry["timestamp"]))

    located = context.located_cell
    if located is None and map_state.status is Status.LOCATED and map_state.top_cells:
        located = map_state.top_cells[0][0]

    drone_cell = context.drone_path[-1] if context.drone_path else None
    persistence_of = cfg.persistence_n

    rows: List[Dict[str, Any]] = []
    # Most recent groups first (the dashboard timeline reads newest-at-top).
    for rank, (cell, g) in enumerate(sorted(groups.items(), key=lambda kv: kv[1]["last_ts"], reverse=True)):
        if rank >= _N_DETECTION_ROWS:
            break
        is_primary = located is not None and cell == located
        pos = _cell_to_norm(cell, grid)
        rows.append(
            {
                "id": f"d{rank + 1}",
                "timestamp": _fmt_clock(g["last_ts"]),
                "confidence": round(g["max_conf"], 3),
                "persistence": {"seen": min(g["count"], persistence_of), "of": persistence_of},
                "label": "LIKELY PERSON" if (is_primary or g["max_conf"] >= 0.8) else "PERSON",
                "distanceM": round(_ground_distance_m(drone_cell, cell, cfg)) if drone_cell else 0,
                "isPrimary": is_primary,
                "pos": pos,
                # Deterministic placeholder thumbnail tint, stable per cell.
                "thumbnailHue": (cell[0] * 73 + cell[1] * 31) % 360,
            }
        )
    return rows


def _loop_steps(map_state: MapState, guidance_status: str = "searching") -> List[Dict[str, Any]]:
    """
    Derive the loop-ribbon step states from the snapshot's status + the guidance phase.

    Args:
        map_state: The brain snapshot (status/update_count).
        guidance_status: "searching" | "guiding" | "arrived" — the guide-home phase.

    Why:
        The loop ribbon tells the story (Build Prior -> ... -> Notify -> Guide Home). We light it
        from real state: the prior/plan are always done; detect/update/redirect are live
        mid-search and done once located; notify lights at the locate; Guide Home lights while
        guiding and completes on arrival. Honest, not scripted.
    """
    located = map_state.status is Status.LOCATED
    updated = map_state.update_count > 0
    guiding = guidance_status == "guiding"
    arrived = guidance_status == "arrived"
    mid = "done" if located else ("live" if updated else "next")
    # Notify fires at the locate; it's "done" once we've moved on to guiding/arrived.
    notify = "done" if (located and (guiding or arrived)) else ("live" if located else "next")
    guide = "done" if arrived else ("live" if guiding else "next")
    return [
        {"id": 1, "label": "Build Prior", "status": "done"},
        {"id": 2, "label": "Plan Search", "status": "done"},
        {"id": 3, "label": "Detect", "status": "done" if located else "live"},
        {"id": 4, "label": "Update Map", "status": mid},
        {"id": 5, "label": "Redirect", "status": mid},
        {"id": 6, "label": "Notify", "status": notify},
        {"id": 7, "label": "Guide Home", "status": guide},
    ]


def _telemetry(map_state: MapState, context: ProjectionContext, cfg: BrainConfig) -> Dict[str, Any]:
    """
    Build the live telemetry readout from the most recent pose + path.

    Why:
        The UI feed header shows altitude/heading/speed/time. Altitude and heading come
        straight off the last CameraPose; speed is estimated from the last path step over the
        ~9 s frame interval (0 during a loiter, which is correct). All cosmetic but real.
    """
    pose = context.last_pose
    alt = round(pose.altitude_agl_m) if pose else 0
    hdg = round(pose.heading_deg) % 360 if pose else 0
    # Speed from the last flown step (cells * cell_size / frame_dt). Loiter -> ~0, as expected.
    spd = 0.0
    if len(context.drone_path) >= 2:
        step_m = _ground_distance_m(context.drone_path[-1], context.drone_path[-2], cfg)
        spd = round(step_m / 9.0, 1)  # 9 s frame interval (mock_stream _FRAME_DT_S)
    return {"altM": alt, "spdMs": spd, "hdgDeg": hdg, "feedTime": _fmt_clock(map_state.timestamp)}


def _waypoints(map_state: MapState, context: ProjectionContext, grid: GridSpec) -> List[Dict[str, Any]]:
    """
    A small set of map markers: the high-probability target, detection cells, searched trail.

    Why:
        The map overlay wants a few labeled points, not the whole grid. We surface the single
        highest-probability cell (high-prob), the recent detection cells (detection), and a
        thinned sample of the flown path (searched) — enough to annotate the map without clutter.
    """
    waypoints: List[Dict[str, Any]] = []
    if map_state.top_cells:
        waypoints.append({"id": "wp-top", "kind": "high-prob", "pos": _cell_to_norm(map_state.top_cells[0][0], grid)})
    # Detection cells (unique), from the log.
    seen: set = set()
    for entry in map_state.detections_log:
        cell = (int(entry["cell"][0]), int(entry["cell"][1]))
        if cell in seen:
            continue
        seen.add(cell)
        waypoints.append({"id": f"wp-d{len(seen)}", "kind": "detection", "pos": _cell_to_norm(cell, grid)})
        if len(seen) >= 4:
            break
    # A thinned sample of the searched path.
    path = context.drone_path
    for i in range(0, len(path), max(1, len(path) // 5)):
        waypoints.append({"id": f"wp-s{i}", "kind": "searched", "pos": _cell_to_norm(path[i], grid)})
    return waypoints


def project(map_state: MapState, context: Optional[ProjectionContext] = None) -> Dict[str, Any]:
    """
    Project a brain MapState (+ accumulated context) into the dashboard's UI MapState dict.

    Args:
        map_state: The brain's published snapshot (interfaces §6.1).
        context: Cross-snapshot UI state (flown path, last pose, trend history, metadata).
            Defaults to an empty context (renders a valid, if path-less, state at t0).

    Returns:
        A plain dict matching dashboard_app/src/types.ts MapState — JSON-serializable as-is,
        with all positions in [0, 1] and confidences/intensities normalized.

    Why:
        This single function IS the brain<->dashboard contract bridge. Everything the React app
        reads is produced here, so if the UI shape drifts, only this file changes — the loop,
        the brain, and the detector are all untouched.
    """
    ctx = context or ProjectionContext()
    grid = map_state.grid_spec
    cfg = BrainConfig(
        cell_size_m=grid.cell_size_m,
        n_rows=grid.n_rows,
        n_cols=grid.n_cols,
    )
    # The thresholds depend on the brain's actual tunables; rebuild a cfg only for geometry
    # above, but read the locate gate off the real config the snapshot implies. We can't see
    # the brain's cfg here, so use the default located_concentration_ratio unless the caller
    # overrides via context. (The server passes the loop's cfg-derived values when it matters.)
    located = map_state.status is Status.LOCATED

    # Confidence gauge: how concentrated is the belief at its peak, normalized to the locate
    # floor. Snaps to 100 exactly when the brain declares (the real, multi-factor trigger) so
    # the gauge can never disagree with the brain about whether we've located.
    peak_cell = map_state.top_cells[0][0] if map_state.top_cells else (grid.n_rows // 2, grid.n_cols // 2)
    conc = concentration_ratio(map_state.posterior, peak_cell, cfg.window_halfwidth_cells)
    floor = cfg.located_concentration_ratio
    confidence = 100 if located else int(round(100 * min(0.99, conc / floor if floor else 0.0)))

    # Accumulated cleared area + flown distance from the path history.
    import numpy as np

    cell_area_km2 = (grid.cell_size_m ** 2) / 1e6
    area_covered = float(np.asarray(map_state.coverage).sum()) * cell_area_km2
    flight_km = 0.0
    for a, b in zip(ctx.drone_path, ctx.drone_path[1:]):
        flight_km += _ground_distance_m(a, b, cfg)
    flight_km /= 1000.0

    drone_pos = _cell_to_norm(ctx.drone_path[-1], grid) if ctx.drone_path else {"x": 0.5, "y": 0.5}

    # Guide-home overlay (all None/"searching" until the server enters the guidance phase).
    guidance_path = (
        [_cell_to_norm(c, grid) for c in ctx.guidance_path] if ctx.guidance_path else None
    )

    return {
        "missionName": ctx.mission_name,
        "region": ctx.region,
        "startedAt": ctx.started_at,
        "loop": _loop_steps(map_state, ctx.guidance_status),
        "confidenceToDeclare": confidence,
        "declareThreshold": 100,  # gauge meets threshold exactly when the brain declares located
        "stats": {
            "searchTime": _fmt_clock(map_state.timestamp),
            "areaCoveredKm2": round(area_covered, 2),
            "flightDistanceKm": round(flight_km, 1),
            "detections": len(map_state.detections_log),
            "personsFound": 1 if located else 0,
        },
        "layers": _LAYERS,
        "detections": _ui_detections(map_state, ctx, grid, cfg),
        "waypoints": _waypoints(map_state, ctx, grid),
        "flightPath": [_cell_to_norm(c, grid) for c in ctx.drone_path],
        "dronePos": drone_pos,
        "heatBlobs": _heat_blobs(map_state.posterior, grid),
        "trend": [{"t": _fmt_hm(t), "mass": round(m)} for t, m in ctx.trend],
        "recentCommands": [],  # populated by the voice layer in Milestone 2
        "telemetry": _telemetry(map_state, ctx, cfg),
        # --- guide-home overlay (additive; null/"searching" during the search phase) ---
        "guidancePath": guidance_path,
        "subjectPos": _cell_to_norm(ctx.subject_cell, grid) if ctx.subject_cell else None,
        "operatorPos": _cell_to_norm(ctx.home_cell, grid) if ctx.home_cell else None,
        "guidanceStatus": ctx.guidance_status,
    }
