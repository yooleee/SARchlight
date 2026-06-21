# =============================================================================
# search_demo.py
# -----------------------------------------------------------------------------
# Responsible for: The C1 CLOSED-LOOP search demo — the flight path EMERGES from the
#                  sector planner instead of being scripted. A drone starts at the LKP;
#                  each round the planner ranks sectors by probability-of-area, the
#                  drone sweeps the top sector, the map updates and re-ranks, and the
#                  drone advances — until it detects and corroborates the subject.
# Role in project: docs/brain_followups.md C1. Shows "the map directs the search" for
#                  real (the scripted showcase pre-knew where the subject was). Single-
#                  drone here; the multi-drone orchestrator is added in the same file.
# Run: PYTHONPATH=. .venv/bin/python -m src.demo.search_demo
# Assumptions: the real DEM + WorldCover rasters exist (errors clearly if absent). The
#              detector is SIMULATED; geo + the brain + the planner are the real code.
# =============================================================================

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from src.common.config import BrainConfig
from src.common.contracts import LocatedEvent, MapState, SensorType
from src.common.grid import GridSpec
from src.geo.georeferencer import GeoReferencer, _pixel_to_ground_local
from src.search.brain import SearchBrain
from src.search.planner import RankedSector, Sector, SectorPlanner
from src.search.terrain_raster import RasterTerrain
from src.demo.detector_sim import DetectorSimulator
from src.demo.mock_stream import REAL_TERRAIN_SUBJECT_OFFSET, loiter_pose, subject_cell, sweep_pose
from src.demo.showcase import _frame_to_image, draw_belief_layer, hillshade

# Real-terrain tuning + a legible 1 km sector size for the tasking grid.
# - swath_spacing 3 (not the default 4): the sweep footprint is narrow along-track (~1 cell of
#   vertical FOV at this altitude), so passes must be <=3 cells apart to avoid along-track gaps.
# - thermal_detection_floor 0.95 (vs the showcase's 0.8): this demo is about COORDINATION, not
#   detection realism (the showcase owns the realistic color-misses-thermal-finds story). A
#   subject on a sector boundary is covered by only one sweep pass, so a near-certain in-frame
#   thermal detection keeps the find reliable across seeds (8/8 single AND multi-drone).
PLANNER_CONFIG = BrainConfig(
    located_concentration_ratio=3.5, thermal_detection_floor=0.95,
    sector_size_cells=20, swath_spacing_cells=3,
)
_SUBJECT_HIT_RADIUS_M = 150.0
_CONFIRM_FRAMES = 6   # thermal loiter frames on contact, to build persistence + concentration
_TOP_K_SECTORS = 5          # how many ranked sectors to label in the viz
# Distinct colors per drone (single-drone uses the first); enough for a small fleet.
_DRONE_COLORS = ["deepskyblue", "gold", "springgreen", "magenta", "orangered"]
_OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "demo_output"


# =============================================================================
# Per-frame state (presentation-free) — multi-drone ready (a list of drone views).
# =============================================================================

@dataclass(frozen=True)
class DroneView:
    """One drone's state for a frame: who, where it's been, and its assigned sector."""

    drone_id: int
    path: List[Tuple[int, int]]
    target_sector: Optional[Sector]
    target_name: Optional[str]


@dataclass(frozen=True)
class PlannerFrame:
    """
    One animation frame of the closed loop — raw state, no matplotlib.

    Why:
        Recorded per observation so the render is a pure function of state (same split as the
        showcase). Holds the belief, every drone's path + assigned sector, the current sector
        ranking (for the overlay), and the caption/stat beats.
    """

    update_count: int
    posterior: np.ndarray
    drones: List[DroneView]
    ranked_sectors: List[RankedSector]
    caption: str
    status: str
    p_out: float
    located_now: bool
    # A snapshot of the brain's full MapState at this frame. Optional/default-None so the GIF
    # path (which only reads posterior/drones/sectors) is unaffected; the live dashboard server
    # uses it to project the side panels per frame. Captured in record() via brain.map_state().
    map_state: Optional["MapState"] = None


@dataclass
class PlannerResult:
    """The recorded run: grid/sectors/subject + the frames + the located outcome."""

    cfg: BrainConfig
    grid: GridSpec
    planner: SectorPlanner
    subject: Tuple[int, int]
    frames: List[PlannerFrame] = field(default_factory=list)
    located_event: Optional[LocatedEvent] = None
    n_drones: int = 1


# =============================================================================
# World setup + small helpers.
# =============================================================================

@dataclass
class _World:
    """The shared loop ingredients (terrain, brain, geo, sim, subject)."""

    cfg: BrainConfig
    grid: GridSpec
    brain: SearchBrain
    geo: GeoReferencer
    sim: DetectorSimulator
    subject: Tuple[int, int]
    subj_e: float
    subj_n: float
    lkp: Tuple[int, int]


def _build_world(cfg: BrainConfig, seed: int) -> _World:
    """
    Assemble the real-terrain world: brain (real), geo (real), detector (simulated), subject.

    Why:
        Same ingredients as the scripted showcase, minus the scripted path — here the path
        comes from the planner. The subject is the same NW real-terrain cell.
    """
    grid = GridSpec.from_config(cfg)
    terrain = RasterTerrain(cfg)
    brain = SearchBrain(cfg, terrain)
    subject = subject_cell(grid, cfg, offset=REAL_TERRAIN_SUBJECT_OFFSET)
    subject_latlon = grid.cell_to_latlon(*subject)
    subj_e, subj_n = grid.latlon_to_local_m(*subject_latlon)
    visibility = terrain.visibility(grid)
    geo = GeoReferencer(grid, visibility=visibility, config=cfg)
    sim = DetectorSimulator(grid, subject_latlon, visibility, config=cfg, seed=seed,
                            false_positive_rate=0.01)
    lkp = grid.latlon_to_cell(*cfg.lkp_latlon)
    return _World(cfg, grid, brain, geo, sim, subject, subj_e, subj_n, lkp)


def _heading(from_cell: Tuple[int, int], to_cell: Tuple[int, int]) -> float:
    """Compass heading (0=N, 90=E) from one cell to another; 90 if they coincide."""
    dr, dc = to_cell[0] - from_cell[0], to_cell[1] - from_cell[1]
    if dr == 0 and dc == 0:
        return 90.0
    return math.degrees(math.atan2(dc, dr)) % 360.0


def _subject_hit_cell(det_out, pose, grid: GridSpec, subj_e: float, subj_n: float) -> Optional[Tuple[int, int]]:
    """
    The geo-projected cell of a real subject detection in this frame, or None.

    Why:
        On a sweep contact the drone must loiter over the ESTIMATED subject location (where the
        detection projects on the ground), not the raw sweep waypoint — the waypoint can be a
        cell or two off the subject, and the lower, more-oblique loiter footprint is too narrow
        to still cover the subject from there, so the confirm pass would never build persistence.
        Classifying by true geo error (meters), as the showcase does, tells a real hit from a
        false positive; returning its cell gives the loiter a target.
    """
    for det in det_out.detections:
        x, y, w, h = det.bbox_xywh
        ge, gn = _pixel_to_ground_local(x + w / 2.0, y + h / 2.0, pose, grid)
        if math.hypot(ge - subj_e, gn - subj_n) <= _SUBJECT_HIT_RADIUS_M:
            return grid.local_m_to_cell(ge, gn)
    return None


# =============================================================================
# Single-drone closed loop.
# =============================================================================

class _Drone:
    """
    One drone's live state in the closed loop: where it is, its locked sector + sweep
    progress, and whether it's confirming a contact.

    Why:
        The multi-drone loop advances every drone one step per tick; this holds the per-drone
        bookkeeping (current plan, waypoint index, confirm countdown) so the tick logic stays a
        simple "if confirming … elif sweeping … else done". A drone keeps its sector LOCKED until
        it finishes sweeping or confirming — that's what guarantees no two drones share ground.
    """

    def __init__(self, drone_id: int, pos: Tuple[int, int]) -> None:
        self.id = drone_id
        self.pos = pos
        self.path: List[Tuple[int, int]] = []
        self.plan = None
        self.wp_i = 0
        self.confirming = False
        self.confirm_left = 0
        self.confirm_cell: Optional[Tuple[int, int]] = None

    @property
    def active(self) -> bool:
        """Has work to do (a sector still being swept, or a contact being confirmed)."""
        return self.plan is not None and (self.confirming or self.wp_i < len(self.plan.waypoints))

    def assign(self, plan) -> None:
        """Take a new sector assignment (None = nothing left to search)."""
        self.plan = plan
        self.wp_i = 0
        self.confirming = False
        self.confirm_left = 0

    def next_waypoint(self) -> Tuple[int, int]:
        wp = self.plan.waypoints[self.wp_i]
        self.wp_i += 1
        return wp

    def swept_out(self) -> bool:
        return self.wp_i >= len(self.plan.waypoints)

    def start_confirm(self, cell: Tuple[int, int]) -> None:
        self.confirming = True
        self.confirm_cell = cell
        self.confirm_left = _CONFIRM_FRAMES

    def finish(self) -> None:
        """Release the sector (swept out or confirm done) so the drone can be re-tasked."""
        self.plan = None
        self.confirming = False
        self.confirm_left = 0


def _caption(drones: List[_Drone], located_by: Optional[int]) -> str:
    """The beat caption for a tick: contact > sweeping > idle."""
    confirming = [d for d in drones if d.active and d.confirming]
    if confirming:
        d = confirming[0]
        return f"CONTACT — drone {d.id} corroborating in sector {d.plan.target_name} (thermal)"
    active = [d for d in drones if d.active]
    names = [d.plan.target_name for d in active if d.plan is not None]
    if names:
        return (f"SECTOR SEARCH — {len(active)} drone(s) sweeping disjoint sectors: "
                f"{', '.join(names)}")
    return "SECTOR SEARCH"


def _run_closed_loop(cfg: BrainConfig, seed: int, n_drones: int, max_obs: int) -> PlannerResult:
    """
    The closed loop for N drones: each tick, assign free drones to disjoint top sectors, then
    advance every active drone one step (sweep or confirm), recording a frame per tick.

    Args:
        cfg: Brain + planner config.
        seed: Detector-sim seed.
        n_drones: How many drones to coordinate (1 = the single-drone demo).
        max_obs: Safety cap on total observations.

    Returns:
        A PlannerResult with the recorded frames and the located outcome.

    Why:
        One implementation serves single- and multi-drone (DRY): with N=1 it is exactly the
        sweep-a-sector-then-re-task loop; with N>1 the SAME logic runs every drone in lockstep,
        feeding all N observation streams into the ONE brain (single writer preserved). The
        no-overlap guarantee is structural: `assign_multi` only ever hands out sectors not
        already locked to an active drone.
    """
    w = _build_world(cfg, seed)
    planner = w.brain.planner
    result = PlannerResult(cfg=cfg, grid=w.grid, planner=planner, subject=w.subject, n_drones=n_drones)
    drones = [_Drone(i, w.lkp) for i in range(n_drones)]
    located = {"event": None, "by": None}
    state = {"frame_no": 1, "t": 0.0, "n_obs": 0}

    def fly_one(drone: _Drone, cell, sensor: SensorType, is_sweep: bool) -> Optional[Tuple[int, int]]:
        """Advance one drone one frame; return the subject-detection cell if a real hit."""
        heading = _heading(drone.pos, cell)
        fid = f"D{drone.id}-{state['frame_no']:04d}"
        pose = (sweep_pose(w.grid, cell, fid, heading) if is_sweep
                else loiter_pose(w.grid, cell, fid, heading, sensor))
        det = w.sim.simulate(pose, sensor, state["t"])
        ev = w.brain.step(w.geo.reference(det, pose))   # the ONE brain — single writer
        drone.pos = cell
        drone.path.append(cell)
        state["frame_no"] += 1
        state["t"] += 9.0
        state["n_obs"] += 1
        if ev is not None and located["event"] is None:
            located["event"], located["by"] = ev, drone.id
            result.located_event = ev
        return _subject_hit_cell(det, pose, w.grid, w.subj_e, w.subj_n)

    def record(caption: str) -> None:
        ranked = planner.rank(w.brain.posterior, w.brain.coverage, k=_TOP_K_SECTORS)
        views = [DroneView(d.id, list(d.path), d.plan.target_sector if d.plan else None,
                           d.plan.target_name if d.plan else None) for d in drones]
        result.frames.append(PlannerFrame(
            update_count=w.brain.update_count, posterior=w.brain.posterior.copy(), drones=views,
            ranked_sectors=ranked, caption=caption, status=w.brain.status.value,
            p_out=w.brain.p_out, located_now=located["event"] is not None,
            map_state=w.brain.map_state(),   # full read-model snapshot for the live dashboard panels
        ))

    record(f"PRIOR — {n_drones} drone(s), sectors ranked by probability-of-area (POA)")

    while located["event"] is None and state["n_obs"] < max_obs:
        # 1) Assign free drones to disjoint top sectors (locking out active drones' sectors).
        locked = {d.plan.target_sector for d in drones if d.active}
        free = [d for d in drones if not d.active]
        if free:
            plans = planner.assign_multi(w.brain.posterior, w.brain.coverage,
                                         [d.pos for d in free], exclude=locked)
            for d, plan in zip(free, plans):
                d.assign(plan)
        if all(not d.active for d in drones):
            break  # nothing left worth searching

        # 2) Advance every active drone one step (sweep one waypoint, or one confirm frame).
        for d in drones:
            if not d.active:
                continue
            if d.confirming:
                fly_one(d, d.confirm_cell, SensorType.THERMAL, is_sweep=False)
                d.confirm_left -= 1
                if d.confirm_left <= 0:
                    d.finish()
            else:
                hit_cell = fly_one(d, d.next_waypoint(), SensorType.THERMAL, is_sweep=True)
                if hit_cell is not None:
                    d.start_confirm(hit_cell)   # contact -> loiter over the estimate to lock on
                elif d.swept_out():
                    d.finish()                  # sector cleared -> re-task next tick
            if located["event"] is not None or state["n_obs"] >= max_obs:
                break

        record(_caption(drones, located["by"]))

    # Re-caption the located frames so the climax reads as the payoff.
    if located["event"] is not None:
        name = planner.sectors.name(planner.sectors.sector_of(located["event"].cell))
        for i, fr in enumerate(result.frames):
            if fr.status == "located":
                result.frames[i] = _relabel(
                    fr, f"★ LOCATED by drone {located['by']} in sector {name} — broadcasting")
    return result


def run_single_drone(cfg: BrainConfig = PLANNER_CONFIG, seed: int = 0, max_obs: int = 160) -> PlannerResult:
    """One drone, planner-directed, closed loop (a thin alias for _run_closed_loop with N=1)."""
    return _run_closed_loop(cfg, seed, n_drones=1, max_obs=max_obs)


def run_multi_drone(cfg: BrainConfig = PLANNER_CONFIG, seed: int = 0,
                    n_drones: int = 3, max_obs: int = 300) -> PlannerResult:
    """
    N drones, coordinated by the planner over DISJOINT sectors, closed loop.

    Why:
        C1's headline payoff: several drones cover the area at once with no overlap (each locked
        to its own sector) and re-balance as the map updates — all feeding the one single-writer
        brain. The win over one drone is coordinated coverage + operator legibility, not raw
        single-drone speed.
    """
    return _run_closed_loop(cfg, seed, n_drones=n_drones, max_obs=max_obs)


def _relabel(frame: PlannerFrame, caption: str) -> PlannerFrame:
    """Return a copy of a frame with a new caption (frames are frozen)."""
    return PlannerFrame(
        update_count=frame.update_count, posterior=frame.posterior, drones=frame.drones,
        ranked_sectors=frame.ranked_sectors, caption=caption, status=frame.status,
        p_out=frame.p_out, located_now=frame.located_now, map_state=frame.map_state,
    )


# =============================================================================
# Rendering: belief + sector overlay + drone paths.
# =============================================================================

def _draw_sectors(ax, planner: SectorPlanner, ranked: List[RankedSector], drones: List[DroneView]) -> None:
    """
    Draw the coarse sector grid, label the top-K by POA, and highlight each drone's sector.

    Why:
        The sector overlay is C1's legibility: faint grid lines show the tasking sectors, the
        top-K carry their name + POA %, and a bold outline in the drone's color shows where each
        drone is assigned — "drone 0 is sweeping D4 (18%)".
    """
    sectors = planner.sectors
    sz = sectors.sector_size_cells
    # Faint sector grid lines.
    for sc in range(sectors.n_sector_cols + 1):
        ax.axvline(sc * sz - 0.5, color="white", lw=0.4, alpha=0.25)
    for sr in range(sectors.n_sector_rows + 1):
        ax.axhline(sr * sz - 0.5, color="white", lw=0.4, alpha=0.25)
    # Top-K sectors: outline + name + POA percentage of in-region mass.
    for rank, rs in enumerate(ranked):
        r0, r1, c0, c1 = sectors.bounds(rs.sector)
        lw = 2.2 if rank == 0 else 1.2
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0,
                               fill=False, edgecolor="white", lw=lw, alpha=0.8 if rank == 0 else 0.5))
        cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
        ax.text(cc, cr, f"{rs.name}\n{rs.poa * 100:.1f}%", color="white", ha="center", va="center",
                fontsize=8, fontweight="bold" if rank == 0 else "normal", alpha=0.9)
    # Each drone's assigned sector, bold in its color.
    for d in drones:
        if d.target_sector is None:
            continue
        r0, r1, c0, c1 = sectors.bounds(d.target_sector)
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False,
                               edgecolor=_DRONE_COLORS[d.drone_id % len(_DRONE_COLORS)], lw=2.6))


def _draw_markers_and_drones(ax, grid: GridSpec, lkp, subject, drones: List[DroneView]) -> None:
    """Region border, LKP, subject (truth), and each drone's flown path + current position."""
    ax.add_patch(Rectangle((-0.5, -0.5), grid.n_cols, grid.n_rows, fill=False,
                           edgecolor="lime", linewidth=2.0, clip_on=False))
    ax.scatter([lkp[1]], [lkp[0]], marker="^", s=80, c="cyan", edgecolors="black", zorder=6, label="LKP")
    ax.scatter([subject[1]], [subject[0]], marker="*", s=170, c="red", edgecolors="black", zorder=6, label="subject (truth)")
    for d in drones:
        color = _DRONE_COLORS[d.drone_id % len(_DRONE_COLORS)]
        if d.path:
            ax.plot([c for _, c in d.path], [r for r, _ in d.path], "-", color=color, lw=1.3, alpha=0.9, zorder=4)
            ax.plot(d.path[-1][1], d.path[-1][0], "o", color=color, ms=7, markeredgecolor="black", zorder=5,
                    label=f"drone {d.drone_id}")
    ax.set_xlabel("column (50 m cells)")
    ax.set_ylabel("row (50 m cells)  •  full frame = 8 km × 8 km")
    ax.legend(loc="upper right", fontsize=7)


def render_planner_frame(ax, result: PlannerResult, hill: np.ndarray, frame: PlannerFrame) -> None:
    """Draw one closed-loop frame: belief + sector overlay + markers + drone paths + caption."""
    ax.clear()
    draw_belief_layer(ax, frame.posterior, hill)
    _draw_sectors(ax, result.planner, frame.ranked_sectors, frame.drones)
    _draw_markers_and_drones(ax, result.grid, result.grid.latlon_to_cell(*result.cfg.lkp_latlon),
                             result.subject, frame.drones)
    ax.set_title(frame.caption, fontsize=12, fontweight="bold", pad=10)
    n_flown = sum(len(d.path) for d in frame.drones)
    ax.text(0.5, -0.10,
            f"update #{frame.update_count}   ·   status = {frame.status}   ·   "
            f"p_out = {frame.p_out:.3f}   ·   frames flown = {n_flown}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9)


def _frame_duration_ms(frame: PlannerFrame, is_last: bool) -> int:
    """Pace the GIF: linger on the prior + the climax, brisk through the sweep."""
    if is_last:
        return 2800
    if frame.update_count == 0:
        return 1500
    if frame.located_now or "CONTACT" in frame.caption:
        return 650
    return 200


def build_animation(result: PlannerResult, out_path: pathlib.Path) -> None:
    """Render the recorded closed loop to an animated GIF (PIL frames + per-frame durations)."""
    hill = hillshade(result.grid)
    fig, ax = plt.subplots(figsize=(8.0, 8.5), dpi=90)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12)
    images: List[Image.Image] = []
    durations: List[int] = []
    for i, frame in enumerate(result.frames):
        render_planner_frame(ax, result, hill, frame)
        images.append(_frame_to_image(fig))
        durations.append(_frame_duration_ms(frame, is_last=(i == len(result.frames) - 1)))
    plt.close(fig)
    images[0].save(str(out_path), save_all=True, append_images=images[1:],
                   duration=durations, loop=0, disposal=2, optimize=False)


def _print_summary(result: PlannerResult) -> None:
    """Console trace: sectors swept, the locate outcome, and a sample broadcast."""
    n_obs = sum(len(d.path) for d in result.frames[-1].drones) if result.frames else 0
    print(f"Closed-loop search ({result.n_drones} drone[s]): {n_obs} frames flown.  "
          f"subject (truth) = {result.subject}")
    if result.located_event is None:
        print("[!] Did not locate within the budget.")
        return
    ev = result.located_event
    err = math.hypot(ev.cell[0] - result.subject[0], ev.cell[1] - result.subject[1])
    lc = ev.terrain_context.get("land_cover", "the area")
    print(f"LOCATED at {ev.cell}  (off by {err:.1f} cells / {err * result.cfg.cell_size_m:.0f} m)")
    print(f'  [broadcast] "We have located a person near {lc}. Stay where you are — help is on the way."')


def main(seed: int = 0, n_drones: int = 1) -> None:
    """
    Run the closed-loop search (single or multi-drone) and write the GIF.

    Args:
        seed: Detector-sim seed.
        n_drones: 1 = single-drone demo; >1 = the multi-drone coordination demo.

    Why:
        One command (`python -m src.demo.search_demo [--drones N]`) produces the C1 artifact:
        a planner-directed search that finds the subject, single- or multi-drone, on real terrain.
    """
    _OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        result = (run_multi_drone(seed=seed, n_drones=n_drones) if n_drones > 1
                  else run_single_drone(seed=seed))
    except FileNotFoundError as exc:
        print(f"[!] {exc}\n[!] The closed-loop demo needs the DEM + WorldCover rasters in data/terrain/.")
        return
    _print_summary(result)
    out = _OUTPUT_DIR / (f"search_demo_{n_drones}drones.gif" if n_drones > 1 else "search_demo.gif")
    build_animation(result, out)
    print(f"\nGIF written to: {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="C1 closed-loop sector search demo.")
    parser.add_argument("--drones", type=int, default=1, help="number of coordinated drones")
    parser.add_argument("--seed", type=int, default=0, help="detector-sim seed")
    args = parser.parse_args()
    main(seed=args.seed, n_drones=args.drones)
