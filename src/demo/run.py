# =============================================================================
# run.py
# -----------------------------------------------------------------------------
# Responsible for: Driving the FULL chain end to end — scripted CameraPose ->
#                  detector simulator -> GeoReferencer -> brain — and showing the
#                  map evolution (console beats + posterior/coverage PNGs). Also
#                  reports geo's localization error: the feasibility result.
# Role in project: The geo+brain integration milestone. The subject's map cell now
#                  EMERGES from the projection chain rather than being hand-placed.
#                  The only simulated piece is the detector (we have no footage);
#                  geo and the brain are the real code on real geography.
# Run: .venv/bin/python -m src.demo.run
# Assumptions: synthetic terrain (real rasters are a separate unit); P_located stays
#              at default and the relative concentration gate fires `located`.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import math
import pathlib
import statistics
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless: render PNGs without a display server
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from src.common.config import BrainConfig
from src.common.contracts import LocatedEvent
from src.common.grid import GridSpec
from src.geo.georeferencer import GeoReferencer, _pixel_to_ground_local
from src.search.brain import SearchBrain
from src.search.terrain import SyntheticTerrain
from src.search.terrain_raster import RasterTerrain
from src.search.trigger import concentration_ratio, windowed_mass
from src.demo.detector_sim import DetectorSimulator
from src.demo.mock_stream import build_scripted_path

# located_concentration_ratio CALIBRATED TO THE REAL CHAIN (not the old hand-built mock).
# The feasibility harness showed the real geo footprints clear less than the fiction mock,
# so the subject concentrates to ~4.6-5.1x (prior baseline ~2.5x). PERSISTENCE (N) is the
# real false-positive discriminator — the subject gets persistence 4-8 while spurious blobs
# stay at <=1 however concentrated. So the concentration gate is a modest FLOOR (3.5) that
# sits comfortably between the diffuse prior (2.5x) and a corroborated find (~5x), with
# persistence carrying the confirmation. The clearance cap (§5.5) keeps the find from being
# eroded by canopy misses during loiter. (p_out absolute gate stays effectively off.)
DEMO_CONFIG = BrainConfig(located_concentration_ratio=3.5)

_OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "demo_output"


def _annotate_axis(ax, grid, lkp, subject, drone_path, next_target) -> None:
    """
    Draw the shared overlays on one panel: region boundary, markers, flown path, and
    the next-target redirect arrow.

    Args:
        ax: The matplotlib axis to draw on.
        grid: GridSpec (for the region extent).
        lkp: (row, col) of the last-known position.
        subject: (row, col) of the planted subject (ground truth).
        drone_path: Ordered [(row, col), ...] of frames flown so far (may be empty).
        next_target: The cell the map currently directs the search toward (or None).

    Why:
        Makes "the grid is the whole region" explicit (the lime border) and turns the
        abstract redirect into something you can see: the blue trail is where we have
        flown, the yellow arrow is where the map says to look next. (imshow x=col, y=row.)
    """
    # The search region = the entire grid. Drawn as a bright border so it's unmistakable.
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5), grid.n_cols, grid.n_rows,
            fill=False, edgecolor="lime", linewidth=2.5, clip_on=False,
            label="search region (the grid)",
        )
    )
    ax.scatter([lkp[1]], [lkp[0]], marker="^", s=90, c="cyan", edgecolors="black", label="LKP", zorder=6)
    ax.scatter([subject[1]], [subject[0]], marker="*", s=180, c="red", edgecolors="black", label="subject (truth)", zorder=6)

    # The flown path so far (the lawnmower sweep, then the move down-corridor).
    if drone_path:
        prows = [p[0] for p in drone_path]
        pcols = [p[1] for p in drone_path]
        ax.plot(pcols, prows, "-", color="deepskyblue", lw=1.3, alpha=0.9, label="flown path", zorder=4)
        ax.plot(pcols[-1], prows[-1], "o", color="deepskyblue", ms=6, markeredgecolor="black", zorder=5)

    # Where the map wants to look next — and an arrow from the drone showing the redirect.
    if next_target is not None:
        nr, nc = next_target
        ax.scatter([nc], [nr], marker="X", s=130, c="yellow", edgecolors="black", label="next target", zorder=7)
        if drone_path:
            ax.annotate(
                "", xy=(nc, nr), xytext=(drone_path[-1][1], drone_path[-1][0]),
                arrowprops=dict(arrowstyle="-|>", color="yellow", lw=2.0),
                zorder=6,
            )
    ax.set_xlabel("column (50 m cells)")
    ax.set_ylabel("row (50 m cells)  •  full frame = 8 km × 8 km")
    ax.legend(loc="upper right", fontsize=7)


def render_state(
    brain: SearchBrain,
    subject: tuple,
    label: str,
    out_path: pathlib.Path,
    drone_path: Optional[list] = None,
) -> None:
    """
    Save a two-panel PNG of the current posterior and coverage.

    Args:
        brain: The brain whose state to render.
        subject: The planted subject cell (drawn as a marker for reference).
        label: A title for this beat.
        out_path: Where to write the PNG.
        drone_path: Ordered list of flown-frame centers so far (for the path overlay).

    Why:
        The posterior is the centerpiece the judges watch; the coverage layer shows
        "where have we looked." Rendering both per beat — now with the region boundary
        and the flown path + next-target arrow — makes the map's reasoning (bloom,
        clear, redirect, spike, lock) visually verifiable before the real dashboard.
    """
    grid = brain.grid
    lkp = grid.latlon_to_cell(*DEMO_CONFIG.lkp_latlon)
    next_target = brain.map_state().next_target
    drone_path = drone_path or []
    post = brain.posterior
    fig, (ax_p, ax_c) = plt.subplots(1, 2, figsize=(13, 6))

    # Posterior on a log scale: probabilities span orders of magnitude, so a linear
    # scale would wash out everything but the single brightest cell.
    floor = post[post > 0].min() if np.any(post > 0) else 1e-12
    im_p = ax_p.imshow(
        np.maximum(post, floor),
        origin="lower",  # row 0 at bottom = south, so the map reads north-up
        cmap="inferno",
        norm=LogNorm(vmin=floor, vmax=post.max()),
    )
    ax_p.set_title(f"posterior — {label}\nstatus={brain.status.value}  p_out={brain.p_out:.3f}")
    fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)

    im_c = ax_c.imshow(brain.coverage, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_c.set_title(f"coverage (cleared) — update #{brain.update_count}")
    fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)

    for ax in (ax_p, ax_c):
        _annotate_axis(ax, grid, lkp, subject, drone_path, next_target)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _print_beat(brain: SearchBrain, subject: tuple, label: str) -> None:
    """
    Print a one-line console summary of the current map state.

    Args:
        brain: The brain to summarize.
        subject: The subject cell (to report its windowed mass).
        label: The beat label.

    Why:
        A compact textual trace of the same beats the PNGs show, so the loop's
        behavior is legible in a terminal without opening images.
    """
    ms = brain.map_state()
    hw = DEMO_CONFIG.window_halfwidth_cells
    wm = windowed_mass(brain.posterior, subject, hw)
    conc = concentration_ratio(brain.posterior, subject, hw)
    cleared = int((brain.coverage > 0.01).sum())
    top_cell, top_p = ms.top_cells[0]
    print(
        f"[{label:<16}] upd#{ms.update_count:<2} status={ms.status.value:<9} "
        f"p_out={ms.p_out:.3f}  cleared={cleared:<4} "
        f"top={top_cell} p={top_p:.5f}  windowed@subj={wm:.5f}  "
        f"conc@subj={conc:.1f}x  next_target={ms.next_target}"
    )


def _print_located(event: LocatedEvent) -> None:
    """
    Print the LocatedEvent and a sample broadcast line the voice layer would speak.

    Args:
        event: The emitted LocatedEvent.

    Why:
        Shows the event carries everything the broadcast/operator surfaces need
        (cell, lat/lon, terrain context) without reaching into the brain — and makes
        the demo's high point concrete.
    """
    lat, lon = event.latlon
    print("\n" + "=" * 72)
    print("LOCATED — LocatedEvent emitted")
    print(f"  cell           : {event.cell}")
    print(f"  latlon         : ({lat:.5f}, {lon:.5f})")
    print(f"  confidence     : {event.confidence:.5f}  (windowed posterior mass)")
    print(f"  terrain_context: {event.terrain_context}")
    lc = event.terrain_context.get("land_cover", "the area")
    print(
        f'  [sample broadcast] "We have located a person near {lc}. '
        'Stay where you are — a ground team is on the way."'
    )
    print("=" * 72 + "\n")


def _print_feasibility(subject, event, geo_errors_m, subject_hits, false_positives, overflight_frames, cfg) -> None:
    """
    Print the feasibility result: geo's localization error and the brain's locate.

    Args:
        subject: True subject cell.
        event: The LocatedEvent (or None).
        geo_errors_m: Per-subject-detection geo placement error, in meters.
        subject_hits / false_positives / overflight_frames: detection counts.
        cfg: For the cell size (cells -> meters).

    Why:
        This is the whole point of the harness: a measured, honest answer to "does the
        geo+brain pipeline locate the subject, and how much error does geo's baseline
        add?" — not a claim, a number.
    """
    print("=" * 72)
    print("FEASIBILITY — geo + brain on SIMULATED detections (real geography)")
    print(f"  true subject cell    : {subject}")
    if event is not None:
        d = math.hypot(event.cell[0] - subject[0], event.cell[1] - subject[1])
        print(f"  brain located cell   : {event.cell}  (off by {d:.1f} cells / {d * cfg.cell_size_m:.0f} m)")
    else:
        print("  brain located cell   : (none — did not locate)")
    if geo_errors_m:
        print(f"  geo placement error  : mean {statistics.mean(geo_errors_m):.0f} m, "
              f"max {max(geo_errors_m):.0f} m  (baseline ignores gimbal pitch + box jitter)")
    print(f"  subject detections   : {subject_hits} of {overflight_frames} overflight frames "
          f"(canopy misses expected)")
    print(f"  false positives      : {false_positives} (handled — did not trigger locate)")
    print("=" * 72)


def main(use_real_terrain: bool = False) -> None:
    """
    Run the full chain (scripted pose -> simulator -> geo -> brain) and write PNGs.

    Args:
        use_real_terrain: If True, build the prior + visibility from the real DEM +
            WorldCover rasters (RasterTerrain) instead of the synthetic stub. The demo's
            scenario (subject placement, flight path, threshold) is tuned for SYNTHETIC
            terrain, so on real terrain it shows the real Marin prior but may not locate
            (the region is densely forested — see docs/brain_followups.md). Falls back to
            synthetic if the rasters are absent.

    Why:
        One command (`python -m src.demo.run [--real-terrain]`) that proves the geo+brain
        integration: the subject's cell EMERGES from projection, and the feasibility
        report states geo's error and the locate outcome.
    """
    logging.basicConfig(level=logging.WARNING, format="  [warn] %(name)s: %(message)s")
    _OUTPUT_DIR.mkdir(exist_ok=True)
    cfg = DEMO_CONFIG
    grid = GridSpec.from_config(cfg)

    if use_real_terrain:
        try:
            terrain = RasterTerrain(cfg)
            terrain_name = "REAL rasters (DEM + WorldCover)"
        except FileNotFoundError as exc:
            print(f"[!] {exc}\n[!] falling back to synthetic terrain")
            terrain, terrain_name = SyntheticTerrain(cfg), "synthetic (raster fallback)"
    else:
        terrain, terrain_name = SyntheticTerrain(cfg), "synthetic"
    brain = SearchBrain(cfg, terrain)

    subject_latlon, frames = build_scripted_path(grid, cfg)
    subject = grid.latlon_to_cell(*subject_latlon)
    subj_e, subj_n = grid.latlon_to_local_m(*subject_latlon)
    visibility = terrain.visibility(grid)
    geo = GeoReferencer(grid, visibility=visibility, config=cfg)
    sim = DetectorSimulator(grid, subject_latlon, visibility, config=cfg, seed=0, false_positive_rate=0.01)

    # The first overflight frame is the first with the drone directly over the subject.
    drone_cells = [grid.latlon_to_cell(*f.pose.drone_latlon) for f in frames]
    n_sweep = next((i for i, dc in enumerate(drone_cells) if dc == subject), len(frames))

    print(f"Demo grid {grid.n_rows}x{grid.n_cols} @ {cfg.cell_size_m:.0f} m | "
          f"LKP {grid.latlon_to_cell(*cfg.lkp_latlon)} | subject {subject} | "
          f"{len(frames)} frames ({n_sweep} sweep)\n"
          f"terrain: {terrain_name}  [detector SIMULATED; geo + brain real]\n")

    drone_path: list = []
    located_event: Optional[LocatedEvent] = None
    first_detection_rendered = False
    geo_errors_m: List[float] = []
    subject_hits = 0
    false_positives = 0

    _print_beat(brain, subject, "prior (t0)")
    render_state(brain, subject, "prior (t0)", _OUTPUT_DIR / "beat0_prior.png", drone_path=list(drone_path))

    for i, frame in enumerate(frames):
        det_out = sim.simulate(frame.pose, frame.sensor_type, frame.timestamp)
        obs = geo.reference(det_out, frame.pose)
        event = brain.step(obs)
        drone_path.append(drone_cells[i])

        # Classify each detection by its true geo placement error (meters), so a subject
        # hit and a false positive are told apart by distance, not by a hidden flag.
        subj_here = False
        for det in det_out.detections:
            x, y, w, h = det.bbox_xywh
            ge, gn = _pixel_to_ground_local(x + w / 2.0, y + h / 2.0, frame.pose, grid)
            err_m = math.hypot(ge - subj_e, gn - subj_n)
            if err_m <= 150.0:      # within ~3 cells of truth -> a subject detection
                geo_errors_m.append(err_m)
                subject_hits += 1
                subj_here = True
            else:
                false_positives += 1

        if i == n_sweep - 1:
            _print_beat(brain, subject, "after sweep")
            render_state(brain, subject, "after sweep", _OUTPUT_DIR / "beat1_swept.png", drone_path=list(drone_path))
        if subj_here and not first_detection_rendered:
            _print_beat(brain, subject, "first detection")
            render_state(brain, subject, "first detection", _OUTPUT_DIR / "beat2_detection.png", drone_path=list(drone_path))
            first_detection_rendered = True
        if event is not None and located_event is None:
            located_event = event
            _print_beat(brain, subject, "LOCATED")
            render_state(brain, subject, "LOCATED", _OUTPUT_DIR / "beat3_located.png", drone_path=list(drone_path))

    print()
    _print_feasibility(subject, located_event, geo_errors_m, subject_hits,
                       false_positives, len(frames) - n_sweep, cfg)
    if located_event is not None:
        _print_located(located_event)
    else:
        print("\n[!] Loop finished WITHOUT locating.\n")

    print(f"PNGs written to: {_OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SAR geo+brain demo.")
    parser.add_argument(
        "--real-terrain", action="store_true",
        help="build the prior + visibility from the real DEM/WorldCover rasters "
             "(shows real Marin terrain; the synthetic-tuned scenario may not locate)",
    )
    args = parser.parse_args()
    main(use_real_terrain=args.real_terrain)
