# =============================================================================
# run.py
# -----------------------------------------------------------------------------
# Responsible for: Driving the brain end-to-end on the mock Observation stream and
#                  showing the map evolution — console beats plus posterior/coverage
#                  heatmap PNGs at each beat (prior -> swept -> detection -> located).
# Role in project: The Tier-1 milestone made visible: a working probability map
#                  that mock observations move, with a persistent detection flipping
#                  status to `located`. Replace mock_stream with the real
#                  GeoReferencer and this runner is unchanged.
# Run: .venv/bin/python -m src.demo.run
# Assumptions: P_located is tuned for the 160x160 demo grid (a fixed window holds a
#              small absolute share of a large grid — see DEMO_CONFIG below).
# =============================================================================

from __future__ import annotations

import pathlib
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: render PNGs without a display server
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from src.common.config import BrainConfig
from src.common.contracts import LocatedEvent
from src.common.grid import GridSpec
from src.search.brain import SearchBrain
from src.search.terrain import SyntheticTerrain
from src.search.trigger import concentration_ratio, windowed_mass
from src.demo.mock_stream import build_demo_stream

# The demo relies on the RELATIVE (concentration) trigger gate, not a brittle absolute
# mass threshold. On this 160x160 grid a fixed window holds only ~0.009 of total mass,
# so the absolute p_located stays at its default (effectively off here) and the
# grid-invariant concentration ratio — which climbs ~2.5x (prior) -> ~5x (color) ->
# ~10x (thermal) — is what fires `located` at the thermal corroboration.
DEMO_CONFIG = BrainConfig()

_OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "demo_output"


def _footprint_center(observation) -> Optional[tuple]:
    """
    The centroid (row, col) of an Observation's footprint — the drone's position.

    Args:
        observation: One frame's Observation.

    Returns:
        (row, col) mean of the footprint cells, or None if the frame covered nothing.

    Why:
        The flown-path overlay needs one point per frame. The footprint centroid is a
        faithful stand-in for where the camera was looking, without needing the real
        CameraPose (which the GeoReferencer will supply later).
    """
    cells = [cc.cell for cc in observation.footprint]
    if not cells:
        return None
    rows = [c[0] for c in cells]
    cols = [c[1] for c in cells]
    return (sum(rows) / len(rows), sum(cols) / len(cols))


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


def main() -> None:
    """
    Run the full demo loop and write the beat PNGs.

    Why:
        One command (`python -m src.demo.run`) that exercises prior -> sweep ->
        detection -> located on the real demo grid, the proof the core loop works.
    """
    _OUTPUT_DIR.mkdir(exist_ok=True)
    cfg = DEMO_CONFIG
    grid = GridSpec.from_config(cfg)
    terrain = SyntheticTerrain(cfg)
    brain = SearchBrain(cfg, terrain)
    subject, stream = build_demo_stream(grid, cfg)
    n_sweep = sum(1 for o in stream if not o.detections_ground)

    print(f"Demo grid {grid.n_rows}x{grid.n_cols} @ {cfg.cell_size_m:.0f} m | "
          f"LKP cell {grid.latlon_to_cell(*cfg.lkp_latlon)} | subject {subject} | "
          f"{len(stream)} frames ({n_sweep} sweep)\n")

    # The drone's flown path, accumulated frame by frame for the path overlay.
    drone_path: list = []

    # Beat 0 — the prior the judge sees first (nothing flown yet).
    _print_beat(brain, subject, "prior (t0)")
    render_state(brain, subject, "prior (t0)", _OUTPUT_DIR / "beat0_prior.png", drone_path=list(drone_path))

    located_event: Optional[LocatedEvent] = None
    first_detection_rendered = False

    for i, obs in enumerate(stream):
        event = brain.step(obs)
        center = _footprint_center(obs)
        if center is not None:
            drone_path.append(center)
        is_last_sweep = i == n_sweep - 1
        has_detection = bool(obs.detections_ground)

        if is_last_sweep:
            _print_beat(brain, subject, "after sweep")
            render_state(brain, subject, "after sweep", _OUTPUT_DIR / "beat1_swept.png", drone_path=list(drone_path))
        if has_detection and not first_detection_rendered:
            _print_beat(brain, subject, "first detection")
            render_state(brain, subject, "first detection (color)", _OUTPUT_DIR / "beat2_detection.png", drone_path=list(drone_path))
            first_detection_rendered = True
        if event is not None and located_event is None:
            located_event = event
            _print_beat(brain, subject, "LOCATED")
            render_state(brain, subject, "LOCATED", _OUTPUT_DIR / "beat3_located.png", drone_path=list(drone_path))

    if located_event is not None:
        _print_located(located_event)
    else:
        print("\n[!] Loop finished WITHOUT locating — check P_located tuning.\n")

    print(f"PNGs written to: {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
