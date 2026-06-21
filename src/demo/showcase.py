# =============================================================================
# showcase.py
# -----------------------------------------------------------------------------
# Responsible for: The REAL-TERRAIN animated showcase of the brain track — drive the
#                  closed loop on real Marin terrain (DEM + WorldCover) and render an
#                  animated GIF of the probability map evolving prior -> sweep ->
#                  detection -> LOCATED, overlaid on a DEM hillshade.
# Role in project: A standalone visual artifact for showing a mentor/reviewer "what
#                  this track does" (distinct from the teammate's production dashboard).
#                  Reuses the exact loop wiring as src/demo/run.py; the only differences
#                  are real terrain, the NW subject placement, the thermal floor, and a
#                  single-panel animated render on a hillshade backdrop.
# Run: PYTHONPATH=. .venv/bin/python -m src.demo.showcase
# Assumptions: the real rasters exist in data/terrain/ (errors clearly if absent).
#              The detector is SIMULATED; geo + the brain are the real code.
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pathlib

import matplotlib

matplotlib.use("Agg")  # headless: render frames without a display server
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from PIL import Image

from src.common.config import BrainConfig
from src.common.contracts import LocatedEvent, SensorType
from src.common.grid import GridSpec
from src.geo.georeferencer import GeoReferencer, _pixel_to_ground_local
from src.search.brain import SearchBrain
from src.search.terrain_raster import RasterTerrain, sample_raster_to_grid, _DEFAULT_DEM
from src.search.trigger import concentration_ratio
from src.demo.detector_sim import DetectorSimulator
from src.demo.mock_stream import REAL_TERRAIN_SUBJECT_OFFSET, build_scripted_path
from src.demo.run import _annotate_axis  # reuse the markers (region, LKP, subject, path, target)

# The real-terrain scenario tuning (the one place these live, so the animation and the
# acceptance test agree — DRY):
#   - located_concentration_ratio = 3.5: above the real subject's prior baseline (~2.6x,
#     it sits in the NW high-prior zone) so the gate discriminates, and well below the
#     thermal-corroborated find peak (~5.3-6.8x across seeds), so it locates with margin.
#     Persistence (n=3) is the false-positive guard. (Calibrated empirically on the real
#     rasters; same value as run.py's synthetic DEMO_CONFIG, which is a coincidence — the
#     baselines differ, ~1.4x synthetic vs ~2.6x real, but the thermal-floored find clears
#     3.5 comfortably either way.)
#   - thermal_detection_floor = 0.8: under ~72% canopy a forested subject is marginal for
#     color; the floor encodes the dusk premise that heat penetrates canopy gaps, so the
#     thermal pass reliably corroborates. (See docs/brain_followups.md B5.)
SHOWCASE_CONFIG = BrainConfig(located_concentration_ratio=3.5, thermal_detection_floor=0.8)

# A detection landing within this many meters of the planted subject counts as a real
# subject hit (vs a false positive) — matches the classification in src/demo/run.py.
_SUBJECT_HIT_RADIUS_M = 150.0


@dataclass(frozen=True)
class FrameState:
    """
    One animation frame's worth of loop state — presentation-free raw data.

    Args (fields):
        update_count: The brain's update index after this frame (0 = the prior, t0).
        posterior: A COPY of the posterior at this frame (n_rows, n_cols).
        drone_path: Ordered [(row, col), ...] of frames flown up to and including this one.
        next_target: The cell the map directs the search toward (or None).
        status: The brain status string ("searching" / "located").
        p_out: Reserved out-of-region mass at this frame.
        conc_at_subj: Concentration ratio in the window around the true subject (a stat).
        phase: "prior" | "sweep" | "overflight-color" | "overflight-thermal".
        subject_detected: Whether THIS frame produced a true subject detection.
        located_now: Whether the located trigger fired on THIS frame.

    Why:
        drive_loop() captures these per frame so the renderer (and the animation) is a pure
        function of recorded state — no re-running the loop while drawing, and the loop stays
        free of any matplotlib/caption concerns (separation of orchestration from rendering).
    """

    update_count: int
    posterior: np.ndarray
    drone_path: List[Tuple[int, int]]
    next_target: Optional[Tuple[int, int]]
    status: str
    p_out: float
    conc_at_subj: float
    phase: str
    subject_detected: bool
    located_now: bool


@dataclass(frozen=True)
class LoopResult:
    """
    Everything the animation (and the acceptance test) needs from one loop run.

    Args (fields):
        cfg: The brain config the run used (for the LKP marker + threshold provenance).
        grid: The shared GridSpec.
        terrain: The RasterTerrain (for the hillshade + located context).
        subject: The planted subject's true cell (row, col).
        frames: The per-frame FrameStates, frames[0] = the prior (t0).
        located_event: The LocatedEvent if the loop located, else None.

    Why:
        A single return value keeps the orchestrator's contract small and lets both the
        renderer and the test consume the same recorded run. Carrying cfg means the render
        can never disagree with the loop about the LKP or the tuning.
    """

    cfg: BrainConfig
    grid: GridSpec
    terrain: RasterTerrain
    subject: Tuple[int, int]
    frames: List[FrameState]
    located_event: Optional[LocatedEvent]


def drive_loop(
    cfg: BrainConfig = SHOWCASE_CONFIG,
    seed: int = 0,
    false_positive_rate: float = 0.01,
) -> LoopResult:
    """
    Run the real-terrain closed loop once and record a FrameState per observation.

    Args:
        cfg: The brain config (defaults to SHOWCASE_CONFIG — real-terrain tuning).
        seed: Detector-simulator RNG seed (the run is deterministic per seed).
        false_positive_rate: Per-frame spurious-detection chance for the simulator.

    Returns:
        A LoopResult with the recorded frames and the located outcome.

    Why:
        This is the single orchestrator both the GIF render and the acceptance test use,
        so "does it locate?" and "what do we animate?" are answered by the exact same run
        (no drift between the tested loop and the shown loop). Mirrors run.py's wiring:
        scripted pose -> simulated detector -> real GeoReferencer -> real brain.
    """
    grid = GridSpec.from_config(cfg)
    terrain = RasterTerrain(cfg)
    brain = SearchBrain(cfg, terrain)

    subject_latlon, frames = build_scripted_path(grid, cfg, offset=REAL_TERRAIN_SUBJECT_OFFSET)
    subject = grid.latlon_to_cell(*subject_latlon)
    subj_e, subj_n = grid.latlon_to_local_m(*subject_latlon)
    visibility = terrain.visibility(grid)
    geo = GeoReferencer(grid, visibility=visibility, config=cfg)
    sim = DetectorSimulator(grid, subject_latlon, visibility, config=cfg, seed=seed,
                            false_positive_rate=false_positive_rate)

    # Which frames are the overflight (drone directly over the subject) vs the sweep —
    # used only to label each frame's phase for the caption. Same logic as run.py.
    drone_cells = [grid.latlon_to_cell(*f.pose.drone_latlon) for f in frames]
    n_sweep = next((i for i, dc in enumerate(drone_cells) if dc == subject), len(frames))
    hw = cfg.window_halfwidth_cells

    def _state(update_count, drone_path, phase, subject_detected, located_now) -> FrameState:
        return FrameState(
            update_count=update_count,
            posterior=brain.posterior.copy(),
            drone_path=list(drone_path),
            next_target=brain.map_state().next_target,
            status=brain.status.value,
            p_out=brain.p_out,
            conc_at_subj=concentration_ratio(brain.posterior, subject, hw),
            phase=phase,
            subject_detected=subject_detected,
            located_now=located_now,
        )

    states: List[FrameState] = [_state(0, [], "prior", False, False)]
    drone_path: List[Tuple[int, int]] = []
    located_event: Optional[LocatedEvent] = None

    for i, frame in enumerate(frames):
        det_out = sim.simulate(frame.pose, frame.sensor_type, frame.timestamp)
        obs = geo.reference(det_out, frame.pose)
        event = brain.step(obs)
        drone_path.append(drone_cells[i])

        # Classify by true geo placement error (meters): a detection within the hit radius
        # of the true subject is a real hit; anything else is a false positive.
        subject_detected = False
        for det in det_out.detections:
            x, y, w, h = det.bbox_xywh
            ge, gn = _pixel_to_ground_local(x + w / 2.0, y + h / 2.0, frame.pose, grid)
            if math.hypot(ge - subj_e, gn - subj_n) <= _SUBJECT_HIT_RADIUS_M:
                subject_detected = True

        located_now = event is not None and located_event is None
        if located_now:
            located_event = event

        if i < n_sweep:
            phase = "sweep"
        elif frame.sensor_type == SensorType.THERMAL:
            phase = "overflight-thermal"
        else:
            phase = "overflight-color"
        states.append(_state(brain.update_count, drone_path, phase, subject_detected, located_now))

    return LoopResult(cfg=cfg, grid=grid, terrain=terrain, subject=subject,
                      frames=states, located_event=located_event)


# =============================================================================
# Hillshade basemap (DEM shaded relief) — the zero-dependency backdrop.
# =============================================================================

def _hillshade_from_dem(
    dem: np.ndarray,
    cell_size_m: float,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
) -> np.ndarray:
    """
    Lambertian shaded relief from an elevation grid (the standard hillshade technique).

    Args:
        dem: (n_rows, n_cols) elevation in meters (row increases north, col increases east).
        cell_size_m: Grid spacing, so the gradient is a true slope (rise/run).
        azimuth_deg: Sun compass bearing (clockwise from north); 315 = NW, the cartographic
            default that makes terrain "pop" for a north-up map.
        altitude_deg: Sun angle above the horizon.

    Returns:
        (n_rows, n_cols) shading in [0, 1]: 1 = fully lit slope, 0 = in shadow.

    Why:
        Shades each cell by how its slope faces a low sun, so ridges and the drainage the
        prior elongates down become visible — the backdrop literally *explains* the prior.
        Pure numpy (no contextily/network), deterministic, and aligned to the grid by
        construction. The dot product of the unit surface normal (-dz/de, -dz/dn, 1) with the
        light-to-sun vector is exactly Lambert's cosine law.
    """
    # Gradients along north (axis 0) and east (axis 1), in meters-rise per meter-run.
    dz_dn, dz_de = np.gradient(dem, cell_size_m)
    az, alt = math.radians(azimuth_deg), math.radians(altitude_deg)
    # Light-to-sun unit vector in (east, north, up): azimuth is clockwise from north.
    lx = math.sin(az) * math.cos(alt)
    ly = math.cos(az) * math.cos(alt)
    lz = math.sin(alt)
    # cos(incidence) = normal_unit · light, with normal ∝ (-dz_de, -dz_dn, 1).
    denom = np.sqrt(dz_de ** 2 + dz_dn ** 2 + 1.0)
    shaded = (-dz_de * lx - dz_dn * ly + lz) / denom
    return np.clip(shaded, 0.0, 1.0)


def hillshade(grid: GridSpec) -> np.ndarray:
    """
    The DEM hillshade sampled onto the grid (same cell frame as the posterior).

    Args:
        grid: The shared GridSpec.

    Returns:
        (n_rows, n_cols) shaded relief in [0, 1], ready to imshow under the posterior.

    Why:
        Because the DEM is sampled ONTO the grid, the hillshade is a (n_rows, n_cols) array
        in the same cell coordinates as the posterior — so the overlay is two imshow calls
        with no geographic extent or reprojection (that complexity is only contextily's).
        Off-DEM cells (the AOI's east edge) are nan; filled with the median so the gradient
        is defined (they render as flat mid-grey, which is honest: we don't know the relief
        there).
    """
    dem = sample_raster_to_grid(str(_DEFAULT_DEM), grid)
    dem = np.where(np.isnan(dem), np.nanmedian(dem), dem)
    return _hillshade_from_dem(dem, grid.cell_size_m)


# =============================================================================
# Frame rendering — one panel: hillshade + posterior (graded alpha) + markers.
# =============================================================================

def _caption(fs: "FrameState") -> str:
    """
    The human-readable beat caption for a frame (the story the animation tells).

    Args:
        fs: The frame state (its phase / detection / located flags).

    Returns:
        A short caption string for the panel title.

    Why:
        Maps the loop's raw phase to the demo beats a viewer reads — prior, sweeping,
        detection, thermal corroboration, LOCATED — keeping the narrative in one place.
    """
    if fs.status == "located":
        return "★ LOCATED — subject found in tree cover — broadcasting to the subject"
    if fs.phase == "prior":
        return "PRIOR — the map's starting hypothesis (NW high-probability zone)"
    if fs.phase == "sweep":
        return "SWEEPING — clearing the corridor (clean non-detections)"
    if fs.phase == "overflight-thermal":
        return "THERMAL PASS — corroborating the find under canopy"
    if fs.subject_detected:
        return "DETECTION — the color camera spots the subject; the map spikes"
    return "OVERFLIGHT — loitering over the high-probability slope"


def draw_belief_layer(ax, posterior: np.ndarray, hill: np.ndarray) -> None:
    """
    Draw the grey hillshade backdrop with the posterior overlaid (graded alpha) onto an axis.

    Args:
        ax: The matplotlib axis (already cleared by the caller).
        posterior: (n_rows, n_cols) probability map to overlay.
        hill: (n_rows, n_cols) hillshade backdrop in [0, 1].

    Why:
        The hillshade + log-scaled, steeply-graded-alpha posterior is the shared visual base for
        BOTH the scripted showcase and the C1 closed-loop planner demo, so it lives in one place
        (DRY). Graded alpha (power 2.5) keeps the cold tail see-through (terrain shows) and the
        hot find opaque (it pops) — a flat alpha washes this grid's broad tail.
    """
    ax.imshow(hill, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    floor = posterior[posterior > 0].min() if np.any(posterior > 0) else 1e-12
    norm = LogNorm(vmin=floor, vmax=posterior.max())
    normed = np.asarray(norm(np.maximum(posterior, floor)))      # in [0, 1]
    alpha = 0.06 + 0.90 * np.clip(normed, 0.0, 1.0) ** 2.5       # see-through cold, opaque hot
    ax.imshow(np.maximum(posterior, floor), origin="lower", cmap="inferno", norm=norm, alpha=alpha)


def render_frame(ax, cfg: BrainConfig, grid: GridSpec, subject: Tuple[int, int],
                 hill: np.ndarray, fs: "FrameState") -> None:
    """
    Draw one animation frame onto an axis: hillshade backdrop, posterior, markers, stats.

    Args:
        ax: The matplotlib axis to draw on (cleared and redrawn per frame).
        cfg: The brain config (for the LKP marker).
        grid: The shared GridSpec.
        subject: The planted subject cell (drawn as ground-truth reference).
        hill: The precomputed hillshade (n_rows, n_cols), drawn as the grey backdrop.
        fs: The FrameState to render.

    Why:
        A single function the animation calls per frame. The posterior is drawn with a
        GRADED alpha (transparent where probability is low, opaque where high) so the
        terrain shows through the cold areas and the hot cell glows over it — far more
        legible than a flat half-tint. Reuses _annotate_axis so the markers match run.py.
    """
    ax.clear()
    draw_belief_layer(ax, fs.posterior, hill)   # hillshade backdrop + posterior (shared)

    lkp = grid.latlon_to_cell(*cfg.lkp_latlon)
    _annotate_axis(ax, grid, lkp, subject, fs.drone_path, fs.next_target)

    ax.set_title(_caption(fs), fontsize=12, fontweight="bold", pad=10)
    # A compact stat strip below the panel: the same numbers the console trace shows.
    ax.text(
        0.5, -0.10,
        f"update #{fs.update_count}   ·   status = {fs.status}   ·   "
        f"p_out = {fs.p_out:.3f}   ·   concentration @ find = {fs.conc_at_subj:.1f}×",
        transform=ax.transAxes, ha="center", va="top", fontsize=9,
    )


# =============================================================================
# Animation assembly + entrypoint.
# =============================================================================

_OUTPUT_DIR = pathlib.Path(__file__).resolve().parents[2] / "demo_output"


def _frame_to_image(fig) -> Image.Image:
    """
    Snapshot the current matplotlib figure as an RGB PIL image.

    Args:
        fig: The figure to capture (already drawn).

    Returns:
        A PIL Image (RGB) of the rendered canvas.

    Why:
        Building the GIF from PIL frames (instead of matplotlib's PillowWriter) lets us set a
        PER-FRAME duration — the writer only supports one fps, and Pillow coalesces repeated
        identical frames, so the usual "repeat the last frame" hold silently collapses. Direct
        PIL frames give us an explicit hold on the climax.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return Image.frombytes("RGBA", (w, h), bytes(fig.canvas.buffer_rgba())).convert("RGB")


def _frame_duration_ms(fs: "FrameState", is_last: bool) -> int:
    """
    How long to hold each frame in the GIF (milliseconds), pacing the story.

    Args:
        fs: The frame state.
        is_last: Whether this is the final frame (the climax to dwell on).

    Returns:
        Display duration in ms for this frame.

    Why:
        A flat frame rate either rushes the payoff or drags the sweep. We linger on the prior
        (let the viewer read the starting hypothesis), move briskly through the sweep, and hold
        the LOCATED climax — the pacing a human would give when narrating the loop.
    """
    if is_last:
        return 2800             # dwell on the find
    if fs.phase == "prior":
        return 1300             # let the starting hypothesis register
    if fs.located_now or fs.subject_detected:
        return 700              # mark the detection / lock beats
    if fs.phase == "sweep":
        return 150              # brisk through the clearing sweep
    return 320                  # overflight loiter


def build_animation(result: LoopResult, out_path: pathlib.Path) -> None:
    """
    Render the recorded loop to an animated GIF (one frame per observation).

    Args:
        result: The recorded LoopResult (drives the frames).
        out_path: Where to write the .gif.

    Why:
        Captures each recorded FrameState as a PIL image (computing the hillshade ONCE since it
        never changes) and writes them as a single looping GIF with per-frame durations, so the
        prior and the LOCATED climax dwell while the sweep moves briskly. Pillow is bundled —
        zero new dependencies. disposal=2 clears each frame to the background so the full redraws
        don't ghost.
    """
    hill = hillshade(result.grid)
    fig, ax = plt.subplots(figsize=(8.0, 8.5), dpi=90)
    # Reserve margins for the title (top) and the stat strip (bottom) — fixed once, since
    # tight_layout would shift the panel between frames and make the GIF jitter.
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12)

    images: List[Image.Image] = []
    durations: List[int] = []
    for i, fs in enumerate(result.frames):
        render_frame(ax, result.cfg, result.grid, result.subject, hill, fs)
        images.append(_frame_to_image(fig))
        durations.append(_frame_duration_ms(fs, is_last=(i == len(result.frames) - 1)))
    plt.close(fig)

    images[0].save(
        str(out_path), save_all=True, append_images=images[1:],
        duration=durations, loop=0, disposal=2, optimize=False,
    )


def _write_located_still(result: LoopResult, out_path: pathlib.Path) -> None:
    """
    Write a single high-res PNG of the LOCATED frame (the money shot).

    Args:
        result: The recorded run.
        out_path: Where to write the .png.

    Why:
        A still of the climax is handy for slides/READMEs where a GIF won't embed. Uses the
        last frame (peak concentration) if the run located, else the final frame.
    """
    fig, ax = plt.subplots(figsize=(8.0, 8.5), dpi=130)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.12)
    hill = hillshade(result.grid)
    render_frame(ax, result.cfg, result.grid, result.subject, hill, result.frames[-1])
    fig.savefig(out_path)
    plt.close(fig)


def _print_summary(result: LoopResult) -> None:
    """
    Print a short console trace of the run and the located outcome.

    Args:
        result: The recorded run.

    Why:
        So `python -m src.demo.showcase` is legible in a terminal (not only via the GIF),
        and surfaces the same LocatedEvent fields the broadcast/operator would speak.
    """
    n_obs = len(result.frames) - 1  # frame 0 is the prior
    print(f"Real-terrain showcase: {result.grid.n_rows}x{result.grid.n_cols} grid, "
          f"{n_obs} observations.  subject (truth) = {result.subject}")
    if result.located_event is None:
        print("[!] Loop finished WITHOUT locating.")
        return
    ev = result.located_event
    lat, lon = ev.latlon
    err = math.hypot(ev.cell[0] - result.subject[0], ev.cell[1] - result.subject[1])
    lc = ev.terrain_context.get("land_cover", "the area")
    print(f"LOCATED at cell {ev.cell}  (off by {err:.1f} cells / "
          f"{err * result.cfg.cell_size_m:.0f} m)  latlon ({lat:.5f}, {lon:.5f})")
    print(f'  [broadcast] "We have located a person near {lc}. '
          'Stay where you are — a ground team is on the way."')


def main(seed: int = 0) -> None:
    """
    Drive the real-terrain loop and write the showcase GIF + a LOCATED still PNG.

    Args:
        seed: Detector-simulator seed (the run is deterministic per seed).

    Why:
        The one command (`python -m src.demo.showcase`) that produces the artifact this
        session is for: the search loop, on real Marin terrain, prior -> sweep -> detection
        -> LOCATED, on a hillshade backdrop. Errors clearly if the rasters are absent.
    """
    _OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        result = drive_loop(seed=seed)
    except FileNotFoundError as exc:
        print(f"[!] {exc}")
        print("[!] The showcase needs the real DEM + WorldCover rasters in data/terrain/.")
        return

    _print_summary(result)
    gif_path = _OUTPUT_DIR / "showcase.gif"
    still_path = _OUTPUT_DIR / "showcase_located.png"
    build_animation(result, gif_path)
    _write_located_still(result, still_path)
    print(f"\nGIF written to:   {gif_path}")
    print(f"Still written to: {still_path}")


if __name__ == "__main__":
    main()
