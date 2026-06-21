# =============================================================================
# integration/loop.py
# -----------------------------------------------------------------------------
# Responsible for: The integration orchestrator — driving the real closed loop
#                  (detector backend -> GeoReferencer -> SearchBrain) one frame at a time
#                  and exposing the published MapState + any LocatedEvent.
# Role in project: The convergence point of Milestone 1. Mirrors src/demo/run.py's wiring
#                  but (a) swaps the hard-coded simulator for a pluggable DetectorBackend
#                  (simulator OR real YOLO), and (b) is STEPPABLE so the FastAPI server can
#                  advance it on a timer and hand snapshots to the dashboard. No matplotlib /
#                  no torch at import time, so the server can import this cheaply.
# Assumptions: Single writer — exactly one SearchBrain, stepped from one place. The demo
#              tuning (located_concentration_ratio=3.5) matches run.py so the simulator path
#              locates identically. GeoReferencer + SearchBrain + the scenario all share one
#              GridSpec built from the same config (the shared-backbone invariant).
# Run: .venv/bin/python -m integration.loop                 (simulator backend; locates)
#      .venv/bin/python -m integration.loop --video V --weights W   (real YOLO backend)
# =============================================================================

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

from src.common.config import BrainConfig
from src.common.contracts import LocatedEvent, MapState
from src.common.grid import GridSpec
from src.geo.georeferencer import GeoReferencer
from src.search.brain import SearchBrain
from src.search.terrain import SyntheticTerrain, TerrainProvider
from src.search.terrain_raster import RasterTerrain
from src.demo.detector_sim import DetectorSimulator

from integration.backends import DetectorBackend, SimulatorBackend, YoloBackend
from integration.telemetry import VideoFrameSource, build_pose_feed

# Same calibration as src/demo/run.py: the real geo footprints clear less than a hand-built
# mock, so the subject concentrates to ~4.6-5.1x while the diffuse prior sits ~2.5x. A modest
# concentration FLOOR of 3.5 sits between them, with PERSISTENCE carrying the confirmation.
# Defined here (one value) rather than importing run.py, which would drag in matplotlib.
DEMO_CONFIG = BrainConfig(located_concentration_ratio=3.5)


def _make_terrain(cfg: BrainConfig, use_real_terrain: bool) -> Tuple[TerrainProvider, str]:
    """
    Build the terrain provider, falling back to synthetic if the real rasters are absent.

    Args:
        cfg: Brain config (raster paths, grid geometry).
        use_real_terrain: If True, try the real DEM + WorldCover rasters first.

    Returns:
        (terrain, name): the provider and a short human label for logging.

    Why:
        The integration loop should run anywhere — including a machine without the rasters —
        so it degrades to synthetic terrain rather than crashing, exactly like run.py.
    """
    if use_real_terrain:
        try:
            return RasterTerrain(cfg), "REAL rasters (DEM + WorldCover)"
        except FileNotFoundError:
            return SyntheticTerrain(cfg), "synthetic (raster fallback)"
    return SyntheticTerrain(cfg), "synthetic"


class SearchLoop:
    """
    A steppable closed loop: one DetectorBackend feeding one SearchBrain through geo.

    Why:
        Holding the loop as an object (not a single run() function) is what lets the server
        advance it frame-by-frame on a timer while a consumer polls MapState between steps —
        the steppable shape is the seam to the live dashboard. The brain inside is the single
        writer; every consumer reads the snapshots step() publishes.
    """

    def __init__(
        self,
        cfg: BrainConfig,
        terrain: TerrainProvider,
        backend: DetectorBackend,
        frames: list,
        subject_latlon: Tuple[float, float],
    ) -> None:
        """
        Args:
            cfg: All tunables (also fixes the shared GridSpec).
            terrain: TerrainProvider for the prior + the GeoReferencer's visibility layer.
            backend: The detection source (SimulatorBackend or YoloBackend) — the loop only
                depends on the DetectorBackend interface, so neither leaks in here.
            frames: The ordered ScriptedFrames (pose + sensor + timestamp) to play through.
            subject_latlon: Ground-truth subject (lat, lon); kept for reporting/eval only —
                the brain never sees it.

        Why:
            Everything spatial is pinned to one GridSpec built from cfg (the shared-backbone
            invariant), and the GeoReferencer's visibility comes from the SAME terrain the
            brain's prior does, so coverage clearing and the prior agree on the ground truth.
        """
        self.cfg = cfg
        self.grid: GridSpec = GridSpec.from_config(cfg)
        self.brain = SearchBrain(cfg, terrain)
        # Same terrain -> same visibility the prior used; geo clears what the prior weighted.
        self._geo = GeoReferencer(self.grid, visibility=terrain.visibility(self.grid), config=cfg)
        self._backend = backend
        self._frames = frames
        self.subject_latlon = subject_latlon
        self.subject_cell = self.grid.latlon_to_cell(*subject_latlon)

        self._i = 0
        self.drone_path: List[Tuple[int, int]] = []
        self.located_event: Optional[LocatedEvent] = None

    @property
    def n_frames(self) -> int:
        """Total scripted frames. Why: lets the server show progress (i / n)."""
        return len(self._frames)

    @property
    def frame_index(self) -> int:
        """The next frame index to play. Why: progress + done-detection for the server."""
        return self._i

    @property
    def is_done(self) -> bool:
        """True once every frame has been played. Why: stop the server's step timer."""
        return self._i >= len(self._frames)

    @property
    def last_pose(self):
        """
        The CameraPose of the most recently played frame, or None before the first step.

        Why:
            The dashboard's live telemetry (altitude/heading) comes from the current pose,
            which the brain's MapState doesn't carry. Exposing it here lets the server build
            the ProjectionContext without re-deriving the flight path it already played.
        """
        if self._i == 0:
            return None
        return self._frames[self._i - 1].pose

    def step(self) -> Optional[Tuple[MapState, Optional[LocatedEvent]]]:
        """
        Advance the loop by exactly one frame.

        Returns:
            (map_state, event): the freshly published MapState snapshot and the LocatedEvent
            if this step tripped the trigger (else None for the event). Returns None overall
            when the loop is already done.

        Why:
            One frame per call is the unit the server times against and the unit the brain's
            single-writer update is atomic over. The detector->geo->brain order is fixed here
            so a caller can't wire it wrong, exactly as run.py does it.
        """
        if self.is_done:
            return None
        frame = self._frames[self._i]
        # detector (pixel-space) -> geo (ground footprint + ground detections) -> brain.
        det_out = self._backend.detect(self._i, frame)
        obs = self._geo.reference(det_out, frame.pose)
        event = self.brain.step(obs)

        self.drone_path.append(self.grid.latlon_to_cell(*frame.pose.drone_latlon))
        if event is not None and self.located_event is None:
            self.located_event = event
        self._i += 1
        return self.brain.map_state(), event

    def run(self) -> Tuple[MapState, Optional[LocatedEvent]]:
        """
        Play the loop to completion.

        Returns:
            (final_map_state, located_event_or_None).

        Why:
            The batch entry point for the CLI and tests; the server uses step() instead so it
            can publish intermediate snapshots. Both share the exact same per-frame logic.
        """
        while not self.is_done:
            self.step()
        return self.brain.map_state(), self.located_event


def build_simulator_loop(
    cfg: BrainConfig = DEMO_CONFIG,
    *,
    use_real_terrain: bool = False,
    offset: Optional[Tuple[int, int]] = None,
    seed: int = 0,
    false_positive_rate: float = 0.01,
) -> Tuple[SearchLoop, str]:
    """
    Build a SearchLoop driven by the geometry-based DetectorSimulator (no footage).

    Args:
        cfg: Tunables (defaults to the demo config that locates).
        use_real_terrain: Use the real rasters if present (else synthetic).
        offset: Subject offset from the LKP (None -> synthetic default).
        seed: Simulator RNG seed (deterministic runs).
        false_positive_rate: Per-frame spurious-detection chance (matches run.py's 0.01).

    Returns:
        (loop, terrain_name): the ready-to-run loop and a label for logging.

    Why:
        This is the demo path the dashboard renders TODAY. It builds the simulator from the
        same scenario (grid, planted subject, visibility) the brain is built from, so the
        simulator's planted subject and the loop's subject_cell are guaranteed the same point.
    """
    grid = GridSpec.from_config(cfg)
    terrain, terrain_name = _make_terrain(cfg, use_real_terrain)
    subject_latlon, frames = build_pose_feed(grid, cfg, offset)
    visibility = terrain.visibility(grid)
    sim = DetectorSimulator(
        grid, subject_latlon, visibility, config=cfg, seed=seed, false_positive_rate=false_positive_rate
    )
    loop = SearchLoop(cfg, terrain, SimulatorBackend(sim), frames, subject_latlon)
    return loop, terrain_name


def build_yolo_loop(
    video_path: str,
    weights: str,
    *,
    cfg: BrainConfig = DEMO_CONFIG,
    use_real_terrain: bool = False,
    offset: Optional[Tuple[int, int]] = None,
    model_id: Optional[str] = None,
    conf: float = 0.25,
    imgsz: int = 960,
) -> Tuple[SearchLoop, str]:
    """
    Build a SearchLoop driven by the real YOLO detector on a video file.

    Args:
        video_path: Footage to run the detector on.
        weights: YOLO weights (yolo11n.pt pretrained today; best.pt after the swap).
        cfg: Tunables.
        use_real_terrain: Use the real rasters if present.
        offset: Subject offset from the LKP (for eval/labeling; the brain never sees it).
        model_id: Provenance string (defaults to the weights filename stem).
        conf: YOLO confidence threshold.
        imgsz: YOLO inference image size.

    Returns:
        (loop, terrain_name).

    Why:
        The production path. It is a STRICT backend swap of build_simulator_loop — same
        scenario, same brain, same geo — proving the source-agnostic seam: only the detection
        source changes. Imports the adapter lazily (inside) so the simulator path never needs
        torch.
    """
    import pathlib

    from integration import detector_adapter  # lazy: only the real path pulls torch

    grid = GridSpec.from_config(cfg)
    terrain, terrain_name = _make_terrain(cfg, use_real_terrain)
    subject_latlon, frames = build_pose_feed(grid, cfg, offset)

    model = detector_adapter.load_detector(weights)
    video = VideoFrameSource(video_path)
    resolved_model_id = model_id or f"{pathlib.Path(weights).stem}"
    backend = YoloBackend(model, video, model_id=resolved_model_id, conf=conf, imgsz=imgsz)
    loop = SearchLoop(cfg, terrain, backend, frames, subject_latlon)
    return loop, terrain_name


def main() -> None:
    """
    CLI entry: run the closed loop and print a compact trace + the locate outcome.

    Why:
        One command to prove the integration loop end-to-end. With no args it runs the
        simulator backend (and should locate); --video/--weights selects the real detector.
    """
    parser = argparse.ArgumentParser(description="Run the SAR integration loop (detector -> geo -> brain).")
    parser.add_argument("--real-terrain", action="store_true", help="use the real DEM/WorldCover rasters if present")
    parser.add_argument("--video", default=None, help="footage for the real YOLO backend (omit -> simulator)")
    parser.add_argument("--weights", default=None, help="YOLO weights for the real backend")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold (real backend)")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size (real backend)")
    args = parser.parse_args()

    if args.video is not None:
        if args.weights is None:
            parser.error("--video requires --weights")
        loop, terrain_name = build_yolo_loop(
            args.video, args.weights, use_real_terrain=args.real_terrain, conf=args.conf, imgsz=args.imgsz
        )
        backend_name = f"YOLO ({args.weights})"
    else:
        loop, terrain_name = build_simulator_loop(use_real_terrain=args.real_terrain)
        backend_name = "DetectorSimulator"

    print(
        f"Integration loop | grid {loop.grid.n_rows}x{loop.grid.n_cols} @ {loop.cfg.cell_size_m:.0f} m | "
        f"subject {loop.subject_cell} | {loop.n_frames} frames\n"
        f"terrain: {terrain_name} | detection source: {backend_name}\n"
    )

    while not loop.is_done:
        ms, event = loop.step()
        if event is not None:
            print(
                f"  frame {loop.frame_index:>2}/{loop.n_frames}: LOCATED at {event.cell} "
                f"(confidence {event.confidence:.4f})"
            )

    final = loop.brain.map_state()
    print(
        f"\nfinal: status={final.status.value} update_count={final.update_count} "
        f"p_out={final.p_out:.3f} top_cell={final.top_cells[0][0]}"
    )
    if loop.located_event is not None:
        ev = loop.located_event
        d = ((ev.cell[0] - loop.subject_cell[0]) ** 2 + (ev.cell[1] - loop.subject_cell[1]) ** 2) ** 0.5
        print(
            f"LOCATED {ev.cell} vs true {loop.subject_cell} -> off by {d:.1f} cells "
            f"({d * loop.cfg.cell_size_m:.0f} m) | latlon ({ev.latlon[0]:.5f}, {ev.latlon[1]:.5f})"
        )
    else:
        print("Loop finished WITHOUT locating.")


if __name__ == "__main__":
    main()
