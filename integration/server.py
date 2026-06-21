# =============================================================================
# integration/server.py
# -----------------------------------------------------------------------------
# Responsible for: Serving the live, projected UI-MapState to the React dashboard over
#                  HTTP, by stepping the integration loop on a background thread and
#                  publishing the latest projected snapshot.
# Role in project: The transport seam (interfaces §6.2's "cross a socket boundary later").
#                  The brain stays the single writer — exactly ONE thread steps the loop;
#                  HTTP handlers only READ the cached snapshot under a lock. This is the
#                  in-process MapState becoming a JSON endpoint without any contract change.
# Pattern: A background stepper thread (the single writer) + lock-guarded read model (the
#          published snapshot). Polling, not push — the dashboard GETs /state on a timer.
#          FastAPI is chosen over Flask because Milestone 2's voice layer wants WS/streaming,
#          which FastAPI gives for free behind this same app.
# Run: .venv/bin/uvicorn integration.server:app --reload
#      (env: SAR_STEP_INTERVAL = seconds per frame, default 0.7 — tune the demo cadence)
# =============================================================================

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from src.common.contracts import Cell, LocatedEvent
from src.search.guide import simulate_guidance
from src.search.return_path import plan_return_path
from integration.dashboard_projection import ProjectionContext, project
from integration.loop import build_showcase_loop
from integration.terrain_render import render_terrain_png

# Seconds between frames. Slow enough to WATCH the map evolve in the demo, fast enough that
# the ~46-frame run locates in well under a minute. A tunable, not a constant in the math.
_STEP_INTERVAL_S = float(os.environ.get("SAR_STEP_INTERVAL", "0.7"))

# The guidance sim emits hundreds of fine ticks; we replay this many on the dashboard so the
# guide-home phase plays out in a watchable time (a display cadence, not a sim change).
_GUIDE_FRAMES = 36

# Allowed browser origins (the Vite dev server). CORS is required because the dashboard is
# served from :5173 while this API is on :8000 — a cross-origin fetch the browser blocks by default.
_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def _located_to_dict(event: LocatedEvent) -> Dict[str, Any]:
    """
    Serialize a LocatedEvent for the /located endpoint.

    Args:
        event: The brain's LocatedEvent.

    Returns:
        A JSON-serializable dict (cell/latlon as lists; terrain_context already plain).

    Why:
        LocatedEvent is a dataclass with tuple fields; the broadcast/voice surfaces consume
        it over HTTP, so we flatten the tuples here rather than leaking dataclass internals.
    """
    return {
        "cell": list(event.cell),
        "latlon": list(event.latlon),
        "confidence": event.confidence,
        "terrain_context": event.terrain_context,
        "timestamp": event.timestamp,
    }


class LoopRunner:
    """
    Owns the integration loop, steps it on a background thread, and publishes snapshots.

    Why:
        Concentrating the single-writer brain behind one stepper thread is what makes the
        "single writer, many readers" invariant hold across the HTTP boundary: only this
        thread mutates the loop; every request just reads the last projected dict under a
        lock. The dashboard polls; it never drives the brain.
    """

    def __init__(self) -> None:
        """Build the loop and publish the t0 (prior) state. Why: /state is valid before any step."""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state_dict: Dict[str, Any] = {}
        self._located: Optional[Dict[str, Any]] = None
        self._trend: list = []
        self._terrain_png: Optional[bytes] = None
        self._build()

    def _build(self) -> None:
        """
        (Re)build the loop from scratch and project its prior state.

        Why:
            Used on construction and on /reset, so a live demo can replay the search without
            restarting the server. Trend/located are cleared because they belong to a run.
        """
        self._loop, self.terrain_name = build_showcase_loop()
        self._trend = []
        self._located = None
        # Guide-home phase state (populated once the search locates). home_cell is known from
        # the start, so the operators marker can show during the search too.
        self.guidance_status = "searching"
        self._guidance_path = None
        self._home_cell: Cell = self._loop.grid.latlon_to_cell(*self._loop.cfg.lkp_latlon)
        self._guide_states = []
        self._guide_i = 0
        self._guide_drone_cells: list = []
        # Render the terrain backdrop ONCE per run (it's static — the grid is fixed). None if
        # the rasters are absent, in which case /terrain.png 404s and the dashboard falls back
        # to its procedural backdrop. Bytes are immutable, so the endpoint reads them lock-free.
        try:
            self._terrain_png = render_terrain_png(self._loop.grid)
        except FileNotFoundError:
            self._terrain_png = None
        with self._lock:
            self._state_dict = project(self._loop.brain.map_state(), self._context())

    def _context(self) -> ProjectionContext:
        """
        Assemble the ProjectionContext from the loop's current path/pose/trend.

        Why:
            The projection is a pure function of (snapshot, context); the server is what
            accumulates the context across steps (the flown path lives in the loop, the trend
            history lives here). Called only from the stepper thread, so reading _trend is safe.
        """
        loop = self._loop
        # During guidance the drone's leading positions extend the flown path, so dronePos
        # follows the leader and flightPath shows it flying the route home.
        subject_cell = None
        if self.guidance_status in ("guiding", "arrived") and self._guide_states:
            subject_cell = self._guide_states[self._guide_i].subject_cell
        return ProjectionContext(
            drone_path=list(loop.drone_path) + self._guide_drone_cells,
            last_pose=loop.last_pose,
            trend=list(self._trend),
            located_cell=loop.located_event.cell if loop.located_event else None,
            started_at="live",
            guidance_path=self._guidance_path,
            subject_cell=subject_cell,
            home_cell=self._home_cell,
            guidance_status=self.guidance_status,
        )

    def start(self) -> None:
        """Spawn the background stepper. Why: begins playing the loop when the server boots."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sar-loop-stepper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the stepper to halt and join it. Why: clean shutdown on server exit/reset."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        """
        The single-writer loop: advance ONE phase-step, then wait — until the run is finished.

        Why:
            This is the ONLY place state advances. It runs two phases in sequence: the SEARCH
            loop (detector->geo->brain) until it's done, then — if it located — the GUIDE-HOME
            phase (drone leads the subject home). One thread, one writer, across both phases.
        """
        while not self._stop.is_set():
            self._advance()
            # Wait on the stop event (not time.sleep) so shutdown is responsive mid-interval.
            self._stop.wait(_STEP_INTERVAL_S)

    def _advance(self) -> None:
        """
        Advance the run by one step of the current phase (search, then guide-home).

        Why:
            Splitting the per-tick logic out of the wait loop keeps each phase legible: step the
            search until done; on the first done-and-located tick build the route + guidance;
            then replay the guidance frames until the subject arrives; then idle.
        """
        loop = self._loop
        if not loop.is_done:
            result = loop.step()                       # SEARCH phase
            if result is not None:
                map_state, event = result
                ui = project(map_state, self._context())
                with self._lock:
                    self._state_dict = ui
                    self._trend.append((map_state.timestamp, ui["confidenceToDeclare"]))
                    if event is not None and self._located is None:
                        self._located = _located_to_dict(event)
        elif self.guidance_status == "searching" and loop.located_event is not None:
            self._build_guidance()                     # transition: plan the route home
            self._publish_guidance()
        elif self.guidance_status == "guiding":         # GUIDE-HOME phase
            if self._guide_i < len(self._guide_states) - 1:
                self._guide_i += 1
            else:
                self.guidance_status = "arrived"
            self._publish_guidance()
        # else: arrived, or never located -> nothing left to advance (idle).

    def _build_guidance(self) -> None:
        """
        Plan the terrain-aware route home and simulate the guidance, ready to replay.

        Why:
            Built once at the search->guide transition: the route the located subject walks back
            to the operators, and the leader/follower positions per tick. Down-sampled to
            _GUIDE_FRAMES so the dashboard plays it out in a watchable time.
        """
        loop = self._loop
        subject = loop.subject_cell
        path = plan_return_path(loop.grid, loop.terrain, subject, self._home_cell, cfg=loop.cfg)
        guidance = simulate_guidance(loop.grid, loop.terrain, path)
        states = guidance.states
        if len(states) > _GUIDE_FRAMES:
            stride = max(1, len(states) // _GUIDE_FRAMES)
            sampled = states[::stride]
            if sampled[-1] is not states[-1]:
                sampled.append(states[-1])
            states = sampled
        self._guidance_path = path
        self._guide_states = states
        self._guide_i = 0
        self._guide_drone_cells = []
        self.guidance_status = "guiding"

    def _publish_guidance(self) -> None:
        """
        Publish the projection for the current guidance frame (drone leading, subject following).

        Why:
            Appends the drone's current lead position to the flown path (so dronePos follows the
            leader and flightPath shows it flying the route), then republishes under the lock.
        """
        st = self._guide_states[self._guide_i]
        self._guide_drone_cells.append(st.drone_cell)
        with self._lock:
            self._state_dict = project(self._loop.brain.map_state(), self._context())

    # --- read model (lock-guarded; called from request handlers) ---

    def state(self) -> Dict[str, Any]:
        """The latest projected UI-MapState. Why: the dashboard's /state poll target."""
        with self._lock:
            return self._state_dict

    def located(self) -> Optional[Dict[str, Any]]:
        """The latched LocatedEvent dict, or None. Why: drives the broadcast/notify surface."""
        with self._lock:
            return self._located

    def terrain_png(self) -> Optional[bytes]:
        """The cached terrain backdrop PNG, or None if rasters were absent. Why: served by /terrain.png."""
        return self._terrain_png

    def health(self) -> Dict[str, Any]:
        """Run progress for /health and the reset response. Why: lets the UI show frame i/n."""
        loop = self._loop
        return {
            "status": "ok",
            "frame": loop.frame_index,
            "n_frames": loop.n_frames,
            "done": loop.is_done,
            "located": self._located is not None,
            "guidance_status": self.guidance_status,
            "terrain": self.terrain_name,
            "step_interval_s": _STEP_INTERVAL_S,
        }

    def reset(self) -> Dict[str, Any]:
        """Stop, rebuild, restart the loop. Why: replay the demo without restarting uvicorn."""
        self.stop()
        self._build()
        self.start()
        return self.health()


# Module-level runner so `uvicorn integration.server:app` finds a ready `app`. The thread is
# started/stopped by the lifespan handler below (not at import) so importing the module — e.g.
# in a test — does not spawn a background thread.
runner = LoopRunner()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Start the stepper on server startup, stop it on shutdown.

    Why:
        Tying the background thread to the app lifespan keeps importing this module side-effect
        free (good for tests) while guaranteeing the loop is running whenever the server serves.
    """
    runner.start()
    try:
        yield
    finally:
        runner.stop()


app = FastAPI(title="SAR integration server", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def get_health() -> Dict[str, Any]:
    """Liveness + run progress. Why: a cheap probe the dashboard/ops can poll."""
    return runner.health()


@app.get("/state")
def get_state() -> Dict[str, Any]:
    """The live projected UI-MapState. Why: THE endpoint the dashboard renders from."""
    return runner.state()


@app.get("/located")
def get_located() -> Dict[str, Any]:
    """
    The LocatedEvent once declared, else {"located": false}.

    Why:
        A null-able event the broadcast/notify surface polls; wrapping the absent case in a
        small object (rather than returning null) keeps the client's parsing uniform.
    """
    event = runner.located()
    return event if event is not None else {"located": False}


@app.get("/terrain.png")
def get_terrain() -> Response:
    """
    The real-terrain backdrop image (muted colorized shaded relief), or 404 if unavailable.

    Why:
        The dashboard renders this behind the heatmap to ground the map in the actual Marin
        region. It's static per run (the grid is fixed), so it's rendered once and served from
        cache. A 404 lets the dashboard cleanly fall back to its procedural backdrop.
    """
    png = runner.terrain_png()
    if png is None:
        return Response(status_code=404)
    # Cache-friendly: the image never changes within a run.
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.post("/reset")
def post_reset() -> Dict[str, Any]:
    """Replay the search from the prior. Why: re-run the live demo without a server restart."""
    return runner.reset()
