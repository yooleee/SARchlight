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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.common.contracts import LocatedEvent
from integration.dashboard_projection import ProjectionContext, project
from integration.loop import build_simulator_loop

# Seconds between frames. Slow enough to WATCH the map evolve in the demo, fast enough that
# the ~46-frame run locates in well under a minute. A tunable, not a constant in the math.
_STEP_INTERVAL_S = float(os.environ.get("SAR_STEP_INTERVAL", "0.7"))

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
        self._build()

    def _build(self) -> None:
        """
        (Re)build the loop from scratch and project its prior state.

        Why:
            Used on construction and on /reset, so a live demo can replay the search without
            restarting the server. Trend/located are cleared because they belong to a run.
        """
        self._loop, self.terrain_name = build_simulator_loop()
        self._trend = []
        self._located = None
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
        return ProjectionContext(
            drone_path=loop.drone_path,
            last_pose=loop.last_pose,
            trend=list(self._trend),
            located_cell=loop.located_event.cell if loop.located_event else None,
            started_at="live",
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
        The single-writer loop: step, project, publish — once per interval until done.

        Why:
            This is the ONLY place the brain advances. It computes the fresh projection, then
            records the current confidence into the trend for the NEXT projection (a one-frame
            lag, invisible in the chart) and latches the LocatedEvent the first time it fires.
        """
        while not self._stop.is_set():
            if not self._loop.is_done:
                result = self._loop.step()
                if result is not None:
                    map_state, event = result
                    ui = project(map_state, self._context())
                    with self._lock:
                        self._state_dict = ui
                        self._trend.append((map_state.timestamp, ui["confidenceToDeclare"]))
                        if event is not None and self._located is None:
                            self._located = _located_to_dict(event)
            # Wait on the stop event (not time.sleep) so shutdown is responsive mid-interval.
            self._stop.wait(_STEP_INTERVAL_S)

    # --- read model (lock-guarded; called from request handlers) ---

    def state(self) -> Dict[str, Any]:
        """The latest projected UI-MapState. Why: the dashboard's /state poll target."""
        with self._lock:
            return self._state_dict

    def located(self) -> Optional[Dict[str, Any]]:
        """The latched LocatedEvent dict, or None. Why: drives the broadcast/notify surface."""
        with self._lock:
            return self._located

    def health(self) -> Dict[str, Any]:
        """Run progress for /health and the reset response. Why: lets the UI show frame i/n."""
        loop = self._loop
        return {
            "status": "ok",
            "frame": loop.frame_index,
            "n_frames": loop.n_frames,
            "done": loop.is_done,
            "located": self._located is not None,
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


@app.post("/reset")
def post_reset() -> Dict[str, Any]:
    """Replay the search from the prior. Why: re-run the live demo without a server restart."""
    return runner.reset()
