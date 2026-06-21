# =============================================================================
# tests/test_integration_server.py
# -----------------------------------------------------------------------------
# Responsible for: Verifying the FastAPI server (integration/server.py) serves a valid
#                  UI-MapState over HTTP and that the background stepper actually advances
#                  the loop to a confident locate through the real endpoints.
# Role in project: Guards the transport seam. The dashboard depends on /state being a
#                  complete UI-MapState and on /located producing the event once found; these
#                  tests exercise both through the real ASGI app (lifespan + stepper thread).
# Assumptions: We drive the app with Starlette's TestClient, which runs the lifespan (so the
#              stepper thread starts). The step interval is forced to ~0 so the ~46-frame run
#              completes near-instantly — the test polls /health until done, no fixed sleeps.
# =============================================================================

from __future__ import annotations

import pathlib
import time

import pytest
from fastapi.testclient import TestClient

import integration.server as server
from src.search.terrain_raster import _DEFAULT_DEM

# Same required top-level keys the dashboard's types.ts MapState declares.
_REQUIRED_KEYS = {
    "missionName", "region", "startedAt", "loop", "confidenceToDeclare", "declareThreshold",
    "stats", "layers", "detections", "waypoints", "flightPath", "dronePos", "heatBlobs",
    "trend", "recentCommands", "telemetry",
}


def test_endpoints_serve_valid_shapes():
    """
    Scenario: hit /health, /state, /located on a freshly-started server.
    Why it matters: the dashboard polls these from the first instant; /state must be a complete
    UI-MapState even before the run finishes, /health must report progress, and /located must
    always return a dict (the event or {"located": false}) so the client parse is uniform.
    """
    with TestClient(server.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        state = client.get("/state")
        assert state.status_code == 200
        assert _REQUIRED_KEYS.issubset(state.json().keys())

        located = client.get("/located")
        assert located.status_code == 200
        assert isinstance(located.json(), dict)


def test_server_steps_loop_to_locate(monkeypatch):
    """
    Scenario: run the server with a ~0 step interval and poll until the loop is done.
    Why it matters: this is the real end-to-end transport test — the background stepper drives
    detector->geo->brain and the result reaches the client as JSON. We assert the run completes,
    /located carries the subject cell, and /state reports declared (confidence 100, person found).
    """
    # Force the stepper to run flat-out so the test doesn't wait on the demo cadence.
    monkeypatch.setattr(server, "_STEP_INTERVAL_S", 0.0)

    with TestClient(server.app) as client:
        # Reset so the run restarts under the fast interval (the lifespan already started one).
        client.post("/reset")

        deadline = time.time() + 15.0
        health = client.get("/health").json()
        while not health["done"] and time.time() < deadline:
            time.sleep(0.05)
            health = client.get("/health").json()

        assert health["done"], "loop did not finish within the deadline"

        located = client.get("/located").json()
        assert "cell" in located, f"expected a located event, got {located}"
        assert len(located["cell"]) == 2

        state = client.get("/state").json()
        assert state["confidenceToDeclare"] == 100
        assert state["stats"]["personsFound"] == 1


def test_server_runs_through_guidance_to_arrived(monkeypatch):
    """
    Scenario: run the server flat-out and poll until the guide-home phase reports 'arrived'.
    Why it matters: this is the new act-2 path end to end through HTTP — after locating, the
    server plans the route, replays the guidance, and /state exposes the route + the moving
    subject + the operators, finishing 'arrived' with a completed Guide Home ribbon step.
    """
    monkeypatch.setattr(server, "_STEP_INTERVAL_S", 0.0)

    with TestClient(server.app) as client:
        client.post("/reset")

        deadline = time.time() + 20.0
        health = client.get("/health").json()
        while health["guidance_status"] != "arrived" and time.time() < deadline:
            time.sleep(0.05)
            health = client.get("/health").json()

        assert health["guidance_status"] == "arrived", f"did not reach arrived: {health}"

        state = client.get("/state").json()
        assert state["guidanceStatus"] == "arrived"
        assert state["guidancePath"] and len(state["guidancePath"]) >= 2
        assert state["subjectPos"] is not None and state["operatorPos"] is not None
        guide_step = next(s for s in state["loop"] if s["label"] == "Guide Home")
        assert guide_step["status"] == "done"


@pytest.mark.skipif(not pathlib.Path(_DEFAULT_DEM).exists(), reason="real DEM raster not present")
def test_terrain_endpoint_serves_png():
    """
    Scenario: GET /terrain.png with the real rasters present.
    Why it matters: the dashboard renders this image behind the heatmap; it must be served as
    a real PNG with the image/png content type so the browser can use it as a backdrop.
    """
    with TestClient(server.app) as client:
        res = client.get("/terrain.png")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/png"
        assert res.content[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes
