# =============================================================================
# test_sar_service.py
# -----------------------------------------------------------------------------
# Responsible for: Verifying the operator-agent backend (voice/backend/
#                  sar_service.py) produces correct, speakable answers in both
#                  paths: live /ops read (mocked) and the offline snapshot.
# Role in project: Guards the SAR repurpose of the Deepgram agent. The live
#                  conversation can't run in CI (it's audio), but the facts the
#                  agent speaks come from here, so these are unit-testable.
# Assumptions: No live HTTP. The live path is exercised by monkeypatching the
#                  fetch; the real urllib fallback is exercised against a
#                  closed local port (connection refused -> snapshot).
# =============================================================================
"""Tests for the SAR status service that backs the operator telephony agent."""
import asyncio
import pathlib
import sys

# The voice agent lives in voice/ and imports its backend as `backend.*`
# (it runs with voice/ as the working dir). Put voice/ on the path so this
# test can import the same module the agent does, without an install step.
_VOICE_DIR = pathlib.Path(__file__).parent.parent / "voice"
sys.path.insert(0, str(_VOICE_DIR))

import backend.sar_service as svc  # noqa: E402
from backend.sar_service import SARService, sar_service, _SNAPSHOT, _spoken_minutes  # noqa: E402


# A crafted "search in progress" /ops payload, in the same shape /ops returns.
# Used to exercise the live (reachable-brain) path and the not-yet-located
# branches without any real HTTP.
_SEARCHING_OPS = {
    "status": "searching",
    "located": False,
    "n_drones": 2,
    "elapsed": "05:30",
    "coverage_pct": 18.5,
    "coverage_km2": 11.8,
    "confidence_pct": 40,
    "highest_prob": {
        "latlon": [37.9, -122.6],
        "landmark": "the Pantoll trailhead",
        "description": "near the Pantoll trailhead",
    },
    "located_info": None,
}


def _patch_state(monkeypatch, payload):
    """
    Force the service to return a fixed state dict instead of doing HTTP.

    Args:
        monkeypatch: pytest's monkeypatch fixture.
        payload: The /ops-shaped dict (or None) that _fetch_ops should yield.

    Why:
        Every answer method is a pure formatter over _state(); pinning the
        underlying fetch lets each test assert formatting for a known state
        with no network. Returning None drives the snapshot fallback.
    """
    async def fake_fetch(self):
        return payload

    monkeypatch.setattr(SARService, "_fetch_ops", fake_fetch)


# ---------------------------------------------------------------------------
# Snapshot fallback (the deployed-agent / unreachable-brain case)
# ---------------------------------------------------------------------------

def test_snapshot_used_when_fetch_returns_none(monkeypatch):
    """When /ops is unreachable, answers come from the truthful demo snapshot."""
    _patch_state(monkeypatch, None)

    status = asyncio.run(sar_service.get_search_status())
    # The snapshot is the located outcome: 3 drones, ~2% covered, found.
    assert status["located"] is True
    assert status["n_drones"] == 3
    assert status["coverage_pct"] == 2.0
    assert "found" in status["summary"].lower()

    located = asyncio.run(sar_service.get_located_status())
    assert located["located"] is True
    assert located["terrain"] == "in the trees"
    # The spoken summary should name where, not read raw coordinates.
    assert "Pantoll" in located["summary"]


def test_snapshot_shape_matches_ops_contract():
    """The snapshot must carry every field the formatters and /ops both use."""
    for key in ("status", "located", "n_drones", "elapsed",
                "coverage_pct", "coverage_km2", "highest_prob", "located_info"):
        assert key in _SNAPSHOT
    assert {"latlon", "terrain", "description"} <= set(_SNAPSHOT["located_info"])


# ---------------------------------------------------------------------------
# Live path (reachable brain) - search still in progress
# ---------------------------------------------------------------------------

def test_search_status_in_progress(monkeypatch):
    """A live searching state is summarized as active, with its real numbers."""
    _patch_state(monkeypatch, _SEARCHING_OPS)

    status = asyncio.run(sar_service.get_search_status())
    assert status["located"] is False
    assert status["n_drones"] == 2
    assert status["coverage_pct"] == 18.5
    assert "active" in status["summary"].lower()
    # Elapsed is spoken as minutes, not a raw clock string.
    assert "5 minutes" in status["summary"]


def test_located_status_not_yet(monkeypatch):
    """While searching, the located answer is a clear 'not yet'."""
    _patch_state(monkeypatch, _SEARCHING_OPS)

    located = asyncio.run(sar_service.get_located_status())
    assert located["located"] is False
    assert "not yet" in located["summary"].lower()
    # No location fields should be fabricated when the person isn't found.
    assert "latlon" not in located


def test_highest_prob_area_while_searching(monkeypatch):
    """The priority-area answer uses the landmark description, not coordinates."""
    _patch_state(monkeypatch, _SEARCHING_OPS)

    area = asyncio.run(sar_service.get_highest_probability_area())
    assert area["located"] is False
    assert area["description"] == "near the Pantoll trailhead"
    assert "Pantoll" in area["summary"]


def test_coverage_reports_pct_and_area(monkeypatch):
    """Coverage answer leads with the percent and grounds it in square km."""
    _patch_state(monkeypatch, _SEARCHING_OPS)

    cov = asyncio.run(sar_service.get_coverage())
    assert cov["coverage_pct"] == 18.5
    assert cov["coverage_km2"] == 11.8
    assert "18.5 percent" in cov["summary"]


# ---------------------------------------------------------------------------
# The real urllib fallback path (no mock) - connection refused -> snapshot
# ---------------------------------------------------------------------------

def test_fetch_ops_returns_none_when_unreachable(monkeypatch):
    """A refused/unreachable SAR_STATE_URL yields None (so _state snapshots)."""
    # Port 9 (discard) on localhost is closed -> immediate connection refused,
    # exercising the real urllib error handling without a slow timeout.
    monkeypatch.setattr(svc, "SAR_STATE_URL", "http://127.0.0.1:9/ops")
    assert asyncio.run(sar_service._fetch_ops()) is None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def test_spoken_minutes_formats_clock():
    """HH:MM:SS (the server's format) and MM:SS both become spoken minutes; junk passes through."""
    # HH:MM:SS — what the integration server's /ops actually emits.
    assert _spoken_minutes("00:22:03") == "about 22 minutes"
    assert _spoken_minutes("01:05:00") == "about 65 minutes"
    # MM:SS — also accepted.
    assert _spoken_minutes("22:03") == "about 22 minutes"
    assert _spoken_minutes("01:00") == "about 1 minute"
    # Not a clock -> returned unchanged.
    assert _spoken_minutes("not-a-clock") == "not-a-clock"
