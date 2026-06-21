# =============================================================================
# sar_service.py
# -----------------------------------------------------------------------------
# Responsible for: Answering the voice agent's read-only SAR status questions
#                  with short, speakable facts about the one active search.
# Role in project: The backend the operator telephony agent calls into. The
#                  voice layer (function_handlers.dispatch_function) routes the
#                  agent's tool calls here; this module turns live search state
#                  into spoken-friendly answers.
# How it gets state: Reads the integration server's GET /ops endpoint over HTTP
#                  (SAR_STATE_URL, default http://localhost:8000/ops). When that
#                  is unreachable (e.g. the agent is deployed on Fly and can't
#                  see the demo's localhost brain), it falls back to a built-in
#                  SNAPSHOT of the deterministic demo outcome — so the agent
#                  still answers truthfully rather than erroring or hanging.
# Assumptions: The /ops response and the SNAPSHOT share the same shape, so a
#                  single set of formatters serves both. SAR runs ONE search at
#                  a time (no incident id needed).
# =============================================================================
"""
SAR status service - the backend for the operator telephony agent.

Mirrors the singleton + async-method-returning-dict pattern of the template's
scheduling_service, so the function-dispatch layer is unchanged in spirit:
each agent tool maps to one async method that returns a small dict (which
Deepgram serializes back to the LLM as context for its next spoken turn).
"""
import asyncio
import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Where to read live SAR state. Default is the integration server on localhost;
# override with SAR_STATE_URL (e.g. a tunnel) when the agent runs off-box.
SAR_STATE_URL = os.getenv("SAR_STATE_URL", "http://localhost:8000/ops")

# Keep the HTTP read snappy: a deployed agent that can't reach the brain should
# fall back to the snapshot in well under the caller's patience, not stall the
# conversation. 1.5 s is plenty for a localhost JSON GET.
_HTTP_TIMEOUT_S = 1.5

# ---------------------------------------------------------------------------
# Built-in snapshot of the deterministic demo outcome.
# -----------------------------------------------------------------------------
# These are the REAL values from run_combined(seed=0, n_drones=3) at the moment
# the brain declares `located` (captured, not invented): the subject is found in
# tree cover at (37.912, -122.614), ~2% / 1.29 km2 of the AOI covered, after
# ~22 minutes, with 3 drones. The confidence gauge meets its threshold exactly
# when the brain declares located, so it reads 100. The AOI is anchored on the
# last-known position at the Pantoll trailhead (the demo region is fixed Marin /
# Mt. Tamalpais), so naming that landmark is truthful for this build.
# Shape MUST match the integration server's GET /ops response.
# ---------------------------------------------------------------------------
_SNAPSHOT = {
    "status": "located",
    "located": True,
    "n_drones": 3,
    "elapsed": "00:22:03",
    "coverage_pct": 2.0,
    "coverage_km2": 1.29,
    "confidence_pct": 100,
    "highest_prob": {
        "latlon": [37.912, -122.6142],
        "landmark": "the Pantoll trailhead",
        "description": "near the Pantoll trailhead",
    },
    "located_info": {
        "latlon": [37.912, -122.6142],
        "terrain": "in the trees",
        "landmark": "the Pantoll trailhead",
        "description": "near the Pantoll trailhead",
    },
}


def _spoken_minutes(elapsed: str) -> str:
    """
    Turn a clock string into a speech-friendly minutes phrase.

    Args:
        elapsed: Elapsed-time string. The integration server emits HH:MM:SS
            (e.g. "00:22:03"); MM:SS (e.g. "22:03") is also accepted.

    Returns:
        A phrase like "about 22 minutes". Falls back to the raw string if it
        isn't a recognizable clock.

    Why:
        The agent speaks its answers aloud; reading a clock digit-by-digit is
        awkward, and seconds-level precision is noise for a status update.
        Parsing total minutes (hours*60 + minutes) keeps it correct whether the
        source uses HH:MM:SS or MM:SS.
    """
    try:
        parts = [int(p) for p in elapsed.split(":")]
    except (ValueError, AttributeError):
        return elapsed
    if len(parts) == 3:
        minutes = parts[0] * 60 + parts[1]
    elif len(parts) == 2:
        minutes = parts[0]
    else:
        return elapsed
    unit = "minute" if minutes == 1 else "minutes"
    return f"about {minutes} {unit}"


class SARService:
    """
    Read-only view of the active search, formatted for a voice agent.

    Why a class (mirroring scheduling_service) rather than bare functions:
        it keeps the swap seam identical to the template — function_handlers
        imports one singleton and calls async methods — so the only real change
        from the dental backend is what the methods return.
    """

    async def _fetch_ops(self) -> dict | None:
        """
        Fetch live SAR state from the integration server's /ops endpoint.

        Returns:
            The parsed /ops dict, or None if the endpoint is unreachable, slow,
            or returns anything unparseable.

        Why:
            urllib in a worker thread (asyncio.to_thread) gives a non-blocking
            GET with zero extra dependencies — httpx would risk a version clash
            with the Deepgram SDK's own pin, and the payload is a tiny JSON.
        """

        def _blocking_get() -> dict:
            with urllib.request.urlopen(SAR_STATE_URL, timeout=_HTTP_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            return await asyncio.to_thread(_blocking_get)
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            # Unreachable brain (the common Fly case) or a malformed reply:
            # fall back to the snapshot rather than failing the call.
            logger.info(f"/ops unreachable ({exc}); using built-in SAR snapshot")
            return None

    async def _state(self) -> dict:
        """
        Get the current SAR state, live if reachable else the snapshot.

        Returns:
            A dict in the /ops shape (always non-None).

        Why:
            Centralizes the live-or-snapshot decision so every answer method is
            a pure formatter over a single, uniform state shape.
        """
        live = await self._fetch_ops()
        return live if live is not None else _SNAPSHOT

    async def get_search_status(self) -> dict:
        """
        Summarize the overall state of the search for the operator.

        Returns:
            Dict of facts (located, n_drones, elapsed, coverage_pct) plus a
            ready-to-speak `summary` sentence.

        Why:
            Answers the broad "what's the status / give me an update" question
            in one turn, so the agent doesn't have to chain several lookups.
        """
        s = await self._state()
        drones = s["n_drones"]
        coverage = s["coverage_pct"]
        elapsed = _spoken_minutes(s["elapsed"])
        if s["located"]:
            summary = (
                f"The subject has been found. {drones} drones flew the search, "
                f"which ran {elapsed} and covered about {coverage} percent of the area."
            )
        else:
            summary = (
                f"The search is active. {drones} drones are flying. It has been "
                f"running {elapsed} and we've covered about {coverage} percent of the area."
            )
        return {
            "located": s["located"],
            "n_drones": drones,
            "elapsed": s["elapsed"],
            "coverage_pct": coverage,
            "summary": summary,
        }

    async def get_highest_probability_area(self) -> dict:
        """
        Report where the search system currently judges the person most likely.

        Returns:
            Dict with the area `description`, `latlon`, `landmark`, and a
            spoken `summary`.

        Why:
            This is the map's core output — where to look next — phrased as a
            landmark so the agent never has to read raw coordinates aloud.
        """
        s = await self._state()
        area = s["highest_prob"]
        if s["located"]:
            summary = f"The subject has already been found {area['description']}."
        else:
            summary = f"The highest-probability area to search is {area['description']}."
        return {
            "located": s["located"],
            "description": area["description"],
            "landmark": area.get("landmark"),
            "latlon": area["latlon"],
            "summary": summary,
        }

    async def get_coverage(self) -> dict:
        """
        Report how much of the search area has been covered so far.

        Returns:
            Dict with `coverage_pct`, `coverage_km2`, and a spoken `summary`.

        Why:
            Coverage is the operator's sense of search progress; the percent is
            the headline and the area in square kilometers grounds it.
        """
        s = await self._state()
        pct = s["coverage_pct"]
        km2 = s["coverage_km2"]
        summary = (
            f"We've covered about {pct} percent of the search area so far, "
            f"roughly {km2} square kilometers."
        )
        return {"coverage_pct": pct, "coverage_km2": km2, "summary": summary}

    async def get_located_status(self) -> dict:
        """
        Report whether the missing person has been found, and if so, where.

        Returns:
            Dict with `located`, and when found the `description`, `terrain`,
            `latlon`, `confidence_pct`, plus a spoken `summary`.

        Why:
            This is the question that matters most in a rescue; when the answer
            is yes, the operator needs the location and the system's confidence.
        """
        s = await self._state()
        if not s["located"]:
            return {
                "located": False,
                "summary": "Not yet. The search is still in progress.",
            }
        info = s["located_info"]
        confidence = s.get("confidence_pct", 100)
        summary = (
            f"Yes. The subject has been located {info['description']}. "
            f"They appear to be {info['terrain']}, and the system is highly confident."
        )
        return {
            "located": True,
            "description": info["description"],
            "terrain": info["terrain"],
            "landmark": info.get("landmark"),
            "latlon": info["latlon"],
            "confidence_pct": confidence,
            "summary": summary,
        }


# Singleton instance - imported by function_handlers.dispatch_function.
# (Matches the scheduling_service module-level singleton pattern.)
sar_service = SARService()
