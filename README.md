# SARchlight

**An AI rescue-intelligence layer that turns drone footage, terrain, and uncertainty into a live map of *where to search next*.**

UC Berkeley AI Hackathon 2026 · wide-area Search & Rescue support.

SARchlight is the *brain* behind a search-and-rescue drone — not the drone itself. It builds a
live probability map of where a missing person likely is from last-known position, terrain, and
land cover; directs drones to the highest-probability sectors; turns image detections into ground
locations; updates the map from **both detections and clean non-detections**; coordinates multiple
drones so they never re-cover the same ground; and declares a subject *located* only after
persistent evidence. Once found, a drone guides the subject home and the system speaks to them.

---

## The closed loop

```
prior map ─▶ directs search path ─▶ detector on footage ─▶ detections + coverage update map
    ▲                                                                      │
    └──────────────── updated map redirects + flags high-probability areas ┘
                                     │
              confident, persistent detection ─▶ subject broadcast + operator alert ─▶ guide home
```

The flight path is **not scripted** — it *emerges* from the map. A detection (or a clean sweep)
changes the belief, and the changed belief changes where the drones go next.

## What's inside

- **Probability map (the core).** A NumPy belief grid with Bayesian updates. Single-writer design
  so map state can't be corrupted by concurrent readers. See `docs/interfaces.md` for the contracts
  and the Bayesian / `located` math.
- **Terrain-aware prior.** Real DEM (elevation) + ESA WorldCover (land cover) shape where the
  subject is likely to be. See `docs/prior_model.md`.
- **Non-detection handling.** A clean sweep *lowers* probability where we've actually looked — but
  canopy means "didn't see" ≠ "not there", so probability is down-weighted, never erased.
- **`located` trigger.** Confidence-as-likelihood-ratio (clipped) + persistence, so a single
  false-positive aerial frame can't trip an alert. Locates at **zero-cell error on real Marin
  terrain** in the demo.
- **Multi-drone planner.** Probability-of-area ranking with disjoint sector assignment and a
  boustrophedon sweep — no overlapping coverage.
- **GeoReferencer.** Projects pixel detection boxes to ground cells; geography lives in exactly one
  swappable place, so the detector stays geography-blind.
- **Swappable detector.** A simulator *and* a real YOLO path behind one adapter, so the loop runs
  immediately while the detector is an independent upgrade.
- **Guide-home.** After locating, a drone leads the mobile subject home along a terrain-aware route.
- **Voice layer.** A synthesized **subject broadcast** (Deepgram TTS) speaks to the found person,
  and a **Twilio-backed operator phone agent** (Deepgram Voice Agent, deployed on Fly.io) lets
  ground operators call in and ask the live system questions — coverage, the current
  highest-probability area, and whether the subject has been found.
- **Live dashboard.** A React/TypeScript/Tailwind app showing the probability heat map over real
  terrain, drone positions, detections, routing, the live video feed, and the call transcript.
- **Observability.** The whole stack is instrumented with **Sentry** — the FastAPI brain, the
  React dashboard, and the remotely deployed phone agent — so failures that would otherwise be
  invisible (a background search-thread crash, a degraded voice line, an error mid-phone-call)
  surface immediately. Fully env-gated: with no DSN it is a complete no-op.

## Repository layout

```
src/            the brain: GridSpec + contracts (common/), GeoReferencer (geo/),
                prior + Bayesian update + located trigger + planner + terrain (search/),
                and runnable demos (demo/)
integration/    the transport seam: FastAPI server, the steppable loop, detector backends,
                broadcast + Deepgram TTS, map rendering, Sentry init (observability.py)
dashboard_app/  the live React/Vite dashboard
voice/          the Twilio + Deepgram operator phone agent (deploys to Fly.io)
detector/       the real YOLO detector adapter
tests/          unit + integration + soak tests (pytest)
docs/           technical reference: interfaces, prior_model, core_loop, data, tech_stack
data/           terrain rasters (gitignored — see Setup)
```

## Setup

```bash
# 1. Python backend (the brain + integration server)
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. Secrets — copy the template and fill in what you need
cp .env.example .env        # DEEPGRAM_API_KEY, ANTHROPIC_API_KEY; SENTRY_DSN is optional

# 3. Terrain rasters are large and gitignored. Verify they are present + intact:
.venv/bin/python check_setup.py     # see docs/data.md for how to fetch them
```

## Run

```bash
# Tests (unit + integration + soak)
.venv/bin/python -m pytest

# Brain demo on synthetic terrain (locates) -> demo_output/
.venv/bin/python -m src.demo.run

# Multi-drone sector search (closed loop)
.venv/bin/python -m src.demo.search_demo --drones 3

# Closed loop CLI: detector -> geo -> brain (simulator backend)
.venv/bin/python -m integration.loop
#   ...or the real YOLO backend:
.venv/bin/python -m integration.loop --video <footage> --weights <weights>

# Live dashboard: start the server, then the React app
.venv/bin/uvicorn integration.server:app        # http://localhost:8000  (serves /state, /map_base.png, /ops, ...)
cd dashboard_app && npm install && npm run dev   # http://localhost:5173

# Operator phone agent (local dev; needs DEEPGRAM_API_KEY)
cd voice && python main.py                       # deploys to Fly.io via voice/ — see voice/README.md
```

## Tech stack

Python · NumPy · rasterio · FastAPI · YOLO · React · TypeScript · Vite · Tailwind ·
Claude (Anthropic) · Deepgram · Twilio · Fly.io · Sentry

## What's next

Live drone telemetry, GPS/camera and thermal footage for real-time updates; richer
probability models from real SAR behavior and trail data; and multilingual voice broadcasts.
Known limitations and deferred work are noted alongside the relevant docs in `docs/`.
