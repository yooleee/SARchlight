# Project Agent Instructions — Wide-Area Search & Rescue Support System

Instructions for any agentic coding tool (and teammate) working in this repo. Start with
`kickoff.md` — the start-here for this directory: your part, the read-order, the build order,
and the priorities. Then read the docs relevant to your task, in particular `docs/interfaces.md`
(the contracts + the Bayesian/`located` math) and `docs/prior_model.md` (the prior), before
doing anything. (The full design is in `docs/SAR_project_plan.md` if you need it.)

---

## ✅ Build mode ON — activated 2026-06-20; the pre-event no-build rule no longer applies.

The event has started. Writing project/pipeline code, building models, and assembling
pipeline components is now permitted and expected. Follow `kickoff.md` (the opening sequence
for this directory) and the component prompts in `prep/`. Build stubs-first against the ratified
contracts in `docs/interfaces.md`; protect the core loop above all else (see below).

---

## Current state (brain/geo/terrain track — updated 2026-06-20)

The **brain + GeoReferencer + terrain** track is **built, tested, and committed** (branch
`brain`): the probability map (prior, Bayesian update, `located` trigger, single-writer
`MapState`), the GeoReferencer (frame→ground projection + input hardening), real-raster terrain
(DEM + WorldCover), a detector *simulator*, the closed loop end-to-end, and the **real-terrain
animated showcase** (`src/demo/showcase.py` → `demo_output/showcase.gif`; locates on real Marin
terrain at 0-cell error — closed `brain_followups.md` B5). Run the synthetic demo with
`python -m src.demo.run`, the showcase with `python -m src.demo.showcase`, the **closed-loop
sector search** (C1) with `python -m src.demo.search_demo [--drones N]`, and the suite with
`pytest` (121 tests). The **C1 sectorized grid-search + multi-drone** layer is now **built**
(`src/search/planner.py`, `src/demo/search_demo.py`): a pure read-only sector planner (POA
ranking, single-drone boustrophedon, disjoint multi-drone assignment) feeding the one
single-writer brain; the closed-loop flight path *emerges* from the planner. Still **teammates'
tracks / not built in this repo**: the dashboard, the voice/broadcast, and the real YOLO11+SAHI
detector. Next planned work is **integrating teammates' code** into the brain track. Deferred
items and honest limitations live in **`docs/brain_followups.md`**.

---

## What we're building

A drone-side perception + decision layer that helps search teams find missing people in
remote terrain. Not the drone — the brain behind it. A closed loop: a probability map of
where the person likely is directs the search; the drone's detections (and clean
non-detections) update that map; the updated map directs the next search. A voice layer
runs it hands-free and speaks to the subject once found.

**The center of gravity is the probability map.** It's what wins, because it visibly
reasons about where to look. A working loop where a detection changes the map beats four
polished but disconnected features. Protect the loop above all else.

### The core loop (must work end to end)
prior map → directs search path → detector on footage → detections + coverage update map →
updated map redirects + flags high-probability areas → confident detection triggers subject
broadcast + operator alert.

---

## Resolved decisions (from plan §10 — do not relitigate mid-build)

- **Main track: deferred, build track-agnostic.** Same build pitches to social impact
  (default) or science/engineering. Hardware presence is the tiebreaker. Decide at submission.
- **Demo region: real regional data, selected.** Expanded Marin (Mt. Tamalpais → Bolinas
  Ridge → southern Point Reyes). DEM + ESA WorldCover + OSM **gathered (done 2026-06-18)**;
  ingestion code is Saturday.
- **Detection: pretrained first, fine-tune as a swappable upgrade.** Off-the-shelf detector
  so the loop runs immediately. Fine-tuning is a separate stretch track whose output drops
  in via a weights/`model_id` swap. **Cap time chasing accuracy** — a passable detector
  feeding a great map beats the reverse.
- **Physical vs software: software-first, hardware additive & isolated.** Core loop runs
  fully in software. Any board (Raspberry Pi / Coral) is one person's optional demo layer,
  fed by the same detection service, off the critical path. At most **one** physical angle
  (Coral *or* QNX, not both). The Coral was **assessed Friday 2026-06-19 → reframed to an
  isolated feasibility-showcase** (not our aerial detector).
- **Voice priority: subject broadcast first (protected), operator interface second.** The
  broadcast is cheap and the demo's high point. The operator interface is the richer build
  and the first thing to cut under time pressure.

### Scope protection
Cut from the edges, not the center. Stretch order, only once the core is solid:
1) operator voice interface · 2) multilingual broadcast (cheapest — message is generated) ·
3) detector fine-tuning · 4) a physical/on-device layer. Each is independent enough to drop
without breaking the core. Cap sponsor integrations so they never eat the loop.

---

## Component contracts — build against these, in parallel

The full spec is in **`docs/interfaces.md`**. Fix/ratify it in the first 30–60 min Saturday,
then build components against stubs without collisions. Three locked architectural decisions:

1. **Separate `GeoReferencer`.** The detector is pixel-space and geography-blind, so the
   backend stays swappable. Geography lives in exactly one place.
2. **In-process single-writer `MapState`.** The brain is the only writer; everyone else
   reads. Keep `MapState` serializable so it can cross a process/socket boundary later
   without a contract change. (Redis only if we actually go multi-process — don't add it for
   the prize alone.)
3. **Confidence-as-likelihood-ratio (clipped) + persistence** before declaring `located`.
   Robust to aerial false positives.

Pipeline: `Detector → GeoReferencer → Search map → {Dashboard, Broadcast, Operator}`.
Shared backbone is a `GridSpec` (local UTM grid) every spatial component references.
**Invariants** (don't change without re-ratifying): the `GridSpec`, the data-flow direction,
single-writer state. Everything else (field sets, the `LR(c)`/`d_i` forms, georeferencing
fidelity, transport) is expected to evolve.

---

## Directory layout

The shared structure now **exists** (built for the brain/geo/terrain track). Actual layout:

```
sar_hackathon/  (this repo)
├── docs/        design docs: interfaces.md, prior_model.md, demo_scenario.md, build_plan.md,
│                brain_followups.md, showcase-kickoff.md, ...                       [exists]
├── data/        terrain/ + behavior/ present here; detection/footage/weights are gathered in
│                the sibling ../sar_system/ prep dir (git-ignored)                  [exists]
├── src/
│   ├── common/    GridSpec, contracts (Observation/MapState/LocatedEvent/DetectorOutput/
│   │              CameraPose), config                                             [built]
│   ├── geo/       GeoReferencer: frame→ground projection + boundary defense       [built]
│   ├── search/    prior, Bayesian update, located trigger, brain, terrain
│   │              (synthetic stub + real raster)                                   [built]
│   └── demo/      scripted flight path, detector simulator, run (loop wiring + PNGs) [built]
└── tests/       unit + integration + soak tests (run with `pytest`)               [built]

# ── teammates' tracks / not in this repo yet ──
#   real detector (YOLO11+SAHI) · dashboard (Streamlit) · voice (broadcast + operator)
```

Notes: the loop wiring lives in `src/demo/run.py` (no separate `pipeline/`); the real detector,
dashboard, and voice are teammates' tracks and not present here. `data/` subfolders are gathering
buckets (see `docs/data.md` for provenance), adjustable, not a structural commitment.

---

## Conventions

- **Loop owner holds the brain.** The map + search loop + its wiring to detection is the
  critical path and convergence point; it wants one strong owner or a tight pair, not wide
  parallelism. Independent tracks (broadcast, operator front-end, dashboard shell,
  fine-tuning) parallelize against the contracts above.
- **Stubs first.** Build every component against a mock of its input contract so integration
  is wiring, not redesign.
- **Tunables in config, not code:** `recall`, `LR_max`, `LR(c)` shape, `P_located`,
  persistence `N`, prior weights, `cell_size_m`. Expected to be tuned live.
- **Secrets in a git-ignored `.env`** (never committed); `.env.example` documents the keys.
  Redact obvious secrets before sending text to an external service.
- **Ensure the submission entry is finalized and submitted before the deadline** so the project is
  guaranteed to be judged. Devpost typically unlocks late and the write-up is largely drafted from
  the docs, so it's an end-of-event task, not a hour-0 one.
- **Sponsors follow the build, not vice versa.** Committed: main track + Anthropic +
  Deepgram. Near-free add: Sentry. Conditional on choices: QNX (physical), Ultimate Bots
  (simulated drone). Everything else only if it falls out of the build for free.
