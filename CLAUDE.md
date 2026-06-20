# Project Agent Instructions — Wide-Area Search & Rescue Support System

Instructions for any agentic coding tool (and teammate) working in this repo. Start with
`docs/strategy_ogsm.md` — the project's north-star, which reconciles every other doc and
carries a document map (§4.3) routing you to the right one. Then read the docs relevant to
your task, in particular `docs/SAR_project_plan.md` (the full design) and `docs/interfaces.md`
(the contracts), before doing anything.

---

## ✅ Build mode ON — activated 2026-06-20; the pre-event no-build rule no longer applies.

The event has started. Writing project/pipeline code, building models, and assembling
pipeline components is now permitted and expected. Follow `prep/runbook.md` (the opening
sequence) and the per-component prompts in `prep/`. Build stubs-first against the ratified
contracts in `docs/interfaces.md`; protect the core loop above all else (see below).

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
  isolated feasibility-showcase** (not our aerial detector), per `docs/board_feasibility.md`.
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

## Proposed directory layout — **ratify with the team Saturday, not yet created**

The `src/` tree below is a **starting proposal only**. Literal file locations are a decision
to make together when building begins — do not materialize this structure pre-event.

```
sar-system/
├── docs/        SAR_project_plan.md, interfaces.md            [exists]
├── data/        terrain/ detection/ footage/ behavior/ weights/ (gathered, git-ignored)  [exists]
│
│   # ── proposed; confirm before creating ──
├── src/
│   ├── detection/   detector wrapper, pixel-space, swappable backend
│   ├── geo/         GeoReferencer: frame→ground projection, grid math
│   ├── search/      probability map + Bayesian update (the brain)
│   ├── voice/       broadcast/ (subject) + operator/ (STT, intent, control)
│   ├── pipeline/    loop wiring + orchestration (single-writer state)
│   └── common/      shared types/contracts, GridSpec, config
├── dashboard/   map UI, detections, alerts
├── scratch/     experiments, including fine-tuning
└── demo/        demo scripts + scenario config
```

`data/` subfolders are gathering buckets (see `data/README.md` for provenance), adjustable,
not a structural commitment.

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
