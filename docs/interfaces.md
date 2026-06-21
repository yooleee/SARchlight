# Component Interface Contracts

**Status: draft for Saturday's first hour.** Per the project plan §11, parallelization
hinges on these contracts being fixed before anyone builds. This document is the thing the
team ratifies (not re-derives) in the opening session so each component can be built
against stubs without collisions.

These are **data contracts and component boundaries**, deliberately language- and
framework-neutral — no implementation, no chosen serialization, no folder paths locked.
Field types are described abstractly; the team picks the concrete representation (dataclass,
Pydantic model, TypedDict, JSON schema, …) when building begins.

> **No-build rule:** nothing here is to be implemented before Saturday morning. This is a
> contract spec, not code.

---

## 0. The pipeline at a glance

```
frames + scripted telemetry
        │
        ▼
   ┌──────────┐
   │ Detector │  pixel-space, geography-blind, swappable backend
   └──────────┘
        │  DetectorOutput  (per frame)
        ▼
   ┌──────────────┐
   │ GeoReferencer│  fuses DetectorOutput + CameraPose → ground
   └──────────────┘
        │  Observation  (per frame: footprint cells + ground detections)
        ▼
   ┌──────────────┐
   │  Search map  │  the brain: Bayesian update, single writer of state
   │  (the brain) │
   └──────────────┘
        │  MapState (read model)        LocatedEvent (on confident find)
        ├───────────────► Dashboard      (reads MapState)
        ├───────────────► Subject broadcast (reads LocatedEvent + MapState)
        └───────────────► Operator       (reads MapState; writes OperatorCommand)
```

**Three locked architectural decisions** behind this shape:

1. **Separate `GeoReferencer`.** The detector never sees geography. This preserves the
   swappable-backend invariant: a fine-tuned model, a different YOLO, or a thermal model
   drops in without touching any spatial code. Geography lives in exactly one place.
2. **In-process single-writer `MapState`.** The brain is the only writer; every other
   component reads. `MapState` is kept **serializable** so it can later cross a process or
   socket boundary (e.g. to a web dashboard, or Redis if we go multi-process) *without a
   contract change*. We build the serializable seam now and defer the transport decision.
3. **Confidence-as-likelihood-ratio (clipped).** Detector confidence enters the Bayesian
   update as a clipped likelihood ratio, with multi-frame persistence required before a
   subject is declared located. See §5.

**Invariants (do not change without re-ratifying):** the `GridSpec` shared frame of
reference; the direction of data flow (detector → geo → map → consumers); single-writer
state. **Stage-appropriate / expected to evolve:** exact field sets, the `LR(c)` and `d_i`
functional forms, georeferencing fidelity (linear vs full projection), transport.

---

## 1. Shared backbone — `GridSpec`

Every spatial component agrees on one grid. `GridSpec` is the single source of truth for
the region's geometry and lives in the shared/common layer.

| Field | Type | Meaning |
|-------|------|---------|
| `crs` | string | Coordinate reference system. **Recommended: local UTM zone** (meters). Equal-area cells + plain Euclidean distance; avoids degree-vs-meter distortion in the Bayesian math. |
| `origin` | (lat, lon) | Anchor of cell (0, 0); fixes the grid over the real region. |
| `cell_size_m` | float | Square cell edge length in meters. |
| `n_rows`, `n_cols` | int | Grid extent. |

**Required conversions** (the only spatial primitives every component shares):
- `latlon_to_cell(lat, lon) -> (row, col)`
- `cell_to_latlon(row, col) -> (lat, lon)`  (cell center)

Referenced by: detector coverage (via GeoReferencer), the map, and the dashboard renderer.
A cell is addressed as `(row, col)` everywhere. Pick `cell_size_m` to balance resolution
vs compute — e.g. 25–50 m for a wide region is a sane starting point, tunable later.

---

## 2. Contract 1 — `DetectorOutput` (perception → geo)

Produced by the **Detector**, one per frame. **Pixel-space and geography-blind** — no
lat/lon, no grid. This is what keeps the backend swappable.

| Field | Type | Meaning |
|-------|------|---------|
| `frame_id` | id | Stable handle for this frame; the join key to its `CameraPose`. |
| `timestamp` | time | When the frame was captured (sim clock is fine). |
| `detections` | list | Zero or more detections (below). |
| `model_id` | string | Which backend produced this (e.g. `yolo-v?-pretrained`, `…-finetuned`). Lets the map record provenance and lets us A/B the fine-tuned swap. |
| `sensor_type` | enum | `color` \| `thermal`. Affects detection probability downstream (§5). |
| `inference_ms` | float? | Optional latency, for the dashboard / Arize if used. |

**Detection** (one element):

| Field | Type | Meaning |
|-------|------|---------|
| `bbox_xywh` | (x, y, w, h) | Pixel-space box: top-left + size, in image pixels. |
| `confidence` | float ∈ [0, 1] | Raw detector score. Consumed as evidence strength, **not** as a probability of presence (§5). |
| `class` | string | `person` for the weekend (single class — keep it person-focused). |

A frame with no people yields `detections = []`. **An empty list is meaningful**, not a
no-op: it is the non-detection signal that lets the map *lower* probability over the area
the frame covered (§5). Coverage matters as much as detections.

---

## 3. Per-frame `CameraPose` (telemetry → geo)

Supplied **alongside** each frame from the scripted flight path (in a real drone: GPS +
IMU + gimbal). Joined to `DetectorOutput` by `frame_id`. The Detector never sees this.

| Field | Type | Meaning |
|-------|------|---------|
| `frame_id` | id | Join key to `DetectorOutput`. |
| `drone_latlon` | (lat, lon) | Camera position. |
| `altitude_agl_m` | float | Height above ground level — sets ground sample distance. |
| `heading_deg` | float | Compass heading of the camera. |
| `gimbal_pitch_deg` | float | Downward tilt (90° = straight down / nadir). |
| `fov_deg` | float or (h, v) | Camera field of view. |
| `image_size_px` | (w, h) | Frame dimensions, to map pixels → ground. |

For the baseline demo this can be a simple scripted table keyed by `frame_id`. The full
camera-intrinsics version is a behind-the-seam upgrade (§4). The demo's concrete per-frame
scripted-path format (these fields split into `drone_lat`/`drone_lon`, `hfov_deg`/`vfov_deg`,
etc., plus a `footage_ref` linking each scenario frame to the real footage that backs it) is
specified in `docs/demo_scenario.md` §5.

---

## 4. Contract 2 — `Observation` (geo → map)

Produced by the **GeoReferencer** from `DetectorOutput` + `CameraPose`. This is the *only*
thing the map consumes. It answers "which ground cells did this frame see, how well, and
where (if anywhere) was a person?"

| Field | Type | Meaning |
|-------|------|---------|
| `frame_id` | id | Provenance back to the source frame. |
| `timestamp` | time | Carried through from the frame. |
| `footprint` | list of cell-coverage | The ground cells this frame covered, with how well — see below. Drives the non-detection update. |
| `detections_ground` | list | Each source detection mapped to a ground cell + its confidence. |
| `sensor_type` | enum | `color` \| `thermal`, carried through; modulates detection probability. |

**Cell-coverage** (one element of `footprint`):

| Field | Type | Meaning |
|-------|------|---------|
| `cell` | (row, col) | A grid cell the frame overlapped. |
| `coverage_fraction` | float ∈ [0, 1] | How much of the cell fell inside the frame footprint. |
| `visibility_weight` | float ∈ [0, 1] | How observable a person would be here this look: from land cover (canopy ↓), sensor type (thermal helps under low light), and view angle. |

**Ground-detection** (one element of `detections_ground`):

| Field | Type | Meaning |
|-------|------|---------|
| `cell` | (row, col) | Where the detection's box center projects on the ground. |
| `confidence` | float ∈ [0, 1] | Carried from the source detection. |

### How the GeoReferencer ties a frame to the ground (open question Q-B, resolved)

Project the camera pose through the frame to a **ground footprint polygon**, discretize it
to the grid → `footprint` (each cell with its `coverage_fraction` and a `visibility_weight`
sampled from the land-cover layer). Map each detection's pixel box center through the same
projection → its ground `cell`. Non-detections apply across **all** covered cells, which is
why coverage is first-class.

- **Baseline (build first):** treat the footprint as a rectangle on the ground; linear
  pixel→ground interpolation within it. Plenty for the demo.
- **Upgrade (behind the same interface):** full perspective projection from intrinsics +
  pose. Swappable without changing the `Observation` contract.

---

## 5. The Bayesian update (open question Q-A, resolved)

This is the brain's core math and the contract the loop owner builds to. It is standard
**Bayesian search theory** (Koopman / probability-of-area). State is a probability mass
function over cells: `p_i = P(person in cell i)`, plus a reserved mass `p_out` for "the
subject is outside the searched region," with `Σ_i p_i + p_out = 1`. `p_out` is **not
optional**: it is what non-detections drain probability *into*, so the map can express
"they're probably not here" instead of endlessly reshuffling mass between cells. Seed it
from behavior stats (the chance the subject left the planning area). Every frame's
`Observation` is one update; detections and non-detections are two cases of the *same*
per-cell update.

**Prior `p_i⁽⁰⁾`** is built before the loop runs, from terrain difficulty + vegetation +
last-known-position + lost-person-behavior statistics (the `data/behavior/` reference). The
prior is the map that "explains where to start and why" in the demo.

### 5.1 Detection probability `d_i` — where *coverage* enters

For each covered cell, define the chance we'd have detected a person there on this look:

```
d_i = coverage_fraction_i  ×  visibility_weight_i  ×  recall(sensor_type, altitude)
```

- `coverage_fraction_i`, `visibility_weight_i` come straight from the `Observation`.
- `recall(...)` is the detector's true-positive rate for the sensor and altitude (a tuned
  constant or small lookup for the weekend — does **not** need calibration data).

`d_i` is the single knob that makes coverage principled: a partial or canopy-occluded look
gives a small `d_i`, so seeing nothing there barely clears it; a full thermal nadir look
gives a large `d_i`, so seeing nothing clears it strongly.

### 5.2 Non-detection update (a cell was covered, nothing seen there)

```
p_i ← p_i · (1 − d_i)      for each covered cell i with no detection
then renormalize so Σ p_i = 1
```

Searching and seeing nothing lowers a cell's probability and (after renormalization) raises
everywhere else — exactly the behavior that makes the map redirect the search.

**Accumulated coverage** (the "cleared" map the dashboard shows) compounds over repeated
looks:

```
cleared_i = 1 − Π_k (1 − d_i⁽ᵏ⁾)
```

A cell looked at many times with good `d_i` approaches fully cleared; one weak glance does
not clear it.

### 5.3 Detection update (a detection landed in cell i) — where *confidence* enters

Confidence is **not** a probability of presence; it is evidence strength. Apply it as a
**Bayes factor** over a small spatial neighborhood:

```
p_k ← p_k · LR(c) · kernel(k, i)   for cells k near the detection's projected cell i
then renormalize (over all cells and p_out)
```

- `LR(c)` is the full likelihood ratio `P(detection here | person here) / P(detection here
  | person not here)`, **clipped to `[1, LR_max]`** so one frame can never collapse the map.
  There is no separate `f` for the other cells: after renormalization only this ratio
  matters, so we fold everything into `LR(c)` — one knob, not two. (Robustness against aerial
  false positives: rocks, shadows, animals.)
- Make `LR(c)` **piecewise / bucketed**, not a smooth curve: raw detector confidence is
  uncalibrated (typically overconfident), so a step function (ignore below threshold →
  modest LR → `LR_max`) is more honest than implying continuous calibration.
- `kernel(k, i)` spreads the update over cells *near* `i` (a small Gaussian), because at
  altitude a few pixels of box-center error is tens of meters on the ground. A single-cell
  update is brittle to projection error; the kernel makes localization uncertainty explicit
  and lets adjacent-cell hits corroborate each other.
- Clip + renormalize means one good detection sharply concentrates mass without zeroing the
  rest, and a lone false positive is recoverable by subsequent non-detections.

### 5.4 The `located` trigger (fires broadcast + operator alert)

A detection alone does **not** declare a person found, and the trigger must not depend on
grid resolution. Require **both**:

1. **Local aggregated mass:** the summed posterior over a small window around the peak cell
   crosses `P_located` — and/or a relative criterion (peak ≫ median, or posterior entropy
   drops below a threshold). Aggregating over a window, rather than testing one cell's
   absolute `p_i`, keeps the trigger invariant to `cell_size_m`.
2. **Persistence:** detections in/adjacent to that peak across `N` overlapping frames.

**Keep the persistence gate separate from the Bayesian evidence.** Consecutive frames of the
*same* person through the *same* detector are correlated, not independent — so do **not**
multiply `LR(c)` afresh every frame for the same blob (that over-concentrates the posterior
and double-counts). Cap per-region detection evidence: treat a run of hits on one blob as a
single piece of accumulating evidence, and let persistence be confirmation logic *on top of*
the posterior, not more multiplications *into* it. (The principled version is a lightweight
detection *track* across frames; the cap is the weekend-appropriate stand-in.)

This higher bar protects the demo's emotional high point — the broadcast — from firing on a
single false positive. When it trips, emit a `LocatedEvent` (§6).

### 5.5 Assumptions & seams (acknowledge now, build later)

- **Stationary subject (the big one).** The update above treats the person as fixed and only
  *measures*. A proper moving-target search adds a **predict step** between measurements — a
  diffusion/transition kernel that spreads the posterior per time step (the predict half of a
  Bayes filter / HMM forward pass). Defensible to omit for a found-in-place demo, but **leave
  the loop seam**: the loop should call `predict()` then `update()`, with `predict()` a no-op
  for now. Behavior stats then feed two things: the prior's distance-decay from LKP, and
  (later) this transition kernel.
- **Correlated looks.** `cleared_i = 1 − Π(1−d_i)` assumes independent glimpses, but repeated
  passes from the same angle/sensor miss identically under canopy, so the product *overstates*
  clearance. Note it; optionally cap cumulative clearance per (cell, sensor).
- **Prior combination rule.** `prior_i ∝ D(dist(i, LKP)) × accessibility_i × corridor_i`,
  normalized alongside `p_out` — now fully specified (formula, defaults, Koester σ ≈ 2.6 km,
  Euclidean→cost-distance upgrade) in `docs/prior_model.md`.

> Tunables to lock as config, not code: `recall(...)`, `LR_max`, the `LR(c)` buckets,
> `P_located`, persistence `N`, the detection kernel width, `p_out`. Stage-appropriate —
> expected to be tuned on Saturday.

---

## 6. Contract 3 — `MapState`, `LocatedEvent`, `OperatorCommand`

### 6.1 `MapState` — the read model (single source of truth)

Written **only** by the brain; read by the dashboard, both voice surfaces, and the search
director. Serializable so it can cross a boundary later unchanged.

| Field | Type | Meaning |
|-------|------|---------|
| `grid_spec` | GridSpec (ref) | The frame of reference for every array below. |
| `update_count` | int | Monotonic; lets consumers detect a new state cheaply. |
| `timestamp` | time | Of the last update. |
| `posterior` | 2D array [n_rows × n_cols] | Current `p_i` over the grid. The centerpiece the judges watch. |
| `coverage` | 2D array | Accumulated `cleared_i` (§5.2). The "where have we looked" layer. |
| `top_cells` | ranked list | Highest-probability cells (with their `p_i`) — the search targets. |
| `next_target` / `search_path` | cells | Where the map directs the drone next. **Implemented** (C1): `next_target` = the most-likely uncleared cell; `search_path` = the sector planner's recommended single-drone boustrophedon sweep (`src/search/planner.py`). |
| `status` | enum | `searching` \| `located` \| (extensible). |
| `detections_log` | list | Confirmed detection events, for the dashboard timeline. |

### 6.2 `LocatedEvent` — emitted on a confident find

Triggers the subject broadcast and the operator alert. Carries everything the message
composer needs without it having to reach back into the brain's internals.

| Field | Type | Meaning |
|-------|------|---------|
| `cell` / `latlon` | location | Where the subject is. |
| `confidence` | float | Aggregate confidence at trigger time. |
| `terrain_context` | struct | Land cover / slope / nearby trail or water at the location — for message content and for routing ground teams. |
| `timestamp` | time | When the trigger fired. |

### 6.3 Voice layer — what it reads and writes

- **Subject broadcast** (protected, priority 1): reads `LocatedEvent` + `MapState`; an LLM
  composes situation-aware content (location-relative guidance, "stay put," reassurance,
  help-status) → TTS out through the simulated drone. Multilingual is the **cheap stretch**:
  the message is *generated*, so a target-language parameter is nearly free.
- **Operator** (stretch, first to cut): **reads** `MapState` to answer hands-free queries
  ("highest-probability area?", "how much have we covered?", "status?"), and **writes**
  `OperatorCommand` back to the orchestrator.

**`OperatorCommand`** — the only write path from voice into the system. Keeps the brain
authoritative (voice never mutates `MapState` directly):

| Field | Type | Meaning |
|-------|------|---------|
| `kind` | enum | e.g. `search_here` \| `broadcast_now` \| `mark_false_positive` \| `query`. |
| `payload` | struct | Kind-specific args (a target cell, a detection id, a query string). |

---

## 7. What this buys us Saturday

With these fixed, the independent tracks (§11 of the plan) build against stubs in parallel:

- **Dashboard** renders a mock `MapState` immediately.
- **Subject broadcast** composes from a mock `LocatedEvent` immediately.
- **Operator front-end** parses speech into mock `OperatorCommand`s immediately.
- **Detector track** emits `DetectorOutput`; the fine-tuned model is a `model_id` swap.
- **The brain** (critical path, single owner) builds the `Observation → MapState` loop
  against a stubbed GeoReferencer and a stubbed detection stream.

Integration is then wiring real producers to real consumers across boundaries that already
agree — not a redesign.
