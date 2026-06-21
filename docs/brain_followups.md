# Brain — Deferred Follow-ups & Known Limitations

Tracked items deliberately **not** built yet, with the reason and the condition that
should trigger doing them. Recorded so they're remembered, not lost. Split into
(A) deferred hardening items (do once real data flows) and (B) honest limitations of
the current design (state these out loud; don't let a demo imply otherwise).

> Status as of the hardening pass: the brain runs the full loop on mock `Observation`s
> and is hardened against the *contract* (malformed input, false positives, long
> streams). The items below are hardening/tuning that depends on **real** data, plus
> modeling assumptions we chose to keep simple.

---

## A. Deferred hardening (do once the GeoReferencer + real detector are wired)

### A1. Cumulative-clearance cap per (cell, sensor) — IMPLEMENTED
- **Status:** built. The geo+brain feasibility test (Geo Unit 4) gave this a concrete,
  biting failure — loitering over the subject under canopy, the *misses* repeatedly
  cleared the subject cell and eroded the located signal back to the prior, so the cap was
  promoted from deferred to done.
- **What:** non-detection clearing is now per-sensor and capped — each sensor's cumulative
  clearance of a cell is bounded by `BrainConfig.clearance_cap_per_sensor` (default 0.6),
  because correlated looks from one sensor over the same canopy miss identically and the
  independent-looks product overstates clearance (interfaces.md §5.5). The first look still
  clears by ~`d_i`; only redundant correlated looks are damped. Different sensors clear
  independently (combined coverage = `1 − Π_sensors (1 − cleared_sensor)`).
- **Where:** `apply_coverage_and_nondetection` (`src/search/update.py`) takes the per-sensor
  `cleared` array + `clearance_cap`; the brain holds one array per sensor and derives the
  displayed `coverage`. Tests: `test_update.py::test_clearance_capped_per_sensor`.
- **Still tunable:** the cap value (0.6) is calibrated to the demo; real re-look patterns
  may move it. The principled invariant (correlated looks can't fully clear) is fixed.

### A2. `P_located` / `located_concentration_ratio` sensitivity sweep
- **What:** the trigger thresholds (`p_located`, `located_concentration_ratio`) are
  calibrated to the **demo** (the relative ratio climbs ~2.5x → ~5x → ~10x; default 7.0
  fires on the thermal corroboration).
- **Why deferred:** a meaningful sweep needs real prior shapes and real detection
  confidence distributions; on synthetic data it only characterizes the fiction. Doing
  it now would give false confidence in a number we'd re-tune anyway.
- **Trigger to do it:** once real detections feed the map — sweep the thresholds against
  recorded runs to confirm `located` fires on true finds and not on plausible false
  positives, and record the chosen values + rationale here.

---

## B. Known limitations of the current design (state honestly)

### B1. Stationary-subject assumption (`predict()` is a no-op)
- The loop calls `predict()` then `update()`, but `predict()` does nothing — the subject
  is treated as fixed. A moving-target search needs a diffusion/transition kernel in
  `predict()` (the predict half of a Bayes filter) that spreads the posterior per time
  step. **This is the largest gap between the demo and reality.** Defensible for a
  found-in-place demo; the seam (`SearchBrain.predict`) is left clean for it. Behavior
  stats would feed both the prior's distance-decay *and* this transition kernel.

### B2. A *persistent* false positive will still locate
- The brain recovers from a **transient** false positive (it never reaches persistence,
  and non-detections drain the spike — see `tests/test_robustness.py`). But a detector
  that fires on the **same rock every frame** is indistinguishable from a real person at
  the map level, so it would eventually satisfy persistence + concentration. That is the
  **detector's** responsibility (its true-positive/false-positive behavior), not the
  brain's. The LR cap (`lr_max`) bounds how far one blob can concentrate the map, which
  limits the damage, but does not eliminate it.

### B3. Concentration ratio is grid-*stable*, not grid-*invariant*
- `located_concentration_ratio` transfers across grid sizes far better than an absolute
  mass threshold (verified in `tests/test_trigger.py`), but it is **not** perfectly
  invariant: an extreme peak holding most of the map's mass still scales with grid size.
  For normal localized finds this is fine; it is called out so the claim isn't overstated.

### B5. Real terrain makes the find harder — demo scenario re-design — RESOLVED
- **Status:** RESOLVED (real-terrain animated showcase, `src/demo/showcase.py`). On the real
  rasters the loop now locates the planted subject at **0-cell error across seeds** (acceptance
  test `tests/test_showcase.py::test_real_terrain_loop_locates_the_subject`).
- **Why it was hard:** the AOI is ~72% dense tree canopy, so a realistic subject has low
  visibility (~0.4 → ~40% color detection, marginal persistence), AND a realistic subject sits
  at ~average prior, so its concentration overlapped with false positives. The synthetic demo
  hid this by placing the subject in a favorable corridor at above-average prior.
- **What closed it (the four scenario changes):**
  1. **Subject placement** — relocated to a real findable cell: `REAL_TERRAIN_SUBJECT_OFFSET =
     (16, -16)` = cell (93, 80), ~1.13 km NW, tree cover, ~2.6× the map-mean prior (chosen by
     scanning the DEM+WorldCover for tree + accessible + above-avg-prior near the LKP). Added via
     an `offset` **seam** on `subject_cell`/`build_scripted_path` so the synthetic default (and
     its test) are untouched — the two scenarios no longer share one tuning.
  2. **Thermal floor** — `BrainConfig.thermal_detection_floor` (default 0.0 = off; showcase 0.8),
     applied to THERMAL only in `detector_sim`, encodes that heat penetrates canopy gaps at dusk
     so the thermal pass reliably corroborates a forested subject.
  3. **Flight path** — `build_scripted_path` generalized to a direction-agnostic sweep (flies the
     LKP→subject band whichever way it lies; reduces to the old SW path for the synthetic default).
  4. **Threshold** — `located_concentration_ratio` recalibrated for real terrain in
     `SHOWCASE_CONFIG` (3.5: above the subject's ~2.6× prior baseline so the gate discriminates,
     below the thermal-corroborated find peak ~5.3–6.8× so it locates with margin; persistence
     n=3 is the false-positive guard). Empirically locates 6/6 seeds at 0-cell error.
- **Default stays synthetic** (`python -m src.demo.run`) so the original demo locates out of the
  box; the real-terrain showcase is `python -m src.demo.showcase`.

### B4. Georeferencing fidelity & real terrain (separate tracks, not the brain)
- The prior uses a **synthetic** terrain stub (a hand-shaped ridge + drainage) and
  **Euclidean** distance from the LKP. Real DEM/land-cover/OSM rasters and a Tobler
  cost-distance are upgrades behind the existing seams (`TerrainProvider`,
  `distance_from_lkp_m`). The GeoReferencer (frame→ground projection) is the next build
  piece; until it exists the brain runs on mock `Observation`s.

---

## C. Post-demo enhancements (C1 built; C2 designed, not built)

### C1. Sectorized "grid search" + multi-drone coordination
A search-planning layer that divides the area into coarse **segments** and tasks search by
each segment's aggregated probability. Important framing: this is **additive — it sits on the
search-director seam, not inside the Bayesian core.** It does NOT change the update, the
located trigger, or the single-writer `MapState`.

- **Coarse search grid, fine belief.** Overlay a coarse grid (e.g. 500 m segments = 10×10 of
  the 50 m probability cells). **The posterior stays fine-grained** — segmentation is a
  *tasking* overlay only; never coarsen the belief (it throws away information).
- **Aggregated POA per segment** = Σ posterior over the segment's cells (optionally weighted by
  `(1 − mean coverage)` so already-searched segments rank lower). Rank segments by POA. This is
  standard SAR sectorization / probability-of-area practice.
- **Single drone:** pick the top-POA segment, sweep it (boustrophedon), fold the observations
  into the map, re-rank, advance. Generalizes today's one-line
  `next_target = argmax(posterior × (1 − coverage))` (`SearchBrain._next_target`) into a
  segment-level planner that emits `MapState.search_path / next_target` (interfaces §6.1).
- **Multi-drone:** assign disjoint top-POA segments to drones (lock a segment to one drone so
  two drones never cover the same ground at once), and re-balance when a detection flags a
  segment (a drone diverts/returns). This is a **planner/assignment layer**; the brain just
  receives N `Observation` streams into the same single-writer state, and the coverage layer
  already tracks "where we've looked" globally.
- **Honest note on "efficiency":** for a *single* drone, greedy probability-guided search is
  already about as efficient as it gets. The grid's real payoff is **multi-drone coordination
  (no overlap)** and **operator legibility** (named, assignable sectors), not single-drone speed.
- **Invariants preserved:** single-writer `MapState`, fine-grained Bayesian belief, the located
  trigger. The new logic is a consumer/producer around the map, not a rewrite of it.
- **Status:** ✅ BUILT (`src/search/planner.py`, `src/demo/search_demo.py`). All the framing
  above held as designed. As-built notes:
  - `SectorPlanner` is a **pure, read-only** search director: `SectorGrid` (coarse overlay, human
    names A1/C4), `sector_poa` (POA = Σ posterior; priority = POA × (1 − mean coverage), vectorized
    via `np.add.reduceat`), `rank_sectors`, `plan_single` (boustrophedon sweep), `assign_multi`
    (disjoint top-sector → nearest-free-drone, locked).
  - `MapState.search_path` (the ratified §6.1 field) is now emitted by the brain — **additively**;
    `next_target`'s legacy behavior is unchanged.
  - **Closed-loop demo** (`run_single_drone` / `run_multi_drone`, one unified per-tick driver):
    the flight path **emerges** from the planner — sweep the top-POA sector → re-rank → advance →
    locate. Multi-drone runs N streams into the **one** brain (single writer); a per-tick assign
    guarantees **disjoint** sectors (verified: zero overlap frames across seeds). Locates 0-cell
    across seeds, single and 3-drone.
  - Two real findings from running it: edge-inclusive sweep coverage is needed (a subject on a
    sector boundary is otherwise grazed by one marginal pass), and the demo sweeps with **thermal**
    (the canopy-penetrating sensor) at a generous floor — detection realism is the showcase's job,
    this demo's subject is **coordination**.
  - Run: `python -m src.demo.search_demo [--drones N]`. Tests: `tests/test_planner.py`,
    `tests/test_search_demo.py`.
- **Not built (future):** detection-triggered re-balance is modest (the detecting drone confirms
  in place rather than vectoring a second drone in); fine for the demo.

### C2. contextily real-map-tiles backdrop (showcase "secondary visual path")
The real-terrain showcase (`src/demo/showcase.py`) renders the posterior over a **DEM hillshade**
backdrop — chosen for zero new deps, offline determinism, and because the shaded relief visually
*explains* the prior (you see the ridge/drainage the search hugs). A nicer-looking alternative is
**`contextily`** real map tiles (OSM streets or satellite imagery) for a "Google-Maps-credible"
backdrop.
- **Why deferred, not split:** `contextily` adds a dependency AND a render-time network fetch (a
  flaky point for a live demo), and aligning the posterior to Web-Mercator tiles needs a
  reprojection the hillshade route avoids (the hillshade is already in the grid's cell frame). We
  deliberately committed to **one** route (hillshade) rather than splitting effort.
- **If picked up:** add it as a *secondary* render path behind a flag (e.g. `--basemap=tiles`),
  reusing the same `FrameState`s and markers; the only new work is fetching tiles for the AOI bbox
  (`GridSpec.cell_to_latlon` on the corners) and reprojecting the posterior overlay to match.
- **Status:** optional polish — pick up only if spare time appears; the hillshade route is the
  committed one.
