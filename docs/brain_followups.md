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

### B5. Real terrain makes the find harder — demo scenario needs re-design
- **Status:** `RasterTerrain` (real DEM + WorldCover) is built and tested (Geo Unit 5) and
  the demo can run on it via `python -m src.demo.run --real-terrain`. But the demo's
  scenario (subject placement, scripted flight path, threshold) is tuned for the SYNTHETIC
  stub, and on **real** terrain it does **not** reliably locate.
- **Why:** the AOI is ~72% dense tree canopy, so a realistic subject has low visibility
  (~0.4 → ~40% detection, marginal persistence), AND a realistic subject sits at ~average
  prior, so its concentration (~3x) overlaps with false positives (~3.5x). The synthetic
  demo hid this by placing the subject in a favorable corridor at above-average prior.
- **What's needed (folded into the advanced-demo work):** relocate the subject to a
  findable real cell, redirect the flight path to match, make thermal detection effective
  under canopy (defensible for the dusk scenario), and recalibrate the threshold. This is
  scenario design, which overlaps the planned advanced visual demo — see that effort.
- **Default stays synthetic** so the demo locates reliably out of the box.

### B4. Georeferencing fidelity & real terrain (separate tracks, not the brain)
- The prior uses a **synthetic** terrain stub (a hand-shaped ridge + drainage) and
  **Euclidean** distance from the LKP. Real DEM/land-cover/OSM rasters and a Tobler
  cost-distance are upgrades behind the existing seams (`TerrainProvider`,
  `distance_from_lkp_m`). The GeoReferencer (frame→ground projection) is the next build
  piece; until it exists the brain runs on mock `Observation`s.

---

## C. Post-demo enhancements (designed, not built)

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
- **Status:** post-demo. Captured here so the design intent isn't lost.
