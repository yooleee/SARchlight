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

### A1. Cumulative-clearance cap per (cell, sensor)
- **What:** `coverage` compounds as `cleared_i = 1 − Π(1 − d_i)` (see the `TODO` at the
  coverage line in `src/search/update.py`). This assumes **independent** looks.
- **Why deferred:** repeated passes from the same angle/sensor miss *identically* under
  canopy, so the product **overstates** clearance (interfaces.md §5.5). The right cap
  depends on how often real flights re-look at a cell from the same sensor and how
  correlated those misses are — numbers we can only get from real coverage patterns.
  Tuning it now against the synthetic mock stream would calibrate against fiction.
- **Trigger to do it:** once the real GeoReferencer is producing footprints and we can
  observe actual re-look frequency/overlap. Likely a `cap` per `(cell, sensor_type)`.
- **Seam:** already isolated to the one coverage-compounding line; no interface change.

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

### B4. Georeferencing fidelity & real terrain (separate tracks, not the brain)
- The prior uses a **synthetic** terrain stub (a hand-shaped ridge + drainage) and
  **Euclidean** distance from the LKP. Real DEM/land-cover/OSM rasters and a Tobler
  cost-distance are upgrades behind the existing seams (`TerrainProvider`,
  `distance_from_lkp_m`). The GeoReferencer (frame→ground projection) is the next build
  piece; until it exists the brain runs on mock `Observation`s.
