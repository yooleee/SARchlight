# Showcase Kickoff — Real-Terrain Animated Demo (brain track)

**Start-here guide for the next work session.** Goal: a standalone **visual showcase of the
brain track** — an animated walkthrough (GIF/MP4) of the search loop running on **real Marin
terrain**, the probability map overlaid on a real basemap, going prior → sweep → detection →
located. This is for showing a mentor/reviewer "what our track does," distinct from the
teammate's production Streamlit dashboard.

> Read this first, then `docs/brain_followups.md` (B5 real-terrain, C1 grid-search) for the why.
> Explain your approach before building (per `~/.claude/CLAUDE.md`).

---

## ✅ BUILT (2026-06-20) — all three parts done
The real-terrain animated showcase is built and tested (`src/demo/showcase.py`,
`tests/test_showcase.py`; 101 tests green). Run it: `python -m src.demo.showcase` →
`demo_output/showcase.gif` (+ `showcase_located.png`). The loop now **locates on real Marin
terrain at 0-cell error across seeds** (closes `brain_followups.md` B5).
- **Part 1** — subject relocated NW (`REAL_TERRAIN_SUBJECT_OFFSET`, an offset *seam* so synthetic
  is untouched), thermal canopy floor (`BrainConfig.thermal_detection_floor`), direction-agnostic
  sweep, threshold recalibrated (`SHOWCASE_CONFIG.located_concentration_ratio = 3.5`).
- **Part 2** — DEM **hillshade** backdrop (zero-dep, chosen over `contextily` — see
  `brain_followups.md` C2 for the deferred tiles route), posterior drawn with a graded alpha.
- **Part 3** — GIF via direct PIL frames with per-frame durations (dwell on prior + climax).

The notes below are the original kickoff brief, kept for context.

---

## Where we are
The brain/geo/terrain track is **built, tested, committed** (branch `brain`): prior + Bayesian
update + located trigger + single-writer `MapState`, GeoReferencer (projection + boundary
defense), real-raster terrain (DEM + WorldCover, `RasterTerrain`), a detector *simulator*, and a
feasibility harness — 95 tests, runs end-to-end (`python -m src.demo.run`), locates at 0 m on
**synthetic** terrain. **Teammates own the production dashboard, voice/broadcast, and real
YOLO+SAHI detector** — not ours to build.

Known gap this work closes: on **real** terrain the loop does **not** yet locate (the AOI is
~72% dense canopy → a realistic subject is hard to detect and sits at ~average prior). See
`brain_followups.md` B5.

---

## The build (three parts)

### Part 1 — Make the loop LOCATE on real terrain (scenario re-design) — DO THIS FIRST
1. **Subject placement** — keep it in **tree cover** (realistic; preserves the canopy/thermal
   story) but **closer to the LKP** (~1–1.3 km) for higher prior. Pick the cell by searching
   real terrain for tree-cover + accessible + above-average-prior (the prior peak is grassland
   NW; tree cells near the LKP have decent prior). Set via `_SUBJECT_OFFSET_*` in
   `src/demo/mock_stream.py`.
2. **Thermal effective under canopy** — make thermal detection less gated by canopy (defensible
   for the dusk scenario: heat penetrates gaps). In `src/demo/detector_sim.py`, give thermal a
   detection *floor* (e.g. `p_detect = max(base_vis*factor, thermal_floor≈0.8)`) so it reliably
   hits a forested subject. This is the realistic premise that makes the canopy story *work*.
3. **Redirect the flight path** — the sweep + overflight in `build_scripted_path` must fly over
   the new subject (currently hard-wired SW); generalize so the path follows the subject.
4. **Recalibrate** `located_concentration_ratio` for real terrain (real-terrain find is
   ~2.9–3.2× vs ~1.4× prior baseline; pick a floor with margin, lean on persistence as the
   false-positive guard). Verify it locates robustly across seeds.

### Part 2 — Basemap overlay
Render the probability map **semi-transparent over a real basemap** of the Marin AOI.
- Use **`contextily`** to fetch a static basemap tile for the AOI bbox; draw it as the
  background, then `imshow` the posterior with `alpha` on top, aligned to the same extent.
- AOI bbox (lat/lon) from `GridSpec` corners (`cell_to_latlon`). Keep the markers (LKP, subject,
  flown path, next-target) from the existing `_annotate_axis` in `src/demo/run.py`.

### Part 3 — Animation
Render **one frame per observation** into an animated **GIF** (default — no ffmpeg) or **MP4**.
- New `src/demo/showcase.py`: drive the real-terrain loop (reusing `build_scripted_path`,
  `RasterTerrain`, `GeoReferencer`, `DetectorSimulator`, `SearchBrain`), render each frame
  (basemap + posterior + markers + beat caption + a small stat strip), write the animation.
- matplotlib `PillowWriter` (GIF, bundled) or `imageio`; ~40–46 frames → a few seconds.
- Beat captions: prior / sweeping / detection / **LOCATED** + the broadcast line.

---

## Libraries (justify before adding, per CLAUDE.md)
- **`contextily`** — static basemap tiles for a bbox, as a matplotlib background. *Why:* a
  real-map backdrop for credibility, rendered as static frames. *Simpler alt:* reuse our own
  WorldCover/DEM raster as the backdrop (zero new deps) — consider if avoiding dependencies
  matters; `contextily` looks nicer. *Rejected:* folium (interactive; we need static frames).
- **GIF** via matplotlib `PillowWriter` (no new dep). `imageio`/ffmpeg only if MP4 wanted.

## Files
- **New:** `src/demo/showcase.py`.
- **Modify:** `src/demo/mock_stream.py` (subject + generalized path), `src/demo/detector_sim.py`
  (thermal floor), `src/common/config.py` (real-terrain threshold + thermal-floor tunable),
  `requirements.txt` (+contextily).
- **Reuse:** `render_state` / `_annotate_axis` (`run.py`), `RasterTerrain` (`terrain_raster.py`),
  the whole loop wiring. Update `brain_followups.md` B5 once real terrain locates.

## Verification
- Real-terrain loop **locates** the subject (seed 0 + a few seeds). Add an acceptance test
  mirroring `tests/test_geo_integration.py::test_full_chain_locates_the_planted_subject` but on
  `RasterTerrain`.
- `python -m src.demo.showcase` produces a watchable GIF/MP4 (prior → clear → spike → located)
  over a real Marin basemap.
- Full suite stays green.

---

## Reference (already done / located)
- **Grid-search / multi-drone enhancement:** documented in `brain_followups.md` **C1** (post-demo,
  additive search-planning layer; not built).
- **Detection datasets (if ever needed):** `/Users/yoolee/Developer/sar_system/data/` —
  `detection/heridal` (7.8 GB), `detection/sard`, `detection/wisard` (color + thermal sequences).
  **Not needed for this showcase** (the detector is simulated); reference the path, don't copy.
