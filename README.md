# SAR-system
AI Berkeley 2026 Hackathon project — wide-area Search & Rescue support.

A drone-side perception + decision layer for finding missing people in remote
terrain. The center of gravity is the **probability map** (the "brain"): a prior of
where the subject likely is directs the search; detections and clean non-detections
update the map; the updated map redirects the next search; a confident, persistent
detection trips a `located` event.

## The brain (`src/`)

The search-map brain is built and runnable today on a mock Observation stream (before
the real GeoReferencer/detector exist). See `docs/interfaces.md` §4–§6 for the
contracts and the Bayesian math, and `docs/prior_model.md` for the prior.

```
src/
├── common/    GridSpec, the ratified contracts (Observation, MapState, LocatedEvent), config
└── search/    terrain stub, prior, the Bayesian update, the located trigger, the brain loop
src/demo/      a scripted Observation stream + a runner (console beats + heatmap PNGs)
tests/         unit tests for the update, prior, and trigger + a brain integration test
```

## Setup & run

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

.venv/bin/python -m pytest                       # run the test suite
.venv/bin/python -m src.demo.run                 # demo on synthetic terrain (locates) -> demo_output/
.venv/bin/python -m src.demo.run --real-terrain  # same, on the real DEM+WorldCover Marin prior
```

The pipeline is `scripted CameraPose → detector simulator → GeoReferencer → brain`. The
detector is **simulated** (we have no aerial footage); geo and the brain are the real code.
`RasterTerrain` builds the prior + visibility from the real rasters; the demo's scenario is
tuned for the synthetic stub, so `--real-terrain` shows the real Marin prior but may not
locate (see `docs/brain_followups.md`).

The demo drives the loop through the `docs/demo_scenario.md` sequence: the prior blooms
SW down the drainage from the last-known position, a sweep clears the corridor, color
detections brighten the subject (status stays `searching`), and the thermal
corroboration trips `located`. All tunables live in `src/common/config.py`.

## Robustness

The brain is hardened against the messy input a real detector/GeoReferencer can produce:

- **Input sanitization** (`src/search/validation.py`) at the `Observation` boundary —
  NaN/out-of-range/off-grid values are clamped or dropped (sanitize-and-log, never crash
  or poison the map), with a per-frame report and a `SearchBrain.input_warning_count`.
- **Relative `located` gate** — a grid-stable concentration ratio (window density vs map
  average) backs up the absolute mass threshold, so the trigger doesn't depend on a
  hand-tuned number that only fits one grid.
- **Regression coverage** (`tests/test_robustness.py`) — false-positive recovery, multiple
  subjects, intermittent detections, off-grid data, edge subjects, plus seeded soak/fuzz
  runs asserting the mass invariant holds and no NaN ever reaches the posterior.

Deferred follow-ups and honest limitations (e.g. the stationary-subject assumption) are
tracked in `docs/brain_followups.md`. Run coverage with
`.venv/bin/python -m pytest --cov=src --cov-report=term-missing`.
