# Data — provenance, licensing, and layout

This folder holds **gathered data only** (datasets, terrain layers, footage, weights).
No code lives here. Large payloads are git-ignored; only this README and the `.gitkeep`
folder markers are tracked. Record provenance for everything you drop in, because
license and attribution terms matter for a public hackathon submission, and because the
map's coordinate math depends on knowing each layer's CRS and resolution.

> Per the project plan: downloading and eyeballing raw data is allowed pre-event.
> The code that ingests, reprojects, and fuses it is **building** and waits until Saturday.

When you add a dataset, fill in its row below (source URL, license, version/date, and —
for geospatial layers — CRS and resolution).

---

## `terrain/` — map prior layers for the chosen region
The DEM, land cover, and OSM layers for the one specific wilderness region we pick this
week. These feed the probability-map prior (terrain difficulty, vegetation, trails/water).

| Layer | Source | License / attribution | CRS | Resolution | Notes |
|-------|--------|-----------------------|-----|------------|-------|
| Elevation (DEM) | USGS 3DEP `USGS10m` via OpenTopography API | USGS public domain | EPSG:4269 (NAD83) | ~10 m (1/3 arc-sec, 0.0000926°) | `terrain/dem_marin_usgs10m.tif` · 2916×2700 px · already subset to the AOI bbox · downloaded 2026-06-18. |
| Land cover | ESA WorldCover 2021 **v200** (AWS Open Data `s3://esa-worldcover`, no-sign) | CC BY 4.0 — attribution required | EPSG:4326 (WGS 84) | 10 m (0.0000833°) | `terrain/worldcover_2021_N36W123.tif` · full 3° tile N36W123 (36–39 N, 123–120 W), 36000×36000 px · covers AOI, **clip Saturday** · downloaded 2026-06-18. Drives canopy/visibility weight. |
| Trails/roads/water | OpenStreetMap — Geofabrik NorCal extract | ODbL — attribution + share-alike | EPSG:4326 | vector | `terrain/osm_norcal.osm.pbf` · `norcal-260618` (2026-06-18) · layers: points/lines/multilinestrings/multipolygons/other_relations · whole NorCal region, **clip to AOI Saturday**. |

**Chosen region (primary): expanded Marin — Mt. Tamalpais → Bolinas Ridge → southern
Point Reyes.** Picked for one contiguous, easy-to-pull AOI that spans the variety the map
needs: dense redwood/Doug-fir canopy, open grassland ridges, and coastal lagoons, with a
recognizable anchor (Mt. Tam), a Berkeley-proximity pitch hook, and genuinely
remote/rugged wilderness with real SAR history toward Point Reyes.

- **Final AOI bbox (recorded 2026-06-18):** **S 37.85, N 38.10, W −122.85, E −122.58**
  (EPSG:4269/4326 lon/lat). ~24 km (E–W) × 28 km (N–S). East edge trimmed to −122.58 to drop
  urban Mill Valley/Sausalito. The DEM above (`dem_marin_usgs10m.tif`) was pulled to exactly
  these corners; WorldCover/OSM are wider and clip to this bbox Saturday. Adjustable —
  "region is config, not code," so any other region is a drop-in data swap.
- **Region is config, not code:** the AOI is just a bbox + the three pulled layers, so the
  system stays region-agnostic and any other region is a drop-in data swap.
- **Optional backup pull: Henry W. Coe State Park** (largest NorCal state park, authentic
  wide-area remoteness). Gathering its layers Friday is free pre-event prep and gives
  swap-in optionality if the remoteness story needs more punch in rehearsal — no build cost.

## `detection/` — labeled imagery for the detector
Used to evaluate, and optionally fine-tune (stretch), the person detector.

| Dataset | Size | License | Notes |
|---------|------|---------|-------|
| HERIDAL | 1,684 hi-res aerial images (4000×3000) + 1,650 VOC annotations, w/ train/val/test splits | **CC BY 3.0** — attribution; redistribution OK | `detection/heridal/heridal_keras_retinanet_voc/` (`JPEGImages/`, `Annotations/` VOC XML, `ImageSets/Main/`) · keras-retinanet PASCAL-VOC mirror (Zenodo `record/5662351`, 8.3 GB zip, integrity-checked) · official src `ipsar.fesb.unist.hr` (slow). Mediterranean non-urban, single `person` class, very small instances. Downloaded 2026-06-18. |
| SARD | 5,755 imgs (train 4041 / valid 1144 / test 570), YOLO format, 1 class `human` | **MIT** (per Kaggle) — public, **cite if published**. OK for this project. The `data.yaml` `license: Private` is Roboflow's *project-visibility* field, not a content license. | `detection/sard/search-and-rescue/` · Kaggle `nikolasgegenava/sard-search-and-rescue` → Roboflow export `datasets-pdabr/sard-8xjhy` v17 (augmented redistribution, **not** the canonical 1,981-img original). Actors simulating tired/injured people; varied terrain + weather. Also a demo-feed source. Downloaded 2026-06-18. |
| WiSARD | ~26.8k color + ~30k thermal (~15.4k synced pairs); full = 40.5 GB | **MIT-style** — redistribution OK w/ notice | `detection/wisard/` · staged the **971 MB synced-pair sample** (1 flight: `VIS_0003` 264× 3840×2160 color + `IR_0004` 264× 640×512 LWIR thermal; YOLO `.txt` labels, class 0 = person, 262/265 IR frames labeled). Full dump deferred. Enables the honest canopy/low-light + thermal story. Downloaded 2026-06-18. |
| SeaDronesSee | maritime | n/a | **Skipped** — out of scope unless water scenarios arise. |
| NOMAD | aerial, occlusion-focused | n/a | **Skipped** — out of scope unless we stress the hardest occlusion cases. |

**Derived (board eval, not source datasets):**
- `detection/coral_eval_frames/` — a curated subset for the Friday Coralboard eval: **11 full
  frames + 11 matching hand-crops** (6 HERIDAL + 3 WiSARD VIS + 2 WiSARD IR), in `full/` and
  `crop/`. Crops simulate what tiling would feed the detector. Staged 2026-06-18.
- `detection/coral_eval_out/` — **outputs** of the eval (annotated stills: ground-level `bus`
  control + aerial crop detections; the 4K `aerial_detect.mp4` run offline on the board). See its
  own `README.md` for captions. Produced 2026-06-19 by running the shipped vendor detector
  (familiarization). Evidence base for `docs/board_feasibility.md`.

> Several of these have research-use / non-commercial terms. Confirm each license before
> redistributing weights trained on them or shipping samples in the demo.

## `footage/` — demo feed
Video or frame sequences to run the detector on as a simulated live feed. SARD is
video-derived and can serve; supplement with openly licensed aerial wilderness footage.
Fallback: step through dataset still images as frames.

> Note: footage geography and the map region need **not** match — the scripted flight
> path defines the footprint over our chosen region regardless of where footage was shot.

**Staged so far (2026-06-18):** no separate footage pull yet. Planned feed sources, both already
on disk under `detection/`: **SARD** (video-derived; pending its Kaggle download) and the **WiSARD
VIS/IR frame sequences** (264 synced frames per modality — can be stepped through as a live feed).
The 4K WiSARD VIS sequence doubles as a frame-sequence feed; the IR sequence backs the thermal beat.

## `behavior/` — lost-person-behavior reference
Notes and statistics (Robert Koester's work + associated incident database) that shape
the map prior — how lost people move by terrain/category. Use this rather than inventing
movement assumptions. High confidence this is the standard reference; **verify specific
numbers before relying on them.**

## `weights/` — detector weights (git-ignored)
Pretrained weights go here first so the loop runs immediately; fine-tuned weights drop in
later. The detector swap is a **path change**, not a code change — record which file is
the active backend here.

| File | Backend | Source | Notes |
|------|---------|--------|-------|
| `weights/yolo11n.pt` | YOLO11n (pretrained, COCO) | ultralytics assets `v8.3.0` | **Default backend** for the working loop — current YOLO family, smallest variant. Downloaded 2026-06-18. |
| `weights/yolov8n.pt` | YOLOv8n (pretrained, COCO) | ultralytics assets `v8.3.0` | Alt/baseline; matches the Coral stock-model family (YOLOv8). |
| `weights/coral/yolov8n_full_integer_quant_320_od.tflite` (+ `yolov8s` variant, `_metadata_od.yaml`) | YOLOv8 INT8 320, COCO | HF `Synaptics/yolo` (**AGPL-3.0** — copyleft, don't redistribute trained-on weights without care) | **Host-side conversion inputs only.** Confirmed on-board 2026-06-19: the board **already ships** the pre-compiled stock model `yolov8n_od.vmfb` (Torq/`.vmfb`, run via IREE) at `/home/root/sl2610-examples/models/`, plus SyNAP `.synap` COCO models under `/usr/share/synap/`. So these TFLite files are needed **only if we convert our own** model on the host (host `torq-compile`/`synap convert`; not on the board). |
| _fill_ | fine-tuned (stretch) | scratch/ run | Drops in if the fine-tuning track produces it. |
