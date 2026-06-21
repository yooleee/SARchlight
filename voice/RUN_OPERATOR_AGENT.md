# Operator Voice Agent — Setup & Run Guide

> **Branch:** `voice-operator`
> **What this is:** the SAR *operator telephony agent*. A field operator speaks to the
> system (by mic locally, or a real phone number once deployed) and asks about the active
> search — "what's the status?", "where should we search?", "how much have we covered?",
> "did we find them?". A Deepgram Voice Agent answers from live search state, and the
> conversation streams to the dashboard's **Live Transcript** panel in real time.

If you're driving this from a Claude Code session: this file is the runbook. The whole
demo runs **locally**, and the agent answers from a **built-in snapshot** when the brain
isn't running — so you do **not** need the terrain rasters or the integration server to
see it work. Read "Architecture" at the bottom for the lay of the land.

---

## TL;DR (fastest path — agent only, no rasters)

```bash
git fetch origin && git checkout voice-operator

# venvs (main = brain; voice = the agent, heavy deps kept separate)
python3 -m venv .venv         && .venv/bin/python -m pip install -r requirements.txt
python3 -m venv voice/.venv   && voice/.venv/bin/pip install -r voice/requirements-dev.txt
# macOS: brew install portaudio    Linux: sudo apt-get install portaudio19-dev   (for the mic client)

# dashboard
cd dashboard_app && npm install
printf 'VITE_AGENT_WS_URL=ws://localhost:8080/transcript\n' > .env && cd ..
```

Then three terminals (run the dashboard and open it **before** you speak):

```bash
cd voice && .venv/bin/python main.py          # T1: the agent (serves /transcript on :8080)
cd voice && .venv/bin/python dev_client.py     # T2: talk to it via your mic
cd dashboard_app && npm run dev                # T3: the dashboard
```

Speak: *"What's the status?"* · *"Where should we search?"* · *"Did we find them?"*
→ you should **hear** SAR answers and **see** the turns appear in Live Transcript.

---

## Prerequisites

- Python 3.12, Node 18+ / npm
- A `DEEPGRAM_API_KEY` — **you already have `voice/.env`** with this (and the server info), so
  no secret setup needed. (See the local-dev caveat just below.)
- PortAudio (only for the mic client `dev_client.py`):
  macOS `brew install portaudio` · Debian/Ubuntu `sudo apt-get install portaudio19-dev`

### ⚠️ One local-dev caveat about `voice/.env`

Your `voice/.env` includes server/Twilio info. For the **local mic demo**, make sure
`WEBHOOK_SECRET` and `SERVER_EXTERNAL_URL` are **unset or commented out**:

```bash
# in voice/.env, these two should be commented for local dev:
# WEBHOOK_SECRET=...
# SERVER_EXTERNAL_URL=...
```

Why: when `WEBHOOK_SECRET` is set, the `/twilio` WebSocket requires a path token, and
`dev_client.py` connects **without** one → the agent closes it with code 1008 and the mic
demo silently won't work. `voice/setup.py` re-adds both automatically when you deploy, so
commenting them out only affects local dev. (`DEEPGRAM_API_KEY` stays set.)

---

## One-time setup (details)

**Two virtual environments, on purpose.** The brain/integration server runs on the repo-root
`.venv`; the voice agent runs on its own `voice/.venv` (it pulls in the Deepgram SDK,
`sounddevice`, etc.). Keep them separate.

```bash
# Brain (integration server + tests)
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Voice agent (+ the mic dev client)
python3 -m venv voice/.venv
voice/.venv/bin/pip install -r voice/requirements-dev.txt
```

**Dashboard env (gitignored — create it):**

```bash
cd dashboard_app
npm install
printf 'VITE_AGENT_WS_URL=ws://localhost:8080/transcript\n' > .env
cd ..
```

That points the dashboard's Live Transcript at your local agent. (Remove the line to fall
back to the deployed Fly agent's default URL.)

---

## Running the demo

### Tier A — agent only (simplest; answers from the snapshot, no rasters)

```bash
cd voice && .venv/bin/python main.py          # T1
cd voice && .venv/bin/python dev_client.py     # T2  (needs PortAudio + a mic)
cd dashboard_app && npm run dev                # T3  (open it before speaking)
```

The agent can't reach a brain, so it answers from a **truthful built-in snapshot** of the
demo outcome (subject found in the trees near the Pantoll trailhead, ~2% covered, 3 drones).

### Tier B — with the live brain (optional; needs the terrain rasters)

The integration server builds the real-terrain Marin run at startup, so it needs two
gitignored rasters in `data/terrain/`:
`dem_marin_usgs10m.tif` and `worldcover_2021_N36W123.tif` (~102 MB).
**Fastest: copy the two `.tif` files from Yousuf.** (Provenance / re-download steps are in
`docs/data.md`.) Then add a terminal **before** the agent:

```bash
.venv/bin/uvicorn integration.server:app --port 8000   # T0: brain + GET /ops
```

Now the agent reads live state from `http://localhost:8000/ops` automatically (override with
`SAR_STATE_URL`), so its answers change as the run progresses — coverage climbs, and
"not yet" flips to "found" with the real location once the brain declares it.

---

## What success looks like

- **T1 (agent)** logs `[TELEPHONY] WebSocket connected` when `dev_client` joins, then
  `USER:` / `AGENT:` transcript lines as you talk.
- You **hear** answers like *"The highest-probability area to search is near the Pantoll
  trailhead."*
- The dashboard **Live Transcript** shows user/assistant bubbles updating live.

---

## Automated tests

```bash
# No rasters, no Deepgram, no venv-juggling — pure logic (14 tests):
.venv/bin/python -m pytest tests/test_sar_service.py tests/test_transcript_hub.py -q

# Full suite (190). The /ops + integration tests SKIP without the rasters — that's expected.
.venv/bin/python -m pytest -q
```

`test_sar_service.py` covers the agent's answer logic (live `/ops` read + snapshot fallback);
`test_transcript_hub.py` covers the transcript bridge (including the `agent`→`assistant`
role mapping the dashboard expects).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Mic demo: agent never logs "WebSocket connected"; nothing happens | `WEBHOOK_SECRET` is set in `voice/.env` → tokenless `/twilio` is rejected (1008). Comment it out for local dev (see caveat above). |
| Transcript panel never resets / no bubbles | Open the dashboard **before** starting a call. `call_started` clears + activates the panel and isn't replayed to late joiners. |
| Agent answers feel "static" | Expected in Tier A — it's the snapshot. Run Tier B (integration server on :8000) for live, changing answers. |
| `dev_client.py` errors importing `sounddevice` | PortAudio missing — `brew install portaudio` (macOS) / `apt-get install portaudio19-dev` (Linux). |
| `integration.server` import errors about a DEM | You're missing the rasters — that's Tier B only. Use Tier A, or get the `.tif` files. |
| Many tests show `skipped` | Normal without the rasters. The two voice test files above always run. |

---

## Architecture (key files)

```
operator speaks ──► Deepgram Voice Agent ──► function call ──► sar_service ──► GET /ops (or snapshot)
                          │                                                          ▲
                          └──► transcript turn ──► transcript_hub ──► /transcript WS ─┴─► dashboard Live Transcript
```

- `voice/voice_agent/agent_config.py` — the SAR dispatcher persona + the 4 read-only voice
  tools (`get_search_status`, `get_highest_probability_area`, `get_coverage`,
  `get_located_status`) + `end_call`. Think model `gpt-4o-mini`, voice = Deepgram (Flux STT +
  Aura TTS).
- `voice/backend/sar_service.py` — turns search state into short, *speakable* answers. Reads
  `SAR_STATE_URL` (default `http://localhost:8000/ops`); **falls back to a built-in snapshot**
  when the brain is unreachable (e.g. deployed off-box).
- `voice/transcript_hub.py` + `voice/main.py` (`/transcript` route) + publish hooks in
  `voice/voice_agent/session.py` — the in-process bus that streams the live conversation to
  the dashboard (maps Deepgram's `agent` role to the dashboard's `assistant`).
- `integration/server.py` — the additive `GET /ops` endpoint (voice-friendly facts derived
  from the per-frame `MapState`; reports locations as landmarks, not raw coordinates).
- `voice/dev_client.py` — local mic/speaker client; pretends to be Twilio so the server can't
  tell the difference.

---

## Optional — cloud deploy (real phone number via Fly + Twilio)

The code is deploy-ready. `voice/fly.toml` targets app `golden-seastar-977` and keeps one VM
warm (`min_machines_running = 1`); `voice/setup.py` is the Fly + Twilio wizard.

- **If you own the `golden-seastar-977` Fly app:** `cd voice && fly deploy` ships this branch.
  Its secrets and the Twilio webhook are already configured, and the dashboard's built-in
  default already points at `wss://golden-seastar-977.fly.dev/transcript` — zero rewiring.
- The deployed agent can't reach a localhost brain, so it answers from the snapshot (truthful
  for the demo scenario) unless you point `SAR_STATE_URL` at a tunnel.

(Yousuf's Fly account can't deploy to `golden-seastar-977` — it's under your account — which
is why the cloud deploy is left to you.)
