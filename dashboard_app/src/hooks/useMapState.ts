// =============================================================================
// useMapState.ts
// -----------------------------------------------------------------------------
// Responsible for: Feeding the dashboard the LIVE MapState from the brain's integration
//                  server, polling GET /state, and falling back to the bundled mockState
//                  whenever the server is offline or unreachable.
// Role in project: The dashboard side of the brain<->dashboard seam. The app was 100%
//                  driven by a static mockState; this hook makes the swap ADDITIVE — the UI
//                  renders live data when the server is up and the exact same mock when it
//                  isn't, so the dashboard never goes blank in a demo if the backend hiccups.
// Assumptions: The server's /state returns a UI-MapState matching ../types.ts (produced by
//              integration/dashboard_projection.py). CORS is configured server-side for the
//              Vite dev origin. The API base is overridable via VITE_API_BASE for deploys.
// =============================================================================

import { useEffect, useState } from 'react'
import type { MapState } from '../types'
import { mockState } from '../data/mockState'
import { API_BASE } from '../lib/api'

// Poll cadence. ~1s is plenty: the loop advances at ~0.7s/frame, and the read model is a
// cheap snapshot — we just want the map to visibly evolve, not stream every micro-update.
const POLL_MS = 1000

/**
 * Subscribe to the live brain MapState, with a mockState fallback.
 *
 * Why a hook (not a one-shot fetch): the dashboard must keep re-rendering as the search
 * progresses, so we poll on an interval and push each fresh snapshot into React state. On any
 * fetch error we simply keep the last good value (which starts as the mock), so an offline
 * server degrades to the static demo instead of an empty screen.
 *
 * @returns `{ state, live }` — the most recent MapState (live data once the server responds,
 *   else the mock), and `live`: whether the most recent poll actually reached the brain. The
 *   UI uses `live` to flag when the map is showing demo data instead of a real search.
 */
export function useMapState(): { state: MapState; live: boolean } {
  // Start from the mock so the very first render is a complete, valid dashboard even before
  // the first successful poll (or with no server at all).
  const [state, setState] = useState<MapState>(mockState)
  // Optimistic: assume connected so a healthy server never flashes an "offline" badge during
  // the first poll; it flips to false only when a poll actually fails.
  const [live, setLive] = useState(true)

  useEffect(() => {
    // `active` guards against a late fetch resolving after the component unmounts.
    let active = true

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/state`)
        if (!res.ok) {
          if (active) setLive(false) // server reachable but erroring -> treat as not live
          return // keep last good state
        }
        const data = (await res.json()) as MapState
        // Guard against an empty/garbage body overwriting a good state.
        if (active && data && Object.keys(data).length > 0) {
          setState(data)
          setLive(true)
        }
      } catch {
        // Network error / server down -> stay on the last good state (mock fallback).
        if (active) setLive(false)
      }
    }

    poll() // fetch immediately so we don't wait a full interval for the first live frame
    const id = setInterval(poll, POLL_MS)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  return { state, live }
}
