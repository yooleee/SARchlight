import { useState } from 'react'
import { Navigation, Home, WifiOff } from 'lucide-react'
import type { MapState } from '../types'
import { API_BASE } from '../lib/api'

// The map is now HYBRID: a server-rendered unified base image (terrain + graded-alpha posterior
// + sectors, GET /map_base.png) under live crisp VECTOR overlays (the drone fleet, their trails,
// and the guide-home route/markers) driven by /state. Both are keyed to `state.frame`, so the
// image and the vectors are always the same frame and never drift.

interface Props {
  state: MapState
  // False when the integration server is unreachable (the map is showing demo data, not a live search).
  live: boolean
}

export default function ProbabilityMap({ state, live }: Props) {
  const guiding = state.guidanceStatus === 'guiding' || state.guidanceStatus === 'arrived'
  const drones = state.drones ?? []

  // SVG path string for a normalized polyline (the drone trails + the guide route).
  const toPath = (pts: { x: number; y: number }[]) =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x * 100} ${p.y * 100}`).join(' ')

  const guidanceD = state.guidancePath && state.guidancePath.length ? toPath(state.guidancePath) : ''

  // Double-buffer the per-frame base image to kill the inter-frame flash. The base map is a
  // server-rendered PNG fetched fresh each frame (?v=frame); naively swapping the <img> src
  // leaves a visible gap while the new PNG downloads + decodes. Instead we keep showing the
  // current frame (`shownSrc`) and load the incoming frame in a hidden <img>; only when THAT
  // finishes (onLoad) do we promote it. The visible swap is then to an already-decoded image,
  // so the animation is smooth instead of a slideshow.
  const baseSrc = `${API_BASE}/map_base.png?v=${state.frame ?? 0}`
  const [shownSrc, setShownSrc] = useState(baseSrc)

  return (
    <div className="panel relative flex-1 overflow-hidden bg-base-950">
      {/* The map content is a centered SQUARE (matching the demo + the square base image), so the
          square base is never stretched into the panel's rectangle (no distortion, sharper). The
          vectors use % of THIS square, so they stay pixel-aligned to the base; the panel-corner
          chrome (compass/legend/...) sits outside the square. */}
      <div className="absolute inset-0 grid place-items-center">
        <div className="relative aspect-square h-full max-w-full">
          {/* The unified base image: terrain + posterior + sectors, rendered server-side per frame,
              keyed to state.frame. A square image in a square box -> no stretch. `shownSrc` is the
              last FULLY-LOADED frame, so the visible image never blanks mid-swap. */}
          <img
            src={shownSrc}
            alt=""
            draggable={false}
            decoding="async"
            className="absolute inset-0 h-full w-full"
          />
          {/* Hidden preloader for the incoming frame: promote it to visible only once decoded. */}
          {baseSrc !== shownSrc && (
            <img
              src={baseSrc}
              alt=""
              aria-hidden
              className="hidden"
              onLoad={() => setShownSrc(baseSrc)}
            />
          )}

      {/* Vector overlays (trails + route + tether). */}
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
        {/* Each drone's flown trail, in its (sector-matching) color. */}
        {drones.map((d) => (
          <path
            key={d.id}
            d={toPath(d.path)}
            fill="none"
            stroke={d.color}
            strokeWidth={0.5}
            strokeLinecap="round"
            vectorEffect="non-scaling-stroke"
            opacity={0.9}
          />
        ))}

        {/* Guide-home: the planned walkable route home (dark casing + green dashes). */}
        {guiding && guidanceD && (
          <>
            <path d={guidanceD} fill="none" stroke="rgba(0,0,0,0.55)" strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
            <path d={guidanceD} fill="none" stroke="#7CFC00" strokeWidth={0.7} strokeDasharray="1.6 1.2" vectorEffect="non-scaling-stroke" />
          </>
        )}

        {/* Line-of-sight tether: subject (follower) -> the drone (leader = dronePos). */}
        {guiding && state.subjectPos && (
          <line
            x1={state.subjectPos.x * 100}
            y1={state.subjectPos.y * 100}
            x2={state.dronePos.x * 100}
            y2={state.dronePos.y * 100}
            stroke="gold"
            strokeWidth={0.5}
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>

      {/* Marker overlay (non-distorting, % positioning). */}
      <div className="absolute inset-0">
        {/* The drone fleet — a colored disc per drone (color matches its sector in the base image). */}
        {drones.map((d) => (
          <div
            key={d.id}
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${d.pos.x * 100}%`, top: `${d.pos.y * 100}%` }}
            title={`drone ${d.id}`}
          >
            <span
              className="absolute left-1/2 top-1/2 h-7 w-7 -translate-x-1/2 -translate-y-1/2 rounded-full animate-pulseRing"
              style={{ backgroundColor: d.color, opacity: 0.28 }}
            />
            <div
              className="relative grid h-5 w-5 place-items-center rounded-full ring-2 ring-white/85 shadow"
              style={{ backgroundColor: d.color }}
            >
              <Navigation className="h-3 w-3 text-base-950" style={{ transform: 'rotate(45deg)' }} />
            </div>
          </div>
        ))}

        {/* Guide-home: operators/home marker + the moving subject (follower). */}
        {guiding && state.operatorPos && (
          <div
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${state.operatorPos.x * 100}%`, top: `${state.operatorPos.y * 100}%` }}
          >
            <div className="grid h-6 w-6 place-items-center rounded-sm bg-white text-base-950 shadow ring-2 ring-black/40">
              <Home className="h-3.5 w-3.5" />
            </div>
            <span className="absolute left-7 top-1/2 -translate-y-1/2 whitespace-nowrap text-[10px] font-bold text-white drop-shadow">
              operators
            </span>
          </div>
        )}
        {guiding && state.subjectPos && (
          <div
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${state.subjectPos.x * 100}%`, top: `${state.subjectPos.y * 100}%` }}
            title="subject (following)"
          >
            {/* White-centered so it stays visible over the posterior glow it sits in. */}
            <span className="absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/40 animate-pulseRing" />
            <div className="relative grid h-4 w-4 place-items-center rounded-full bg-white ring-[3px] ring-accent-red shadow-[0_0_0_1.5px_rgba(0,0,0,0.65)]">
              <span className="h-1.5 w-1.5 rounded-full bg-accent-red" />
            </div>
          </div>
        )}
      </div>
        </div>
      </div>

      {/* Compass */}
      <div className="absolute right-3 top-3 grid h-9 w-9 place-items-center rounded-full bg-base-900/80 text-[10px] font-bold text-slate-300 ring-1 ring-white/10">
        <span className="text-accent-red">N</span>
      </div>

      {/* Offline flag: shown when the integration server is unreachable, so the bundled demo
          (mock) data the dashboard falls back to isn't mistaken for a live search. */}
      {!live && (
        <div className="absolute left-3 top-3 z-10 flex items-center gap-1.5 rounded-md bg-accent-amber/15 px-2 py-1 text-[10px] font-semibold text-accent-amber ring-1 ring-accent-amber/30">
          <WifiOff className="h-3 w-3" />
          Search server offline — demo data
        </div>
      )}

      {/* Legend */}
      <div className="absolute bottom-3 right-3 space-y-1 rounded-lg bg-base-900/85 p-2.5 text-[10px] text-slate-300 ring-1 ring-white/10">
        <LegendRow color="#38bdf8" label="Drones (per-sector)" />
        <LegendRow dashed label="Route Home" />
        <LegendRow color="#ffffff" label="Operators" />
        <LegendRow color="#ef4444" label="Subject" />
        <div className="pt-0.5 text-[9px] text-slate-400">heat = probability · grid = sectors</div>
      </div>

      {/* Scale bar */}
      <div className="absolute bottom-3 left-3 text-[10px] text-slate-300">
        <div className="flex items-center">
          <span className="mr-1">0</span>
          <div className="h-1.5 w-12 border-y border-l border-white/60" />
          <div className="h-1.5 w-12 border border-white/60 bg-white/20" />
          <span className="ml-1">500</span>
          <span className="ml-2">750 m</span>
        </div>
      </div>
    </div>
  )
}

function LegendRow({ color, label, dashed }: { color?: string; label: string; dashed?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <span className="grid w-4 place-items-center">
        {dashed ? (
          <span className="h-0 w-4 border-t border-dashed border-[#7CFC00]" />
        ) : (
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
        )}
      </span>
      {label}
    </div>
  )
}
