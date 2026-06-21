import { Plus, Minus, Layers, Navigation, Home } from 'lucide-react'
import type { MapState } from '../types'
import { API_BASE } from '../lib/api'

// The map is now HYBRID: a server-rendered unified base image (terrain + graded-alpha posterior
// + sectors, GET /map_base.png) under live crisp VECTOR overlays (the drone fleet, their trails,
// and the guide-home route/markers) driven by /state. Both are keyed to `state.frame`, so the
// image and the vectors are always the same frame and never drift.

interface Props {
  state: MapState
}

export default function ProbabilityMap({ state }: Props) {
  const guiding = state.guidanceStatus === 'guiding' || state.guidanceStatus === 'arrived'
  const drones = state.drones ?? []

  // SVG path string for a normalized polyline (the drone trails + the guide route).
  const toPath = (pts: { x: number; y: number }[]) =>
    pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x * 100} ${p.y * 100}`).join(' ')

  const guidanceD = state.guidancePath && state.guidancePath.length ? toPath(state.guidancePath) : ''

  return (
    <div className="panel relative flex-1 overflow-hidden">
      {/* Dark fallback while the base image loads (or if the server is offline). */}
      <div className="absolute inset-0 bg-base-950" />

      {/* The unified base image: terrain + posterior + sectors, rendered server-side per frame.
          Keyed to state.frame so it stays in lockstep with the vector overlays below. object-fill
          stretches the square base to the panel; the vectors use the same % frame, so they align. */}
      <img
        src={`${API_BASE}/map_base.png?v=${state.frame ?? 0}`}
        alt=""
        draggable={false}
        decoding="async"
        className="absolute inset-0 h-full w-full object-fill"
      />

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

      {/* Compass */}
      <div className="absolute right-3 top-3 grid h-9 w-9 place-items-center rounded-full bg-base-900/80 text-[10px] font-bold text-slate-300 ring-1 ring-white/10">
        <span className="text-accent-red">N</span>
      </div>

      {/* Zoom + layers controls */}
      <div className="absolute right-3 top-16 flex flex-col gap-1.5">
        <CtrlBtn><Plus className="h-4 w-4" /></CtrlBtn>
        <CtrlBtn><Minus className="h-4 w-4" /></CtrlBtn>
        <CtrlBtn><Layers className="h-4 w-4" /></CtrlBtn>
      </div>

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

function CtrlBtn({ children }: { children: React.ReactNode }) {
  return (
    <button className="grid h-8 w-8 place-items-center rounded-md bg-base-900/80 text-slate-300 ring-1 ring-white/10 hover:bg-base-700">
      {children}
    </button>
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
