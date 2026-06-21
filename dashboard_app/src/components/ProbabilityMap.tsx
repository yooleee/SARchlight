import { useEffect, useRef, useState } from 'react'
import { Plus, Minus, Layers, Navigation, Info, X, Home } from 'lucide-react'
import type { MapState } from '../types'
import { heatColor } from '../lib/colors'
import { API_BASE } from '../lib/api'

interface Props {
  state: MapState
  showHeat: boolean
  showPath: boolean
  showDetections: boolean
  showSearched: boolean
  showTerrain: boolean
}

// Renders the probability heatmap onto a canvas by summing gaussian blobs,
// then maps the summed intensity through the heat color scale with alpha.
function useHeatmap(
  canvasRef: React.RefObject<HTMLCanvasElement>,
  blobs: MapState['heatBlobs'],
  visible: boolean,
) {
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = (canvas.width = canvas.offsetWidth)
    const H = (canvas.height = canvas.offsetHeight)
    ctx.clearRect(0, 0, W, H)
    if (!visible) return

    const STEP = 4 // downsample for performance
    const img = ctx.createImageData(W, H)
    for (let y = 0; y < H; y += STEP) {
      for (let x = 0; x < W; x += STEP) {
        let v = 0
        for (const b of blobs) {
          const dx = x / W - b.x
          const dy = y / H - b.y
          const d2 = dx * dx + dy * dy
          const r = b.radius
          v += b.intensity * Math.exp(-d2 / (2 * r * r))
        }
        v = Math.min(1, v)
        if (v < 0.06) continue
        const [r, g, bl] = heatColor(v)
        const alpha = Math.round(Math.min(0.78, v * 0.95) * 255)
        for (let yy = 0; yy < STEP && y + yy < H; yy++) {
          for (let xx = 0; xx < STEP && x + xx < W; xx++) {
            const idx = ((y + yy) * W + (x + xx)) * 4
            img.data[idx] = r
            img.data[idx + 1] = g
            img.data[idx + 2] = bl
            img.data[idx + 3] = alpha
          }
        }
      }
    }
    ctx.putImageData(img, 0, 0)
  }, [canvasRef, blobs, visible])
}

export default function ProbabilityMap({
  state,
  showHeat,
  showPath,
  showDetections,
  showSearched,
  showTerrain,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  useHeatmap(canvasRef, state.heatBlobs, showHeat)

  // Real-terrain backdrop served by the integration server. If it fails to load (server
  // offline / no rasters), fall back to the procedural backdrop underneath it.
  const [terrainOk, setTerrainOk] = useState(true)

  const pathD = state.flightPath
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x * 100} ${p.y * 100}`)
    .join(' ')

  const primary = state.detections.find((d) => d.isPrimary)

  // Guide-home phase: the drone leads the located subject back to the operators. We draw the
  // planned route, the operators/home marker, the moving subject (follower), and a line of
  // sight tether to the drone (the leader = dronePos). Active only while guiding/arrived.
  const guiding = state.guidanceStatus === 'guiding' || state.guidanceStatus === 'arrived'
  const guidanceD =
    state.guidancePath && state.guidancePath.length
      ? state.guidancePath.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x * 100} ${p.y * 100}`).join(' ')
      : ''

  return (
    <div className="panel relative flex-1 overflow-hidden">
      {/* Terrain base (procedural forest look) — the fallback when the real terrain image
          is off (toggle) or unavailable (server offline / no rasters -> onError). */}
      <div className="absolute inset-0 bg-base-950" />
      <div
        className="absolute inset-0 opacity-90"
        style={{
          backgroundImage:
            'radial-gradient(120% 90% at 30% 20%, #16321f 0%, #0e2417 40%, #08160e 100%),' +
            'repeating-linear-gradient(48deg, rgba(255,255,255,0.018) 0px, rgba(255,255,255,0.018) 1px, transparent 1px, transparent 9px),' +
            'repeating-linear-gradient(-42deg, rgba(0,0,0,0.12) 0px, rgba(0,0,0,0.12) 1px, transparent 1px, transparent 13px)',
        }}
      />

      {/* Real Marin terrain (muted colorized shaded relief). object-fill stretches it to the
          same normalized 0..1 box every overlay uses, so terrain + heat + markers stay aligned. */}
      {showTerrain && terrainOk && (
        <img
          src={`${API_BASE}/terrain.png`}
          alt=""
          draggable={false}
          onError={() => setTerrainOk(false)}
          className="absolute inset-0 h-full w-full object-fill"
        />
      )}

      {/* Heatmap */}
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full mix-blend-screen" />

      {/* Vector overlays */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full"
      >
        {showPath && (
          <path
            d={pathD}
            fill="none"
            stroke="rgba(255,255,255,0.85)"
            strokeWidth={0.4}
            strokeDasharray="1.4 1.2"
            vectorEffect="non-scaling-stroke"
            className="animate-dash"
          />
        )}

        {/* Guide-home: the planned walkable route back to the operators (green dashed, with a
            dark casing underneath so it reads over the bright heatmap). */}
        {guiding && guidanceD && (
          <>
            <path
              d={guidanceD}
              fill="none"
              stroke="rgba(0,0,0,0.55)"
              strokeWidth={1.4}
              vectorEffect="non-scaling-stroke"
            />
            <path
              d={guidanceD}
              fill="none"
              stroke="#7CFC00"
              strokeWidth={0.7}
              strokeDasharray="1.6 1.2"
              vectorEffect="non-scaling-stroke"
            />
          </>
        )}

        {/* Line-of-sight tether: the subject (follower) to the drone (leader = dronePos). */}
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

      {/* Marker overlay (non-distorting, uses % positioning) */}
      <div className="absolute inset-0">
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
            {/* White-centered so it stays visible ON TOP of the red heatmap glow it sits in. */}
            <span className="absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/40 animate-pulseRing" />
            <div className="relative grid h-4 w-4 place-items-center rounded-full bg-white ring-[3px] ring-accent-red shadow-[0_0_0_1.5px_rgba(0,0,0,0.65)]">
              <span className="h-1.5 w-1.5 rounded-full bg-accent-red" />
            </div>
          </div>
        )}

        {/* Search-phase waypoints declutter during guide-home so the route home reads cleanly. */}
        {showSearched && !guiding &&
          state.waypoints
            .filter((w) => w.kind === 'searched')
            .map((w) => <Marker key={w.id} x={w.pos.x} y={w.pos.y} kind="searched" />)}
        {showDetections && !guiding &&
          state.waypoints
            .filter((w) => w.kind === 'detection')
            .map((w) => <Marker key={w.id} x={w.pos.x} y={w.pos.y} kind="detection" />)}
        {primary && showDetections && (
          <Marker x={primary.pos.x} y={primary.pos.y} kind="high-prob" />
        )}

        {/* Drone */}
        <div
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${state.dronePos.x * 100}%`, top: `${state.dronePos.y * 100}%` }}
        >
          <span className="absolute left-1/2 top-1/2 h-8 w-8 -translate-x-1/2 -translate-y-1/2 rounded-full border border-accent-green/50 animate-pulseRing" />
          <div className="relative grid h-6 w-6 place-items-center rounded-full bg-accent-green text-base-950 shadow-glow">
            <Navigation className="h-3.5 w-3.5" style={{ transform: 'rotate(45deg)' }} />
          </div>
        </div>
      </div>

      {/* Compass */}
      <div className="absolute right-3 top-3 grid h-9 w-9 place-items-center rounded-full bg-base-900/80 text-[10px] font-bold text-slate-300 ring-1 ring-white/10">
        <span className="text-accent-red">N</span>
      </div>

      {/* Zoom + layers controls */}
      <div className="absolute right-3 top-16 flex flex-col gap-1.5">
        <CtrlBtn>
          <Plus className="h-4 w-4" />
        </CtrlBtn>
        <CtrlBtn>
          <Minus className="h-4 w-4" />
        </CtrlBtn>
        <CtrlBtn>
          <Layers className="h-4 w-4" />
        </CtrlBtn>
      </div>

      {/* Legend */}
      <div className="absolute bottom-3 right-3 space-y-1 rounded-lg bg-base-900/85 p-2.5 text-[10px] text-slate-300 ring-1 ring-white/10">
        <LegendRow color="#22c55e" label="Drone (Current)" />
        <LegendRow dashed label="Planned Path" />
        <LegendRow icon="detection" label="Detection" />
        <LegendRow icon="searched" label="Searched (No Detection)" />
        <LegendRow color="#ef4444" label="High Probability" />
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

function Marker({
  x,
  y,
  kind,
}: {
  x: number
  y: number
  kind: 'detection' | 'searched' | 'high-prob'
}) {
  const base =
    'absolute -translate-x-1/2 -translate-y-1/2 grid place-items-center rounded-full shadow'
  if (kind === 'high-prob') {
    return (
      <div className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${x * 100}%`, top: `${y * 100}%` }}>
        <span className="absolute left-1/2 top-1/2 h-9 w-9 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent-red/30 animate-pulseRing" />
        <div className="relative grid h-7 w-7 place-items-center rounded-full bg-accent-red text-white ring-2 ring-white/80">
          <Info className="h-4 w-4" />
        </div>
      </div>
    )
  }
  if (kind === 'searched') {
    return (
      <div
        className={`${base} h-5 w-5 bg-base-900/90 text-slate-300 ring-1 ring-white/30`}
        style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
      >
        <X className="h-3 w-3" />
      </div>
    )
  }
  return (
    <div
      className={`${base} h-5 w-5 bg-accent-blue text-white ring-2 ring-white/40`}
      style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-white" />
    </div>
  )
}

function LegendRow({
  color,
  label,
  dashed,
  icon,
}: {
  color?: string
  label: string
  dashed?: boolean
  icon?: 'detection' | 'searched'
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="grid w-4 place-items-center">
        {dashed ? (
          <span className="h-0 w-4 border-t border-dashed border-white/80" />
        ) : icon === 'detection' ? (
          <span className="h-3 w-3 rounded-full bg-accent-blue ring-1 ring-white/50" />
        ) : icon === 'searched' ? (
          <X className="h-3 w-3 text-slate-300" />
        ) : (
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
        )}
      </span>
      {label}
    </div>
  )
}
