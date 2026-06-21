import { WifiOff } from 'lucide-react'

// DEMO MODE: the central map shows a pre-rendered animation of the 3-drone search + guide-home
// run (demo_output/search_and_guide_3drones.gif, copied into public/). This stands in for the
// live server-rendered map (/map_base.png + vector overlays) so the dashboard can be demoed
// without the integration server running. The gif already bakes in the terrain, posterior heat,
// the drone fleet + trails, and the guide-home route, so no live overlays are drawn over it.
const DEMO_GIF = '/search_and_guide_3drones.gif'

interface Props {
  // False when the integration server is unreachable. In demo mode it's effectively always false.
  live: boolean
}

export default function ProbabilityMap({ live }: Props) {
  return (
    <div className="panel relative flex-1 overflow-hidden bg-base-950">
      {/* The map content is a centered SQUARE (matching the demo gif's aspect), so it is never
          stretched into the panel's rectangle. */}
      <div className="absolute inset-0 grid place-items-center">
        <div className="relative aspect-square h-full max-w-full">
          <img
            src={DEMO_GIF}
            alt="3-drone search and guide-home demo"
            draggable={false}
            className="absolute inset-0 h-full w-full object-contain"
          />
        </div>
      </div>

      {/* Compass */}
      <div className="absolute right-3 top-3 grid h-9 w-9 place-items-center rounded-full bg-base-900/80 text-[10px] font-bold text-slate-300 ring-1 ring-white/10">
        <span className="text-accent-red">N</span>
      </div>

      {/* Offline flag: shown when the integration server is unreachable, so the bundled demo
          playback isn't mistaken for a live search. */}
      {!live && (
        <div className="absolute left-3 top-3 z-10 flex items-center gap-1.5 rounded-md bg-accent-amber/15 px-2 py-1 text-[10px] font-semibold text-accent-amber ring-1 ring-accent-amber/30">
          <WifiOff className="h-3 w-3" />
          Demo playback — recorded run
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
