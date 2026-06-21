import { ChevronRight } from 'lucide-react'
import type { Detection } from '../types'

export default function DetectionsList({ detections }: { detections: Detection[] }) {
  return (
    <div className="panel flex min-h-0 flex-1 flex-col overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2">
        <div className="panel-header">
          Detections <span className="text-accent-cyan">(Live)</span>
        </div>
        <span className="text-[11px] text-slate-400">{detections.length} Total</span>
      </div>

      <div className="min-h-0 flex-1 divide-y divide-white/5 overflow-y-auto">
        {detections.map((d) => (
          <Row key={d.id} d={d} />
        ))}
      </div>
    </div>
  )
}

function Row({ d }: { d: Detection }) {
  const confColor =
    d.confidence >= 0.85 ? 'text-accent-red' : d.confidence >= 0.7 ? 'text-accent-amber' : 'text-slate-300'
  return (
    <div className="flex items-center gap-2.5 px-3 py-2">
      {/* thumbnail */}
      <div
        className="h-9 w-9 shrink-0 rounded ring-1 ring-white/10"
        style={{
          backgroundImage: `radial-gradient(circle at 50% 40%, hsl(${d.thumbnailHue} 35% 38%), hsl(${d.thumbnailHue} 40% 16%))`,
        }}
      />
      <div className="grid flex-1 grid-cols-[auto_1fr_auto] items-center gap-x-2 text-[11px]">
        <span className="font-mono text-slate-300">{d.timestamp}</span>
        <span className={`font-mono font-semibold ${confColor}`}>{d.confidence.toFixed(2)}</span>
        {d.isPrimary ? (
          <span className="text-[10px] font-bold text-accent-red">{d.label}</span>
        ) : (
          <span className="text-slate-500">
            {d.persistence.seen}/{d.persistence.of}
          </span>
        )}
        <span className="col-start-1 text-slate-500">
          {d.isPrimary ? `Persistent: ${d.persistence.seen}/${d.persistence.of}` : ''}
        </span>
        <span className="col-start-3 row-start-1 text-slate-400">{d.distanceM} m</span>
      </div>
      <ChevronRight className="h-4 w-4 shrink-0 text-slate-600" />
    </div>
  )
}
