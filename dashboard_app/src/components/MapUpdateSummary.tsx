import { ArrowUp, ArrowDown, Flame } from 'lucide-react'

export default function MapUpdateSummary({ lastUpdate }: { lastUpdate: string }) {
  const rows = [
    {
      icon: ArrowUp,
      color: 'text-accent-green',
      ring: 'bg-accent-green/15',
      title: 'Detection evidence added',
      sub: '7 detections (capped & fused)',
    },
    {
      icon: ArrowDown,
      color: 'text-accent-blue',
      ring: 'bg-accent-blue/15',
      title: 'Searched areas (no detection)',
      sub: '42.1 ha updated',
    },
    {
      icon: Flame,
      color: 'text-accent-red',
      ring: 'bg-accent-red/15',
      title: 'High probability area',
      sub: '0.18 km² (↑ 23%)',
    },
  ]

  return (
    <div className="panel flex flex-col p-3">
      <div className="panel-header">Map Update Summary</div>

      <div className="mt-2 flex-1 space-y-2.5">
        {rows.map(({ icon: Icon, color, ring, title, sub }) => (
          <div key={title} className="flex items-center gap-3">
            <div className={`grid h-8 w-8 shrink-0 place-items-center rounded-full ${ring}`}>
              <Icon className={`h-4 w-4 ${color}`} />
            </div>
            <div className="leading-tight">
              <div className="text-[13px] font-medium text-white">{title}</div>
              <div className="text-[11px] text-slate-500">{sub}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-3 flex items-center border-t border-white/5 pt-2">
        <span className="text-[11px] text-slate-500">Last Update: {lastUpdate}</span>
      </div>
    </div>
  )
}
