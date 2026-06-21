import { Video } from 'lucide-react'
import type { MapState } from '../types'

export default function LiveVideoFeed({ telemetry }: { telemetry: MapState['telemetry'] }) {
  return (
    <div className="panel overflow-hidden">
      <div className="flex items-center justify-between px-3 py-2">
        <div className="panel-header flex items-center gap-1.5">
          <Video className="h-3 w-3 text-accent-cyan" /> Live Video Feed
        </div>
        <span className="flex items-center gap-1 rounded bg-accent-red/15 px-1.5 py-0.5 text-[10px] font-bold text-accent-red">
          <span className="h-1.5 w-1.5 rounded-full bg-accent-red animate-blink" /> LIVE
        </span>
      </div>

      {/* Feed frame (looping drone footage) */}
      <div className="relative aspect-video w-full overflow-hidden">
        <video
          className="absolute inset-0 h-full w-full object-cover"
          src="/drone.mov"
          autoPlay
          loop
          muted
          playsInline
        />
        {/* HUD overlay */}
        <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/70 to-transparent px-3 py-1.5 font-mono text-[10px] text-slate-200">
          <span>ALT {telemetry.altM} m</span>
          <span>SPD {telemetry.spdMs.toFixed(1)} m/s</span>
          <span>HDG {telemetry.hdgDeg}°</span>
          <span>{telemetry.feedTime}</span>
        </div>
      </div>
    </div>
  )
}
