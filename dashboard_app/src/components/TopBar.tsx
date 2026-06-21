import { Crosshair } from 'lucide-react'

export default function TopBar() {
  return (
    <header className="flex h-14 items-center justify-between border-b border-white/5 bg-base-900/90 px-4">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="grid h-9 w-9 place-items-center rounded-lg bg-accent-cyan/10 ring-1 ring-accent-cyan/30">
          <Crosshair className="h-5 w-5 text-accent-cyan" />
        </div>
        <div className="leading-tight">
          <div className="text-sm font-bold tracking-wide text-white">SAR SUPPORT SYSTEM</div>
          <div className="text-[10px] text-slate-400">Search & Rescue Decision Layer</div>
        </div>
      </div>

      {/* Right cluster */}
      <div className="flex items-center gap-3">
        <div className="hidden text-right leading-tight lg:block">
          <div className="text-[10px] uppercase tracking-wider text-slate-500">System Status</div>
          <div className="text-xs font-semibold text-accent-green">OPERATIONAL</div>
        </div>
        <div className="grid h-9 w-9 place-items-center rounded-full bg-gradient-to-br from-accent-cyan to-accent-blue text-xs font-bold text-base-950">
          OP
        </div>
      </div>
    </header>
  )
}
