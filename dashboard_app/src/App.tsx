import { useState } from 'react'
import TopBar from './components/TopBar'
import StatusBar from './components/StatusBar'
import MissionSidebar from './components/MissionSidebar'
import SearchLoop from './components/SearchLoop'
import ConfidencePanel from './components/ConfidencePanel'
import ProbabilityMap from './components/ProbabilityMap'
import LiveVideoFeed from './components/LiveVideoFeed'
import DetectionsList from './components/DetectionsList'
import ProbabilityTrend from './components/ProbabilityTrend'
import MapUpdateSummary from './components/MapUpdateSummary'
import VoiceComms from './components/VoiceComms'
import LiveTranscript from './components/LiveTranscript'
import { useMapState } from './hooks/useMapState'

export default function App() {
  // Live MapState from the brain's integration server (falls back to mockState when offline).
  const state = useMapState()

  // Local copy so layer toggles are interactive (dashboard reads state; toggles are view-only).
  // Seeded once from the first state; the layer list is static, so polling never clobbers a toggle.
  const [layers, setLayers] = useState(state.layers)

  const toggleLayer = (id: string) =>
    setLayers((ls) => ls.map((l) => (l.id === id ? { ...l, enabled: !l.enabled } : l)))

  return (
    <div className="flex h-screen flex-col bg-base-950 text-slate-200">
      <TopBar />

      <div className="flex min-h-0 flex-1">
        <MissionSidebar
          missionName={state.missionName}
          startedAt={state.startedAt}
          stats={state.stats}
          layers={layers}
          onToggleLayer={toggleLayer}
        />

        {/* Main work area */}
        <main className="flex min-w-0 flex-1 flex-col gap-3 overflow-y-auto p-3">
          {/* Top strip: loop + confidence + live feed header sit across the row */}
          <div className="flex gap-3">
            <div className="min-w-0 flex-1">
              <SearchLoop steps={state.loop} />
            </div>
            <ConfidencePanel
              confidence={state.confidenceToDeclare}
              threshold={state.declareThreshold}
            />
          </div>

          {/* Middle: map (center) + right rail */}
          <div className="flex min-h-[440px] flex-1 gap-3">
            <div className="flex min-w-0 flex-1 flex-col">
              {/* The map is now a server-rendered unified base (terrain+posterior+sectors) under
                  live vector overlays; the sidebar layer toggles are cosmetic for it. */}
              <ProbabilityMap state={state} />
            </div>

            <div className="flex w-[320px] shrink-0 flex-col gap-3">
              <LiveVideoFeed telemetry={state.telemetry} />
              <DetectionsList detections={state.detections} />
            </div>
          </div>

          {/* Bottom row */}
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            <ProbabilityTrend data={state.trend} now={state.confidenceToDeclare} />
            <MapUpdateSummary lastUpdate={state.telemetry.feedTime} />
            <VoiceComms commands={state.recentCommands} />
          </div>

          {/* Live transcript from the deployed voice agent (real data) */}
          <LiveTranscript />
        </main>
      </div>

      <StatusBar />
    </div>
  )
}
