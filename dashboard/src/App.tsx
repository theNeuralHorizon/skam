import { useCallback } from 'react'
import { Header } from './components/Header'
import { ServiceTopology } from './components/ServiceTopology'
import { AnomalyTimeline } from './components/AnomalyTimeline'
import { ChaosPanel } from './components/ChaosPanel'
import { RecoveryTimeline } from './components/RecoveryTimeline'
import { MetricsPanel } from './components/MetricsPanel'
import { EventLog } from './components/EventLog'
import { useWebSocket } from './hooks/useWebSocket'
import { usePolling } from './hooks/usePolling'
import {
  fetchServiceHealth,
  fetchAnomalies,
  fetchExperiments,
  fetchRecoveryActions,
} from './utils/api'

function App() {
  const { events, isConnected } = useWebSocket()

  const fetchHealth = useCallback(() => fetchServiceHealth(), [])
  const fetchAnoms = useCallback(() => fetchAnomalies(), [])
  const fetchExps = useCallback(() => fetchExperiments(), [])
  const fetchActions = useCallback(() => fetchRecoveryActions(), [])

  const { data: services } = usePolling(fetchHealth, 10000)
  const { data: anomalies } = usePolling(fetchAnoms, 15000)
  const { data: experiments, refresh: refreshExps } = usePolling(fetchExps, 10000)
  const { data: recoveryActions } = usePolling(fetchActions, 10000)

  const healthyCount = services?.filter((s) => s.status === 'healthy').length ?? 0

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <Header
        isConnected={isConnected}
        healthyCount={healthyCount}
        totalServices={services?.length ?? 6}
      />

      <main className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column */}
        <div className="space-y-4">
          <ServiceTopology services={services ?? []} />
          <MetricsPanel services={services ?? []} />
        </div>

        {/* Center Column */}
        <div className="space-y-4">
          <AnomalyTimeline anomalies={anomalies ?? []} />
          <RecoveryTimeline actions={recoveryActions ?? []} />
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          <ChaosPanel experiments={experiments ?? []} onRefresh={refreshExps} />
          <EventLog events={events} />
        </div>
      </main>
    </div>
  )
}

export default App
