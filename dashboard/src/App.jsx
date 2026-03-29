import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import ServiceTopology from './components/ServiceTopology'
import AnomalyTimeline from './components/AnomalyTimeline'
import ChaosPanel from './components/ChaosPanel'
import LiveMetrics from './components/LiveMetrics'
import EventLog from './components/EventLog'
import PredictionDashboard from './components/PredictionDashboard'
import ErrorBoundary from './components/ErrorBoundary'

const API = import.meta.env.VITE_API_BASE || ''

const TABS = [
  { id: 'topology', label: 'Topology' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'chaos', label: 'Chaos' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'events', label: 'Events' },
]

function App() {
  const [scores, setScores] = useState([])
  const [events, setEvents] = useState([])
  const [detectorStatus, setDetectorStatus] = useState(null)
  const [engineStatus, setEngineStatus] = useState(null)
  const [tab, setTab] = useState('topology')
  const [predictionMode, setPredictionMode] = useState(false)
  const [predictions, setPredictions] = useState([])
  const [backendConnected, setBackendConnected] = useState(true)
  const [wsConnected, setWsConnected] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)
  const wsRef = useRef(null)
  const failCountRef = useRef(0)

  const fetchScores = useCallback(async () => {
    try {
      const r = await fetch(`${API}/anomaly/api/scores`)
      if (!r.ok) return
      const d = await r.json()
      setScores(d.scores || [])
      setBackendConnected(true)
      setLastUpdated(new Date())
      failCountRef.current = 0
    } catch (_) {
      // Only show disconnected after 2 consecutive failures to avoid flicker
      failCountRef.current += 1
      if (failCountRef.current >= 2) {
        setBackendConnected(false)
      }
    }
  }, [])

  const fetchStatus = useCallback(async () => {
    try {
      const [det, eng] = await Promise.allSettled([
        fetch(`${API}/anomaly/api/status`).then(r => r.json()),
        fetch(`${API}/decision/api/status`).then(r => r.json()),
      ])
      if (det.status === 'fulfilled') setDetectorStatus(det.value)
      if (eng.status === 'fulfilled') setEngineStatus(eng.value)
    } catch (_) { }
  }, [])

  const fetchPredictions = useCallback(async () => {
    try {
      const r = await fetch(`${API}/anomaly/api/predictions`)
      if (!r.ok) return
      const d = await r.json()
      setPredictions(d.predictions || [])
    } catch (_) { }
  }, [])

  useEffect(() => {
    fetchScores()
    fetchStatus()
    const a = setInterval(fetchScores, 5000)
    const b = setInterval(fetchStatus, 10000)
    return () => { clearInterval(a); clearInterval(b) }
  }, [fetchScores, fetchStatus])

  // Fetch predictions when prediction mode is active
  useEffect(() => {
    if (!predictionMode) return
    fetchPredictions()
    const c = setInterval(fetchPredictions, 5000)
    return () => clearInterval(c)
  }, [predictionMode, fetchPredictions])

  useEffect(() => {
    let ws
    let timer
    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/decision/ws/events`)
      ws.onmessage = (e) => {
        try {
          const evt = JSON.parse(e.data)
          setEvents(prev => [evt, ...prev].slice(0, 200))
          if (evt.type === 'prediction_raised' && evt.data) {
            setPredictions(prev => [evt.data, ...prev].slice(0, 100))
          }
        } catch (_) { }
      }
      ws.onopen = () => { setWsConnected(true) }
      ws.onclose = () => { setWsConnected(false); timer = setTimeout(connect, 3000) }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(timer); ws?.close() }
  }, [])

  const anomalyCount = scores.filter(s => (s.per_ensemble?.xgboost_lstm ?? s.ensemble_score ?? 0) > 0.7).length
  const isLoading = !lastUpdated && backendConnected

  return (
    <div className="app">
      {!backendConnected && (
        <div className="connection-banner disconnected">
          Backend disconnected — data may be stale. Retrying...
        </div>
      )}
      {isLoading && (
        <div className="connection-banner loading">
          Connecting to backend...
        </div>
      )}
      <header className="header">
        <div className="header-left">
          <h1 className="logo">
            <div className="logo-mark">S</div>
            SKAM
          </h1>
          <span className="subtitle">Kubernetes Self-Healing Platform</span>
          {lastUpdated && (
            <span className="last-updated" title="Last successful data fetch">
              {lastUpdated.toLocaleTimeString('en-GB', { hour12: false })}
            </span>
          )}
        </div>
        <div className="header-stats">
          <div className="stat-pill healthy">
            <span className="stat-dot" />
            {scores.length - anomalyCount} ok
          </div>
          {anomalyCount > 0 && (
            <div className="stat-pill anomaly">
              <span className="stat-dot" />
              {anomalyCount} anomal{anomalyCount === 1 ? 'y' : 'ies'}
            </div>
          )}
          <div className="stat-pill recovery">
            <span className="stat-dot" />
            {engineStatus?.total_recoveries || 0} recoveries
          </div>
          {predictionMode && predictions.length > 0 && (
            <div className="stat-pill prediction">
              <span className="stat-dot" />
              {predictions.filter(p => p.confidence >= 0.6).length} prediction{predictions.filter(p => p.confidence >= 0.6).length !== 1 ? 's' : ''}
            </div>
          )}
          <button
            className={`mode-toggle ${predictionMode ? 'active' : ''}`}
            onClick={() => setPredictionMode(m => !m)}
          >
            {predictionMode ? 'Prediction' : 'Detection'}
          </button>
        </div>
      </header>

      {predictionMode ? (
        <main className="content">
          <ErrorBoundary>
            <PredictionDashboard scores={scores} predictions={predictions} />
          </ErrorBoundary>
        </main>
      ) : (
        <>
          <nav className="tab-bar">
            {TABS.map(t => (
              <button
                key={t.id}
                className={`tab${tab === t.id ? ' active' : ''}`}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>

          <main className="content">
            <ErrorBoundary>
              {tab === 'topology' && <ServiceTopology scores={scores} />}
              {tab === 'timeline' && <AnomalyTimeline scores={scores} />}
              {tab === 'chaos' && <ChaosPanel api={API} />}
              {tab === 'metrics' && <LiveMetrics scores={scores} detector={detectorStatus} />}
              {tab === 'events' && <EventLog events={events} engine={engineStatus} wsConnected={wsConnected} />}
            </ErrorBoundary>
          </main>
        </>
      )}
    </div>
  )
}

export default App
