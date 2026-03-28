import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'
import ServiceTopology from './components/ServiceTopology'
import AnomalyTimeline from './components/AnomalyTimeline'
import ChaosPanel from './components/ChaosPanel'
import LiveMetrics from './components/LiveMetrics'
import EventLog from './components/EventLog'

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
  const wsRef = useRef(null)

  const fetchScores = useCallback(async () => {
    try {
      const r = await fetch(`${API}/anomaly/api/scores`)
      if (!r.ok) return
      const d = await r.json()
      // Each score object now includes severity_label and severity_level from the backend
      setScores(d.scores || [])
    } catch (_) { /* backend not reachable yet */ }
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

  useEffect(() => {
    fetchScores()
    fetchStatus()
    const a = setInterval(fetchScores, 5000)
    const b = setInterval(fetchStatus, 10000)
    return () => { clearInterval(a); clearInterval(b) }
  }, [fetchScores, fetchStatus])

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
        } catch (_) { }
      }
      ws.onclose = () => { timer = setTimeout(connect, 3000) }
      wsRef.current = ws
    }
    connect()
    return () => { clearTimeout(timer); ws?.close() }
  }, [])

  const anomalyCount = scores.filter(s => (s.per_ensemble?.xgboost_lstm ?? s.ensemble_score ?? 0) > 0.7).length

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <h1 className="logo">
            <div className="logo-mark">S</div>
            SKAM
          </h1>
          <span className="subtitle">Kubernetes Self-Healing Platform</span>
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
        </div>
      </header>

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
        {tab === 'topology' && <ServiceTopology scores={scores} />}
        {tab === 'timeline' && <AnomalyTimeline scores={scores} />}
        {tab === 'chaos' && <ChaosPanel api={API} />}
        {tab === 'metrics' && <LiveMetrics scores={scores} detector={detectorStatus} />}
        {tab === 'events' && <EventLog events={events} engine={engineStatus} />}
      </main>
    </div>
  )
}

export default App
