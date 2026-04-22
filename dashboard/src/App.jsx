import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import './App.css'
import ServiceTopology from './components/ServiceTopology'
import AnomalyTimeline from './components/AnomalyTimeline'
import ChaosPanel from './components/ChaosPanel'
import LiveMetrics from './components/LiveMetrics'
import EventLog from './components/EventLog'
import PredictionDashboard from './components/PredictionDashboard'
import ErrorBoundary from './components/ErrorBoundary'
import {
  resolveApiBase,
  buildWsUrl,
  parseWsMessage,
  isTypedEnvelope,
  safeArr,
  safeObj,
  safeNum,
} from './lib/security'

const API = resolveApiBase()

const NAV = [
  { id: 'topology', label: 'Topology', group: 'Operate', icon: TopologyIcon },
  { id: 'timeline', label: 'Timeline', group: 'Operate', icon: TimelineIcon },
  { id: 'metrics', label: 'Metrics', group: 'Operate', icon: MetricsIcon },
  { id: 'events', label: 'Events', group: 'Operate', icon: EventsIcon },
  { id: 'chaos', label: 'Chaos Lab', group: 'Experiment', icon: ChaosIcon },
]

const FETCH_TIMEOUT_MS = 8000

async function safeFetch(input, init = {}) {
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS)
  try {
    const res = await fetch(input, {
      ...init,
      signal: ctrl.signal,
      credentials: 'same-origin',
      redirect: 'error',
      referrerPolicy: 'no-referrer',
    })
    if (!res.ok) return null
    const ct = res.headers.get('content-type') || ''
    if (!ct.includes('application/json')) return null
    return await res.json()
  } catch {
    return null
  } finally {
    clearTimeout(t)
  }
}

function App() {
  const [scores, setScores] = useState([])
  const [events, setEvents] = useState([])
  const [detectorStatus, setDetectorStatus] = useState(null)
  const [engineStatus, setEngineStatus] = useState(null)
  const [active, setActive] = useState('topology')
  const [predictionMode, setPredictionMode] = useState(false)
  const [predictions, setPredictions] = useState([])
  const [backendConnected, setBackendConnected] = useState(true)
  const [wsConnected, setWsConnected] = useState(false)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const wsRef = useRef(null)
  const failCountRef = useRef(0)

  const fetchScores = useCallback(async () => {
    const data = await safeFetch(`${API}/anomaly/api/scores`)
    if (!data) {
      failCountRef.current += 1
      if (failCountRef.current >= 2) setBackendConnected(false)
      return
    }
    setScores(safeArr(data.scores))
    setBackendConnected(true)
    setLastUpdated(new Date())
    failCountRef.current = 0
  }, [])

  const fetchStatus = useCallback(async () => {
    const [det, eng] = await Promise.all([
      safeFetch(`${API}/anomaly/api/status`),
      safeFetch(`${API}/decision/api/status`),
    ])
    if (det) setDetectorStatus(safeObj(det))
    if (eng) setEngineStatus(safeObj(eng))
  }, [])

  const fetchPredictions = useCallback(async () => {
    const data = await safeFetch(`${API}/anomaly/api/predictions`)
    if (data) setPredictions(safeArr(data.predictions))
  }, [])

  useEffect(() => {
    fetchScores()
    fetchStatus()
    const a = setInterval(fetchScores, 5000)
    const b = setInterval(fetchStatus, 10000)
    return () => {
      clearInterval(a)
      clearInterval(b)
    }
  }, [fetchScores, fetchStatus])

  useEffect(() => {
    if (!predictionMode) return
    fetchPredictions()
    const c = setInterval(fetchPredictions, 5000)
    return () => clearInterval(c)
  }, [predictionMode, fetchPredictions])

  useEffect(() => {
    let ws
    let timer
    let retries = 0
    const connect = () => {
      ws = new WebSocket(buildWsUrl('/decision/ws/events'))
      ws.onmessage = (e) => {
        const msg = parseWsMessage(e.data)
        if (!msg) return
        // Typed envelopes (e.g. prediction_raised) carry the real payload
        // under `data`. Everything else is a flat recovery event.
        if (isTypedEnvelope(msg)) {
          if (msg.type === 'prediction_raised') {
            setPredictions((prev) => [msg.data, ...prev].slice(0, 100))
          }
          return
        }
        setEvents((prev) => [msg, ...prev].slice(0, 200))
      }
      ws.onopen = () => {
        setWsConnected(true)
        retries = 0
      }
      ws.onclose = () => {
        setWsConnected(false)
        const delay = Math.min(30000, 1000 * 2 ** Math.min(retries, 5))
        retries += 1
        timer = setTimeout(connect, delay)
      }
      ws.onerror = () => {
        try {
          ws.close()
        } catch {
          // ignore
        }
      }
      wsRef.current = ws
    }
    connect()
    return () => {
      clearTimeout(timer)
      try {
        ws?.close()
      } catch {
        // ignore
      }
    }
  }, [])

  const anomalyCount = useMemo(
    () =>
      scores.filter((s) => safeNum(s?.per_ensemble?.xgboost_lstm ?? s?.ensemble_score) > 0.7).length,
    [scores],
  )
  const isLoading = !lastUpdated && backendConnected
  const activePage = NAV.find((n) => n.id === active) || NAV[0]
  const predictionCount = predictions.filter((p) => safeNum(p.confidence) >= 0.6).length

  return (
    <div className={`app ${predictionMode ? 'no-sidebar' : ''}`}>
      {!backendConnected && (
        <div className="connection-banner disconnected">
          Backend disconnected — retrying with exponential backoff…
        </div>
      )}
      {isLoading && <div className="connection-banner loading">Bootstrapping dashboard…</div>}

      {!predictionMode && (
        <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`} aria-label="Primary navigation">
          <div className="sidebar-brand">
            <div className="logo-mark" aria-hidden="true">S</div>
            <div className="brand-text">
              <span className="brand-name">SKAM</span>
              <span className="brand-sub">Self-heal · K8s</span>
            </div>
          </div>

          {['Operate', 'Experiment'].map((group) => (
            <div key={group} className="nav-section">
              <div className="nav-section-label">{group}</div>
              {NAV.filter((n) => n.group === group).map((n) => {
                const Icon = n.icon
                return (
                  <button
                    key={n.id}
                    type="button"
                    className={`nav-item ${active === n.id ? 'active' : ''}`}
                    onClick={() => {
                      setActive(n.id)
                      setSidebarOpen(false)
                    }}
                    aria-current={active === n.id ? 'page' : undefined}
                  >
                    <Icon className="nav-icon" />
                    {n.label}
                  </button>
                )
              })}
            </div>
          ))}

          <div className="sidebar-foot">
            <strong>v2.0 · ocean</strong>
            <br />
            XGBoost+LSTM · AUC 0.98
          </div>
        </aside>
      )}

      <header className="header" role="banner">
        <div className="header-left">
          {predictionMode ? (
            <>
              <div className="logo-mark" style={{ width: 30, height: 30, borderRadius: 8 }}>S</div>
              <div>
                <div className="page-title">Prediction Mode</div>
              </div>
            </>
          ) : (
            <>
              <div className="page-title">{activePage.label}</div>
              <div className="page-subtitle">
                {scores.length} services · {anomalyCount} anomalous
              </div>
            </>
          )}
          {lastUpdated && (
            <span className="last-updated" title="Last successful poll">
              {lastUpdated.toLocaleTimeString('en-GB', { hour12: false })}
            </span>
          )}
          <span className="ws-status" title={wsConnected ? 'Live stream connected' : 'Reconnecting…'}>
            <span className={`ws-dot ${wsConnected ? 'connected' : 'disconnected'}`} />
            {wsConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>

        <div className="header-stats">
          <div className="stat-pill healthy">
            <span className="stat-dot" />
            {Math.max(0, scores.length - anomalyCount)} ok
          </div>
          {anomalyCount > 0 && (
            <div className="stat-pill anomaly">
              <span className="stat-dot" />
              {anomalyCount} anomal{anomalyCount === 1 ? 'y' : 'ies'}
            </div>
          )}
          <div className="stat-pill recovery">
            <span className="stat-dot" />
            {safeNum(engineStatus?.total_recoveries)} recoveries
          </div>
          {predictionMode && predictions.length > 0 && (
            <div className="stat-pill prediction">
              <span className="stat-dot" />
              {predictionCount} prediction{predictionCount !== 1 ? 's' : ''}
            </div>
          )}
          <button
            type="button"
            className={`mode-toggle ${predictionMode ? 'active' : ''}`}
            onClick={() => setPredictionMode((m) => !m)}
            aria-pressed={predictionMode}
          >
            {predictionMode ? '◉ Prediction' : '◯ Detection'}
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
        <main className="content">
          <ErrorBoundary>
            {active === 'topology' && <ServiceTopology scores={scores} />}
            {active === 'timeline' && <AnomalyTimeline scores={scores} />}
            {active === 'chaos' && <ChaosPanel api={API} />}
            {active === 'metrics' && <LiveMetrics scores={scores} detector={detectorStatus} />}
            {active === 'events' && (
              <EventLog events={events} engine={engineStatus} wsConnected={wsConnected} />
            )}
          </ErrorBoundary>
        </main>
      )}
    </div>
  )
}

/* ---------- Inline icons (no external deps, CSP-friendly) -------- */

function TopologyIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <circle cx="5" cy="6" r="2.2" />
      <circle cx="19" cy="6" r="2.2" />
      <circle cx="12" cy="18" r="2.2" />
      <path d="M6.6 7.4 10.6 16.4M17.4 7.4 13.4 16.4M7 6h10" />
    </svg>
  )
}

function TimelineIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M3 12h3l2-6 4 12 3-9 2 3h4" />
    </svg>
  )
}

function MetricsIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M4 20V10M10 20V4M16 20v-8M22 20H2" />
    </svg>
  )
}

function EventsIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M4 6h16M4 12h16M4 18h10" />
    </svg>
  )
}

function ChaosIcon(props) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4M5.6 5.6l2.8 2.8M15.6 15.6l2.8 2.8M18.4 5.6l-2.8 2.8M8.4 15.6l-2.8 2.8" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  )
}

export default App
