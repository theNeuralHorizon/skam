import { useEffect, useRef, useState } from 'react'
import { safeStr, safeArr } from '../lib/security'

const FAULTS = [
  { id: 'pod_kill', name: 'Pod Kill', desc: 'Terminate random pods' },
  { id: 'pod_crashloop', name: 'CrashLoop', desc: 'Force restart backoff' },
  { id: 'cpu_stress', name: 'CPU Stress', desc: 'Saturate CPU via stress job' },
  { id: 'memory_pressure', name: 'Memory Pressure', desc: 'Push toward OOM limits' },
  { id: 'network_partition', name: 'Network Block', desc: 'Drop ingress with NetworkPolicy' },
  { id: 'latency_injection', name: 'Latency Inject', desc: 'Add TC netem delay' },
]
const FAULT_IDS = new Set(FAULTS.map((f) => f.id))

const TARGETS = [
  'api-gateway',
  'user-service',
  'product-service',
  'order-service',
  'cart-service',
  'payment-service',
  'notification-service',
]
const TARGET_SET = new Set(TARGETS)

const DURATIONS = [15, 30, 60, 120]
const DURATION_SET = new Set(DURATIONS)

const MAX_LOG = 50

export default function ChaosPanel({ api }) {
  const [target, setTarget] = useState('user-service')
  const [duration, setDuration] = useState(30)
  const [experiments, setExperiments] = useState([])
  const [busy, setBusy] = useState(false)
  const abortRef = useRef(null)

  useEffect(() => () => abortRef.current?.abort(), [])

  const inject = async (faultId) => {
    if (!FAULT_IDS.has(faultId) || !TARGET_SET.has(target) || !DURATION_SET.has(duration)) return
    setBusy(true)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    const timer = setTimeout(() => ctrl.abort(), 10000)
    const ts = Date.now()
    try {
      const res = await fetch(`${api}/chaos/api/experiments`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        signal: ctrl.signal,
        body: JSON.stringify({
          name: `${faultId}-${target}-${ts}`,
          fault_type: faultId,
          target: { namespace: 'default', label_selector: `app=${target}` },
          duration_seconds: duration,
          parameters: {},
        }),
      })
      if (!res.ok) {
        const err = safeStr(await res.text(), 240)
        setExperiments((prev) =>
          [{ fault_type: faultId, target, status: 'failed', error: err, ts }, ...prev].slice(0, MAX_LOG),
        )
        return
      }
      const data = await res.json()
      setExperiments((prev) =>
        [
          {
            id: safeStr(data?.id, 64),
            fault_type: safeStr(data?.fault_type || faultId, 32),
            status: safeStr(data?.status || 'running', 16),
            duration_seconds: Number(data?.duration_seconds) || duration,
            target: { label_selector: `app=${target}` },
            ts,
          },
          ...prev,
        ].slice(0, MAX_LOG),
      )
    } catch (e) {
      if (e.name !== 'AbortError') {
        setExperiments((prev) =>
          [
            { fault_type: faultId, target, status: 'failed', error: safeStr(e.message, 240), ts },
            ...prev,
          ].slice(0, MAX_LOG),
        )
      }
    } finally {
      clearTimeout(timer)
      setBusy(false)
    }
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="card-header">
          <div>
            <div className="card-title">Chaos Engineering</div>
            <div className="card-subtitle">
              Inject controlled failures — the decision engine will heal them automatically.
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <select
              className="sel"
              value={target}
              onChange={(e) => TARGET_SET.has(e.target.value) && setTarget(e.target.value)}
              aria-label="Target service"
            >
              {TARGETS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
            <select
              className="sel"
              value={duration}
              onChange={(e) => {
                const v = +e.target.value
                if (DURATION_SET.has(v)) setDuration(v)
              }}
              aria-label="Fault duration"
            >
              {DURATIONS.map((d) => (
                <option key={d} value={d}>
                  {d}s
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="fault-grid">
          {FAULTS.map((f) => (
            <button
              key={f.id}
              type="button"
              className="fault-btn"
              disabled={busy}
              onClick={() => inject(f.id)}
              onMouseMove={(e) => {
                const r = e.currentTarget.getBoundingClientRect()
                e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`)
                e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`)
              }}
              aria-label={`Inject ${f.name} on ${target}`}
            >
              <div className="f-name">{f.name}</div>
              <div className="f-sub">{f.desc}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="card-title">Experiment Log</div>
          <div className="card-subtitle">{safeArr(experiments).length} runs</div>
        </div>

        {experiments.length === 0 ? (
          <div className="placeholder">
            No experiments yet. Pick a target and inject a fault to watch auto-healing in action.
          </div>
        ) : (
          <div className="event-list">
            {experiments.map((exp, i) => (
              <div key={`${exp.id || 'x'}-${i}`} className="event-row">
                <div className={`evt-dot ${safeStr(exp.status, 16) || 'pending'}`} />
                <span className="evt-svc">{safeStr(exp.fault_type, 32)}</span>
                <span className="evt-act">{safeStr(exp.target?.label_selector || exp.target, 48)}</span>
                <span className={`severity-badge ${statusToSev(exp.status)}`} style={{ marginTop: 0 }}>
                  {safeStr(exp.status, 16) || 'queued'}
                </span>
                {exp.error && (
                  <span style={{ fontSize: 11, color: 'var(--critical)' }} title={exp.error}>
                    {safeStr(exp.error, 60)}
                  </span>
                )}
                <span className="evt-time">{exp.duration_seconds || duration}s</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function statusToSev(status) {
  switch (status) {
    case 'completed':
      return 'sev-normal'
    case 'running':
      return 'sev-medium'
    case 'failed':
      return 'sev-critical'
    default:
      return 'sev-low'
  }
}
