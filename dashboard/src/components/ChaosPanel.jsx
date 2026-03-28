import { useState } from 'react'

const FAULTS = [
    { id: 'pod_kill', name: 'Pod Kill', desc: 'Terminate random pods' },
    { id: 'pod_crashloop', name: 'CrashLoop', desc: 'Force restart backoff' },
    { id: 'cpu_stress', name: 'CPU Stress', desc: 'Saturate CPU via stress job' },
    { id: 'memory_pressure', name: 'Memory Pressure', desc: 'Push toward OOM limits' },
    { id: 'network_partition', name: 'Network Block', desc: 'Drop ingress with NetworkPolicy' },
    { id: 'latency_injection', name: 'Latency Inject', desc: 'Add TC netem delay' },
]

const TARGETS = [
    'api-gateway', 'user-service', 'product-service',
    'order-service', 'cart-service', 'payment-service', 'notification-service',
]

export default function ChaosPanel({ api }) {
    const [target, setTarget] = useState('user-service')
    const [duration, setDuration] = useState(30)
    const [experiments, setExperiments] = useState([])
    const [busy, setBusy] = useState(false)

    const inject = async (faultId) => {
        setBusy(true)
        try {
            const res = await fetch(`${api}/chaos/api/experiments`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: `${faultId}-${target}-${Date.now()}`,
                    fault_type: faultId,
                    target: { namespace: 'default', label_selector: `app=${target}` },
                    duration_seconds: duration,
                    parameters: {},
                }),
            })
            if (!res.ok) {
                const err = await res.text()
                setExperiments(prev => [{ fault_type: faultId, status: 'failed', error: err, ts: Date.now() }, ...prev])
                return
            }
            const data = await res.json()
            setExperiments(prev => [{ ...data, ts: Date.now() }, ...prev])
        } catch (e) {
            setExperiments(prev => [{ fault_type: faultId, status: 'failed', error: e.message, ts: Date.now() }, ...prev])
        } finally {
            setBusy(false)
        }
    }

    return (
        <div>
            <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-header">
                    <div className="card-title">Fault Injection</div>
                    <div style={{ display: 'flex', gap: 8 }}>
                        <select className="sel" value={target} onChange={e => setTarget(e.target.value)}>
                            {TARGETS.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                        <select className="sel" value={duration} onChange={e => setDuration(+e.target.value)}>
                            {[15, 30, 60, 120].map(d => <option key={d} value={d}>{d}s</option>)}
                        </select>
                    </div>
                </div>

                <div className="fault-grid">
                    {FAULTS.map(f => (
                        <button key={f.id} className="fault-btn" disabled={busy}
                            onClick={() => inject(f.id)}>
                            <div className="f-name">{f.name}</div>
                            <div className="f-desc">{f.desc}</div>
                        </button>
                    ))}
                </div>
            </div>

            <div className="card">
                <div className="card-header">
                    <div className="card-title">Experiment Log</div>
                    <div className="card-subtitle">{experiments.length} total</div>
                </div>

                {experiments.length === 0 ? (
                    <div className="placeholder">No experiments triggered yet.</div>
                ) : (
                    <div className="event-list">
                        {experiments.map((exp, i) => (
                            <div key={exp.id || i} className="event-row">
                                <div className={`evt-dot ${exp.status || 'pending'}`} />
                                <span className="evt-svc">{exp.fault_type}</span>
                                <span className="evt-act">{exp.target?.label_selector || target}</span>
                                <span className={`tag ${exp.status === 'completed' ? 'low' : exp.status === 'running' ? 'medium' : 'high'}`}>
                                    {exp.status || 'queued'}
                                </span>
                                {exp.error && <span style={{ fontSize: 11, color: 'var(--err)' }}>{exp.error.slice(0, 50)}</span>}
                                <span className="evt-time">{exp.duration_seconds || duration}s</span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    )
}
