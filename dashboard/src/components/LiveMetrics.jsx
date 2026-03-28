import { useMemo, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const SEVERITY_LEVELS = ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
const SEVERITY_COLORS = {
    NORMAL: '#10b981',
    LOW: '#3b82f6',
    MEDIUM: '#f59e0b',
    HIGH: '#f97316',
    CRITICAL: '#f43f5e',
}

// Available ensembles for comparison (EWMA and Z-Score dropped)
const ENSEMBLES = [
    { key: 'isolation_forest', label: 'Isolation Forest', color: '#3b82f6' },
    { key: 'lstm_autoencoder', label: 'LSTM Autoenc.', color: '#14b8a6' },
    { key: 'if_lstm_combined', label: 'IF+LSTM (Prod)', color: '#8b5cf6' },
    { key: 'xgboost_lstm', label: 'XGBoost+LSTM', color: '#2ecc71' },
    { key: 'xgboost_attention', label: 'XGBoost+Attn', color: '#27ae60' },
    { key: 'ocsvm', label: 'One-Class SVM', color: '#f39c12' },
]

function primaryScore(s) {
    return s?.per_ensemble?.xgboost_lstm ?? s?.ensemble_score ?? 0
}

function deriveSeverity(score) {
    const v = primaryScore(score)
    if (v >= 0.85) return 'CRITICAL'
    if (v >= 0.7) return 'HIGH'
    if (v >= 0.5) return 'MEDIUM'
    if (v >= 0.3) return 'LOW'
    return 'NORMAL'
}

export default function LiveMetrics({ scores, detector }) {
    const [selectedEnsembles, setSelectedEnsembles] = useState(
        () => new Set(ENSEMBLES.map(e => e.key))
    )

    function toggleEnsemble(key) {
        setSelectedEnsembles(prev => {
            const next = new Set(prev)
            if (next.has(key)) {
                if (next.size > 1) next.delete(key)  // keep at least 1
            } else {
                next.add(key)
            }
            return next
        })
    }

    const barData = useMemo(() =>
        scores.map(s => {
            const ens = s.per_ensemble || {}
            const row = {
                name: s.service.replace('-service', '').replace('api-', 'gw'),
            }
            for (const e of ENSEMBLES) {
                row[e.key] = ens[e.key] ?? (
                    e.key === 'isolation_forest' ? s.isoforest_score :
                    e.key === 'lstm_autoencoder' ? s.lstm_score :
                    e.key === 'if_lstm_combined' ? s.ensemble_score : 0
                )
            }
            return row
        }), [scores])

    const severityDist = useMemo(() => {
        const counts = {}
        for (const level of SEVERITY_LEVELS) counts[level] = 0
        for (const s of scores) {
            const sev = deriveSeverity(s)
            counts[sev] = (counts[sev] || 0) + 1
        }
        return counts
    }, [scores])

    const activeEnsembles = ENSEMBLES.filter(e => selectedEnsembles.has(e.key))

    return (
        <div>
            <div className="grid-3" style={{ marginBottom: 16 }}>
                <div className="card">
                    <div className="metric-lbl">Poll Interval</div>
                    <div className="metric-val" style={{ color: 'var(--accent)' }}>15s</div>
                </div>
                <div className="card">
                    <div className="metric-lbl">Services Tracked</div>
                    <div className="metric-val" style={{ color: 'var(--teal)' }}>
                        {detector?.services_monitored ?? scores.length}
                    </div>
                </div>
                <div className="card">
                    <div className="metric-lbl">Anomalies Detected</div>
                    <div className="metric-val" style={{ color: 'var(--err)' }}>
                        {detector?.total_anomalies ?? scores.filter(s => s.is_anomaly).length}
                    </div>
                </div>
            </div>

            {/* Severity Distribution */}
            <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-title" style={{ marginBottom: 14 }}>Severity Distribution</div>
                <div className="sev-dist">
                    {SEVERITY_LEVELS.map(level => (
                        <div key={level} className="sev-dist-item">
                            <div className="sev-dist-dot" style={{ background: SEVERITY_COLORS[level] }} />
                            <div>
                                <div className="sev-dist-count" style={{ color: SEVERITY_COLORS[level] }}>
                                    {severityDist[level]}
                                </div>
                                <div className="sev-dist-label">{level}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Model Score Comparison with ensemble checkboxes */}
            <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-header">
                    <div className="card-title">Model Score Comparison</div>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 14 }}>
                    {ENSEMBLES.map(e => (
                        <label key={e.key} style={{
                            display: 'flex', alignItems: 'center', gap: 5,
                            cursor: 'pointer', fontSize: 11, color: selectedEnsembles.has(e.key) ? e.color : 'var(--text-2)',
                            padding: '3px 8px', borderRadius: 4,
                            background: selectedEnsembles.has(e.key) ? `${e.color}18` : 'transparent',
                            border: `1px solid ${selectedEnsembles.has(e.key) ? e.color + '40' : 'var(--border)'}`,
                            transition: 'all 0.15s',
                        }}>
                            <input type="checkbox" checked={selectedEnsembles.has(e.key)}
                                onChange={() => toggleEnsemble(e.key)}
                                style={{ accentColor: e.color, width: 12, height: 12 }} />
                            {e.label}
                        </label>
                    ))}
                </div>
                {barData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={barData} barGap={1} barCategoryGap="15%">
                            <CartesianGrid strokeDasharray="3 3" stroke="#243049" vertical={false} />
                            <XAxis dataKey="name" stroke="#6b7a94" fontSize={10} tickLine={false} />
                            <YAxis domain={[0, 1]} stroke="#6b7a94" fontSize={10} tickLine={false} />
                            <Tooltip
                                contentStyle={{
                                    background: '#131a28', border: '1px solid #243049',
                                    borderRadius: 6, fontSize: 11,
                                }}
                            />
                            <Legend wrapperStyle={{ fontSize: 10, color: '#6b7a94' }} />
                            {activeEnsembles.map(e => (
                                <Bar key={e.key} dataKey={e.key} fill={e.color}
                                    name={e.label} radius={[2, 2, 0, 0]} />
                            ))}
                        </BarChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="placeholder">Waiting for scores...</div>
                )}
            </div>

            <div className="card">
                <div className="card-title" style={{ marginBottom: 14 }}>Feature Breakdown</div>
                <div style={{ overflowX: 'auto' }}>
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Service</th>
                                <th>Severity</th>
                                <th>Req/s</th>
                                <th>Err %</th>
                                <th>P50</th>
                                <th>P99</th>
                                <th>CPU</th>
                                <th>Mem</th>
                                <th>Restarts</th>
                            </tr>
                        </thead>
                        <tbody>
                            {scores.map(s => {
                                const f = s.features || {}
                                const sevLabel = deriveSeverity(s)
                                const sevClass = `sev-${sevLabel.toLowerCase()}`
                                return (
                                    <tr key={s.service}>
                                        <td className="svc-col">{s.service}</td>
                                        <td>
                                            <span className={`severity-badge ${sevClass}`} style={{ marginTop: 0 }}>
                                                {sevLabel}
                                            </span>
                                        </td>
                                        <td>{(f.request_rate ?? 0).toFixed(1)}</td>
                                        <td>{((f.error_ratio ?? 0) * 100).toFixed(1)}%</td>
                                        <td>{((f.latency_p50 ?? 0) * 1000).toFixed(0)}ms</td>
                                        <td>{((f.latency_p99 ?? 0) * 1000).toFixed(0)}ms</td>
                                        <td>{(f.cpu_usage ?? 0).toFixed(3)}</td>
                                        <td>{(f.memory_usage_mb ?? 0).toFixed(1)}MB</td>
                                        <td>{f.restart_count ?? 0}</td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}
