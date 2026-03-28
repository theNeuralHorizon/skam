import { useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const SEVERITY_LEVELS = ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
const SEVERITY_COLORS = {
    NORMAL: '#10b981',
    LOW: '#3b82f6',
    MEDIUM: '#f59e0b',
    HIGH: '#f97316',
    CRITICAL: '#f43f5e',
}

function deriveSeverity(score) {
    if (score?.severity_label) return score.severity_label
    const v = score?.ensemble_score ?? 0
    if (v >= 0.8) return 'CRITICAL'
    if (v >= 0.65) return 'HIGH'
    if (v >= 0.5) return 'MEDIUM'
    if (v >= 0.3) return 'LOW'
    return 'NORMAL'
}

export default function LiveMetrics({ scores, detector }) {
    const barData = useMemo(() =>
        scores.map(s => ({
            name: s.service.replace('-service', '').replace('api-', 'gw'),
            isoforest: s.isoforest_score ?? 0,
            lstm: s.lstm_score ?? 0,
            ensemble: s.ensemble_score ?? 0,
        })), [scores])

    // Compute severity distribution from scores
    const severityDist = useMemo(() => {
        const counts = {}
        for (const level of SEVERITY_LEVELS) counts[level] = 0
        for (const s of scores) {
            const sev = deriveSeverity(s)
            counts[sev] = (counts[sev] || 0) + 1
        }
        return counts
    }, [scores])

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
                        {detector?.total_anomalies ?? 0}
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

            <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-title" style={{ marginBottom: 14 }}>Model Score Comparison</div>
                {barData.length > 0 ? (
                    <ResponsiveContainer width="100%" height={280}>
                        <BarChart data={barData} barGap={2}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#243049" vertical={false} />
                            <XAxis dataKey="name" stroke="#6b7a94" fontSize={10} tickLine={false} />
                            <YAxis domain={[0, 1]} stroke="#6b7a94" fontSize={10} tickLine={false} />
                            <Tooltip
                                contentStyle={{
                                    background: '#131a28', border: '1px solid #243049',
                                    borderRadius: 6, fontSize: 11,
                                }}
                            />
                            <Legend wrapperStyle={{ fontSize: 11, color: '#6b7a94' }} />
                            <Bar dataKey="isoforest" fill="#3b82f6" name="Isolation Forest" radius={[3, 3, 0, 0]} />
                            <Bar dataKey="lstm" fill="#14b8a6" name="LSTM Autoenc." radius={[3, 3, 0, 0]} />
                            <Bar dataKey="ensemble" fill="#8b5cf6" name="Ensemble" radius={[3, 3, 0, 0]} />
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
