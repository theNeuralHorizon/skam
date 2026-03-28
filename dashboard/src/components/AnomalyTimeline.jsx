import { useState, useEffect, useRef } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea } from 'recharts'

const PALETTE = {
    'api-gateway': '#3b82f6',
    'user-service': '#14b8a6',
    'product-service': '#10b981',
    'order-service': '#f59e0b',
    'cart-service': '#8b5cf6',
    'payment-service': '#f43f5e',
    'notification-service': '#fb923c',
}

const MAX_POINTS = 60

export default function AnomalyTimeline({ scores }) {
    const [history, setHistory] = useState([])
    const [filter, setFilter] = useState('all')
    const prevRef = useRef(null)

    // Only append when score data actually changes (by checking JSON hash)
    useEffect(() => {
        if (!scores.length) return
        const hash = scores.map(s => `${s.service}:${s.ensemble_score}`).join('|')
        if (hash === prevRef.current) return
        prevRef.current = hash

        const point = { t: new Date().toLocaleTimeString('en-GB', { hour12: false }) }
        for (const s of scores) point[s.service] = s.ensemble_score
        setHistory(prev => [...prev.slice(-(MAX_POINTS - 1)), point])
    }, [scores])

    const services = scores.map(s => s.service)
    const visible = filter === 'all' ? services : [filter]

    return (
        <div>
            <div className="card">
                <div className="card-header">
                    <div className="card-title">Anomaly Scores Over Time</div>
                    <select className="sel" value={filter}
                        onChange={e => setFilter(e.target.value)}>
                        <option value="all">All services</option>
                        {services.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                </div>

                {history.length < 2 ? (
                    <div className="placeholder">
                        Accumulating data points — chart renders after 2 ticks.
                    </div>
                ) : (
                    <ResponsiveContainer width="100%" height={380}>
                        <AreaChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                            <defs>
                                {visible.map(svc => (
                                    <linearGradient key={svc} id={`g-${svc}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={PALETTE[svc] || '#3b82f6'} stopOpacity={0.25} />
                                        <stop offset="100%" stopColor={PALETTE[svc] || '#3b82f6'} stopOpacity={0} />
                                    </linearGradient>
                                ))}
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#243049" vertical={false} />

                            {/* Severity background bands */}
                            <ReferenceArea y1={0} y2={0.5} fill="#10b981" fillOpacity={0.06}
                                label={{ value: 'NORMAL', fill: '#10b981', fontSize: 9, position: 'insideTopLeft', opacity: 0.5 }} />
                            <ReferenceArea y1={0.5} y2={0.65} fill="#f59e0b" fillOpacity={0.06}
                                label={{ value: 'MEDIUM', fill: '#f59e0b', fontSize: 9, position: 'insideTopLeft', opacity: 0.5 }} />
                            <ReferenceArea y1={0.65} y2={0.8} fill="#f97316" fillOpacity={0.06}
                                label={{ value: 'HIGH', fill: '#f97316', fontSize: 9, position: 'insideTopLeft', opacity: 0.5 }} />
                            <ReferenceArea y1={0.8} y2={1.0} fill="#f43f5e" fillOpacity={0.08}
                                label={{ value: 'CRITICAL', fill: '#f43f5e', fontSize: 9, position: 'insideTopLeft', opacity: 0.5 }} />

                            <XAxis dataKey="t" stroke="#6b7a94" fontSize={10} tickLine={false} />
                            <YAxis domain={[0, 1]} stroke="#6b7a94" fontSize={10} tickLine={false} />
                            <Tooltip
                                contentStyle={{
                                    background: '#131a28', border: '1px solid #243049',
                                    borderRadius: 6, fontSize: 11,
                                }}
                                itemStyle={{ padding: 0 }}
                            />
                            <ReferenceLine y={0.7} stroke="#f43f5e" strokeDasharray="4 3"
                                label={{ value: '0.7 threshold', fill: '#f43f5e', fontSize: 10, position: 'insideTopRight' }} />
                            {visible.map(svc => (
                                <Area key={svc} type="monotone" dataKey={svc}
                                    stroke={PALETTE[svc] || '#3b82f6'}
                                    fill={`url(#g-${svc})`}
                                    strokeWidth={1.5} dot={false} isAnimationActive={false} />
                            ))}
                        </AreaChart>
                    </ResponsiveContainer>
                )}
            </div>

            <div className="grid-3" style={{ marginTop: 16 }}>
                {scores.map(s => {
                    const c = PALETTE[s.service]
                    const sevLabel = s.severity_label || 'NORMAL'
                    const sevClass = `sev-${sevLabel.toLowerCase()}`
                    return (
                        <div key={s.service} className="card">
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                                <span style={{ width: 8, height: 8, borderRadius: 2, background: c, flexShrink: 0 }} />
                                <span className="card-title">{s.service}</span>
                                <span className={`severity-badge ${sevClass}`} style={{ marginTop: 0, marginLeft: 'auto' }}>
                                    {sevLabel}
                                </span>
                            </div>
                            <div style={{ display: 'flex', gap: 16 }}>
                                <div>
                                    <div className="metric-val" style={{ fontSize: 16, color: '#3b82f6' }}>
                                        {(s.isoforest_score ?? 0).toFixed(3)}
                                    </div>
                                    <div className="metric-lbl">IsoForest</div>
                                </div>
                                <div>
                                    <div className="metric-val" style={{ fontSize: 16, color: '#14b8a6' }}>
                                        {(s.lstm_score ?? 0).toFixed(3)}
                                    </div>
                                    <div className="metric-lbl">LSTM</div>
                                </div>
                                <div>
                                    <div className="metric-val" style={{ fontSize: 16, color: s.is_anomaly ? '#f43f5e' : '#10b981' }}>
                                        {(s.ensemble_score ?? 0).toFixed(3)}
                                    </div>
                                    <div className="metric-lbl">Ensemble</div>
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
