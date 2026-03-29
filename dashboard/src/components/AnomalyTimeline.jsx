import { useState, useEffect, useRef } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea, Legend } from 'recharts'

const SVC_PALETTE = {
    'api-gateway': '#3b82f6',
    'user-service': '#e879f9',
    'product-service': '#10b981',
    'order-service': '#f59e0b',
    'cart-service': '#f43f5e',
    'payment-service': '#06b6d4',
    'notification-service': '#fb923c',
}

const MODELS = [
    { key: 'isolation_forest', label: 'Isolation Forest', color: '#3b82f6' },
    { key: 'lstm_autoencoder', label: 'LSTM Autoenc.', color: '#e879f9' },
    { key: 'if_lstm_combined', label: 'IF+LSTM (Prod)', color: '#8b5cf6' },
    { key: 'xgboost_lstm', label: 'XGBoost+LSTM', color: '#10b981' },
    { key: 'xgboost_attention', label: 'XGBoost+Attn', color: '#f59e0b' },
    { key: 'xgboost_meta', label: 'XGBoost Meta', color: '#f43f5e' },
]

const MAX_POINTS = 60

export default function AnomalyTimeline({ scores }) {
    const [history, setHistory] = useState([])
    const [modelHistory, setModelHistory] = useState([])
    const [visibleServices, setVisibleServices] = useState(() => new Set())
    const [visibleModels, setVisibleModels] = useState(() => new Set(MODELS.map(m => m.key)))
    const [view, setView] = useState('services')
    const [selectedService, setSelectedService] = useState('')
    const prevRef = useRef(null)
    const initRef = useRef(false)

    // Initialize visibleServices with all services on first data
    useEffect(() => {
        if (!initRef.current && scores.length > 0) {
            setVisibleServices(new Set(scores.map(s => s.service)))
            initRef.current = true
        }
    }, [scores])

    useEffect(() => {
        if (!scores.length) return
        const xgbScore = s => s.per_ensemble?.xgboost_lstm ?? s.ensemble_score ?? 0
        const hash = scores.map(s => `${s.service}:${xgbScore(s)}`).join('|')
        if (hash === prevRef.current) return
        prevRef.current = hash

        const t = new Date().toLocaleTimeString('en-GB', { hour12: false })

        const point = { t }
        for (const s of scores) point[s.service] = xgbScore(s)
        setHistory(prev => [...prev.slice(-(MAX_POINTS - 1)), point])

        const modelPoint = { t }
        const target = selectedService || scores[0]?.service || ''
        const svc = scores.find(s => s.service === target)
        if (svc) {
            const ens = svc.per_ensemble || {}
            for (const m of MODELS) {
                modelPoint[m.key] = ens[m.key] ?? (
                    m.key === 'isolation_forest' ? svc.isoforest_score :
                    m.key === 'lstm_autoencoder' ? svc.lstm_score :
                    m.key === 'if_lstm_combined' ? svc.ensemble_score : 0
                )
            }
        }
        setModelHistory(prev => [...prev.slice(-(MAX_POINTS - 1)), modelPoint])
    }, [scores, selectedService])

    const services = scores.map(s => s.service)

    useEffect(() => {
        if (view === 'models' && !selectedService && services.length > 0) {
            setSelectedService(services[0])
        }
    }, [view, services, selectedService])

    const prevSvc = useRef(selectedService)
    useEffect(() => {
        if (prevSvc.current !== selectedService) {
            setModelHistory([])
            prevSvc.current = selectedService
        }
    }, [selectedService])

    function toggleService(svc) {
        setVisibleServices(prev => {
            const next = new Set(prev)
            if (next.has(svc)) {
                if (next.size > 1) next.delete(svc)
            } else {
                next.add(svc)
            }
            return next
        })
    }

    function toggleModel(key) {
        setVisibleModels(prev => {
            const next = new Set(prev)
            if (next.has(key)) {
                if (next.size > 1) next.delete(key)
            } else {
                next.add(key)
            }
            return next
        })
    }

    const visibleSvcList = services.filter(s => visibleServices.has(s))
    const visibleModelList = MODELS.filter(m => visibleModels.has(m.key))

    return (
        <div>
            <div className="card">
                <div className="card-header">
                    <div className="card-title">
                        {view === 'services' ? 'Anomaly Scores Over Time' : `Model Comparison — ${selectedService}`}
                    </div>
                    <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                        <button onClick={() => setView('services')}
                            style={{ cursor: 'pointer', fontSize: 11, padding: '4px 10px',
                                background: view === 'services' ? 'var(--accent-dim)' : 'transparent',
                                color: view === 'services' ? 'var(--accent)' : 'var(--text-2)',
                                border: `1px solid ${view === 'services' ? 'var(--accent)' : 'var(--border)'}`,
                                borderRadius: 4 }}>
                            By Service
                        </button>
                        <button onClick={() => setView('models')}
                            style={{ cursor: 'pointer', fontSize: 11, padding: '4px 10px',
                                background: view === 'models' ? 'var(--accent-dim)' : 'transparent',
                                color: view === 'models' ? 'var(--accent)' : 'var(--text-2)',
                                border: `1px solid ${view === 'models' ? 'var(--accent)' : 'var(--border)'}`,
                                borderRadius: 4 }}>
                            By Model
                        </button>
                        {view === 'models' && (
                            <select className="sel" value={selectedService}
                                onChange={e => setSelectedService(e.target.value)}>
                                {services.map(s => <option key={s} value={s}>{s}</option>)}
                            </select>
                        )}
                    </div>
                </div>

                {/* Clickable service chips (By Service) or model checkboxes (By Model) */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
                    {view === 'services' ? (
                        services.map(svc => {
                            const active = visibleServices.has(svc)
                            const color = SVC_PALETTE[svc] || '#3b82f6'
                            return (
                                <button key={svc} onClick={() => toggleService(svc)}
                                    style={{
                                        cursor: 'pointer', fontSize: 10, padding: '3px 8px',
                                        borderRadius: 4, border: `1px solid ${active ? color + '60' : 'var(--border)'}`,
                                        background: active ? color + '18' : 'transparent',
                                        color: active ? color : 'var(--text-2)',
                                        transition: 'all 0.15s', fontWeight: active ? 600 : 400,
                                    }}>
                                    <span style={{
                                        display: 'inline-block', width: 6, height: 6,
                                        borderRadius: '50%', background: active ? color : 'var(--text-2)',
                                        marginRight: 5, verticalAlign: 'middle', opacity: active ? 1 : 0.4,
                                    }} />
                                    {svc.replace('-service', '').replace('api-', 'gw')}
                                </button>
                            )
                        })
                    ) : (
                        MODELS.map(m => {
                            const active = visibleModels.has(m.key)
                            return (
                                <label key={m.key} style={{
                                    display: 'flex', alignItems: 'center', gap: 5,
                                    cursor: 'pointer', fontSize: 10, padding: '3px 8px',
                                    borderRadius: 4, border: `1px solid ${active ? m.color + '40' : 'var(--border)'}`,
                                    background: active ? m.color + '18' : 'transparent',
                                    color: active ? m.color : 'var(--text-2)',
                                    transition: 'all 0.15s',
                                }}>
                                    <input type="checkbox" checked={active}
                                        onChange={() => toggleModel(m.key)}
                                        style={{ accentColor: m.color, width: 11, height: 11 }} />
                                    {m.label}
                                </label>
                            )
                        })
                    )}
                </div>

                {(view === 'services' ? history.length : modelHistory.length) < 2 ? (
                    <div className="placeholder">
                        Accumulating data points — chart renders after 2 ticks.
                    </div>
                ) : view === 'services' ? (
                    <ResponsiveContainer width="100%" height={380}>
                        <AreaChart data={history} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                            <defs>
                                {visibleSvcList.map(svc => (
                                    <linearGradient key={svc} id={`g-${svc}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={SVC_PALETTE[svc] || '#3b82f6'} stopOpacity={0.25} />
                                        <stop offset="100%" stopColor={SVC_PALETTE[svc] || '#3b82f6'} stopOpacity={0} />
                                    </linearGradient>
                                ))}
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#243049" vertical={false} />
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
                            <Tooltip contentStyle={{ background: '#131a28', border: '1px solid #243049', borderRadius: 6, fontSize: 11 }} />
                            <ReferenceLine y={0.7} stroke="#f43f5e" strokeDasharray="4 3"
                                label={{ value: '0.7 threshold', fill: '#f43f5e', fontSize: 10, position: 'insideTopRight' }} />
                            {visibleSvcList.map(svc => (
                                <Area key={svc} type="monotone" dataKey={svc}
                                    stroke={SVC_PALETTE[svc] || '#3b82f6'}
                                    fill={`url(#g-${svc})`}
                                    strokeWidth={1.5} dot={false} isAnimationActive={false} />
                            ))}
                        </AreaChart>
                    </ResponsiveContainer>
                ) : (
                    <ResponsiveContainer width="100%" height={380}>
                        <AreaChart data={modelHistory} margin={{ top: 4, right: 8, bottom: 0, left: -10 }}>
                            <defs>
                                {visibleModelList.map(m => (
                                    <linearGradient key={m.key} id={`gm-${m.key}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={m.color} stopOpacity={0.2} />
                                        <stop offset="100%" stopColor={m.color} stopOpacity={0} />
                                    </linearGradient>
                                ))}
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#243049" vertical={false} />
                            <ReferenceArea y1={0} y2={0.5} fill="#10b981" fillOpacity={0.06} />
                            <ReferenceArea y1={0.5} y2={0.65} fill="#f59e0b" fillOpacity={0.06} />
                            <ReferenceArea y1={0.65} y2={0.8} fill="#f97316" fillOpacity={0.06} />
                            <ReferenceArea y1={0.8} y2={1.0} fill="#f43f5e" fillOpacity={0.08} />
                            <XAxis dataKey="t" stroke="#6b7a94" fontSize={10} tickLine={false} />
                            <YAxis domain={[0, 1]} stroke="#6b7a94" fontSize={10} tickLine={false} />
                            <Tooltip contentStyle={{ background: '#131a28', border: '1px solid #243049', borderRadius: 6, fontSize: 11 }} />
                            <ReferenceLine y={0.7} stroke="#f43f5e" strokeDasharray="4 3"
                                label={{ value: '0.7 threshold', fill: '#f43f5e', fontSize: 10, position: 'insideTopRight' }} />
                            {visibleModelList.map(m => (
                                <Area key={m.key} type="monotone" dataKey={m.key}
                                    stroke={m.color} fill={`url(#gm-${m.key})`}
                                    strokeWidth={1.5} dot={false} isAnimationActive={false}
                                    name={m.label} />
                            ))}
                        </AreaChart>
                    </ResponsiveContainer>
                )}
            </div>

            <div className="grid-3" style={{ marginTop: 16 }}>
                {scores.map(s => {
                    const c = SVC_PALETTE[s.service]
                    const sevLabel = (s.severity_label || 'NORMAL').toUpperCase()
                    const sevClass = `sev-${sevLabel.toLowerCase()}`
                    const ens = s.per_ensemble || {}
                    return (
                        <div key={s.service} className="card">
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                                <span style={{ width: 8, height: 8, borderRadius: 2, background: c, flexShrink: 0 }} />
                                <span className="card-title">{s.service}</span>
                                <span className={`severity-badge ${sevClass}`} style={{ marginTop: 0, marginLeft: 'auto' }}>
                                    {sevLabel}
                                </span>
                            </div>
                            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                                <div>
                                    <div className="metric-val" style={{ fontSize: 14, color: '#3b82f6' }}>
                                        {(s.isoforest_score ?? 0).toFixed(3)}
                                    </div>
                                    <div className="metric-lbl">IF</div>
                                </div>
                                <div>
                                    <div className="metric-val" style={{ fontSize: 14, color: '#14b8a6' }}>
                                        {(s.lstm_score ?? 0).toFixed(3)}
                                    </div>
                                    <div className="metric-lbl">LSTM</div>
                                </div>
                                <div>
                                    <div className="metric-val" style={{ fontSize: 14, color: '#8b5cf6' }}>
                                        {(s.ensemble_score ?? 0).toFixed(3)}
                                    </div>
                                    <div className="metric-lbl">Prod</div>
                                </div>
                                <div>
                                    <div className="metric-val" style={{ fontSize: 14, color: '#2ecc71' }}>
                                        {(ens.xgboost_lstm ?? 0).toFixed(3)}
                                    </div>
                                    <div className="metric-lbl">XGB</div>
                                </div>
                                <div>
                                    <div className="metric-val" style={{ fontSize: 14, color: '#f39c12' }}>
                                        {(ens.xgboost_meta ?? 0).toFixed(3)}
                                    </div>
                                    <div className="metric-lbl">META</div>
                                </div>
                            </div>
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
