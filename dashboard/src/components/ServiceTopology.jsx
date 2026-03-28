import { useMemo } from 'react'

const EDGES = [
    ['api-gateway', 'user-service'],
    ['api-gateway', 'product-service'],
    ['api-gateway', 'order-service'],
    ['api-gateway', 'cart-service'],
    ['order-service', 'payment-service'],
    ['order-service', 'notification-service'],
    ['payment-service', 'notification-service'],
]

const POSITIONS = {
    'api-gateway': { x: 400, y: 45 },
    'user-service': { x: 140, y: 145 },
    'product-service': { x: 300, y: 145 },
    'order-service': { x: 500, y: 145 },
    'cart-service': { x: 660, y: 145 },
    'payment-service': { x: 350, y: 245 },
    'notification-service': { x: 550, y: 245 },
}

const ALL_SERVICES = Object.keys(POSITIONS)

const SEVERITY_COLORS = {
    NORMAL: { fill: 'var(--sev-normal)', bg: 'var(--sev-normal-dim)' },
    LOW: { fill: 'var(--sev-low)', bg: 'var(--sev-low-dim)' },
    MEDIUM: { fill: 'var(--sev-medium)', bg: 'var(--sev-medium-dim)' },
    HIGH: { fill: 'var(--sev-high)', bg: 'var(--sev-high-dim)' },
    CRITICAL: { fill: 'var(--sev-critical)', bg: 'var(--sev-critical-dim)' },
}

function scoreClass(v) {
    if (v > 0.7) return 'crit'
    if (v > 0.4) return 'warn'
    return 'safe'
}

function getSeverityLabel(score) {
    // Use backend severity_label if present, otherwise derive from score
    if (score?.severity_label) return score.severity_label
    const v = score?.ensemble_score ?? 0
    if (v >= 0.8) return 'CRITICAL'
    if (v >= 0.65) return 'HIGH'
    if (v >= 0.5) return 'MEDIUM'
    if (v >= 0.3) return 'LOW'
    return 'NORMAL'
}

export default function ServiceTopology({ scores }) {
    const byService = useMemo(() => {
        const m = {}
        for (const s of scores) m[s.service] = s
        return m
    }, [scores])

    return (
        <div>
            <div className="card" style={{ marginBottom: 16 }}>
                <div className="card-header">
                    <div className="card-title">Service Map</div>
                    <div className="card-subtitle">{scores.length} services reporting</div>
                </div>

                <svg viewBox="0 0 800 300" style={{ width: '100%', height: 280 }}>
                    {EDGES.map(([a, b], i) => {
                        const pa = POSITIONS[a], pb = POSITIONS[b]
                        const hot = byService[a]?.is_anomaly || byService[b]?.is_anomaly
                        return (
                            <line key={i}
                                x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                                stroke={hot ? 'var(--err)' : 'var(--border)'}
                                strokeWidth={hot ? 1.5 : 1}
                                strokeDasharray={hot ? '5 3' : 'none'}
                                opacity={hot ? 0.7 : 0.4}
                            />
                        )
                    })}

                    {ALL_SERVICES.map(svc => {
                        const p = POSITIONS[svc]
                        const d = byService[svc]
                        const v = d?.ensemble_score ?? 0
                        const bad = d?.is_anomaly ?? false
                        const sevLabel = getSeverityLabel(d)
                        const sevColor = SEVERITY_COLORS[sevLabel] || SEVERITY_COLORS.NORMAL
                        const fill = bad ? 'var(--err)' : v > 0.4 ? 'var(--warn)' : 'var(--ok)'

                        return (
                            <g key={svc}>
                                <circle cx={p.x} cy={p.y} r={22}
                                    fill="var(--bg-2)" stroke={fill} strokeWidth={1.5} />
                                <text x={p.x} y={p.y + 1} textAnchor="middle"
                                    dominantBaseline="middle"
                                    fill={fill} fontSize={10} fontWeight={600}
                                    fontFamily="JetBrains Mono, monospace">
                                    {v.toFixed(2)}
                                </text>
                                <text x={p.x} y={p.y + 36} textAnchor="middle"
                                    fill="var(--text-2)" fontSize={10}>
                                    {svc.replace('-service', '').replace('api-', 'gateway')}
                                </text>
                                {/* Severity badge on SVG node */}
                                <rect
                                    x={p.x - 20} y={p.y - 38}
                                    width={40} height={14}
                                    rx={3} ry={3}
                                    fill={sevColor.bg}
                                    stroke={sevColor.fill}
                                    strokeWidth={0.5}
                                    opacity={0.9}
                                />
                                <text
                                    x={p.x} y={p.y - 28}
                                    textAnchor="middle"
                                    dominantBaseline="middle"
                                    fill={sevColor.fill}
                                    fontSize={7}
                                    fontWeight={700}
                                    fontFamily="Inter, sans-serif"
                                    letterSpacing="0.5"
                                >
                                    {sevLabel}
                                </text>
                            </g>
                        )
                    })}
                </svg>
            </div>

            <div className="topology-grid">
                {ALL_SERVICES.map(svc => {
                    const d = byService[svc]
                    const v = d?.ensemble_score ?? 0
                    const bad = d?.is_anomaly ?? false
                    const cls = bad ? 'anomaly' : v > 0.4 ? 'warning' : ''
                    const sevLabel = getSeverityLabel(d)
                    const sevClass = `sev-${sevLabel.toLowerCase()}`

                    return (
                        <div key={svc} className={`service-node ${cls}`}>
                            <div className="svc-name">{svc}</div>
                            <div className={`svc-score score-${scoreClass(v)}`}>
                                {v.toFixed(3)}
                            </div>
                            <div className="svc-label">anomaly score</div>
                            <div className={`severity-badge ${sevClass}`}>
                                {sevLabel}
                            </div>
                            {d?.features && (
                                <div className="svc-detail">
                                    <span>{(d.features.request_rate ?? 0).toFixed(1)} req/s</span>
                                    <span>{((d.features.error_ratio ?? 0) * 100).toFixed(1)}% errors</span>
                                    <span>p99 {((d.features.latency_p99 ?? 0) * 1000).toFixed(0)}ms</span>
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>
        </div>
    )
}
