export default function EventLog({ events, engine, wsConnected }) {
    const cooldowns = engine?.services_in_cooldown || []

    function getHealingTime(evt) {
        // Prefer healing_time_ms from backend (most accurate)
        if (evt.healing_time_ms) return (evt.healing_time_ms / 1000).toFixed(1)
        if (evt.status !== 'completed' || !evt.timestamp) return null
        if (evt.detected_at) {
            const detected = new Date(evt.detected_at).getTime()
            const recovered = new Date(evt.timestamp).getTime()
            const diffSec = (recovered - detected) / 1000
            if (diffSec > 0) return diffSec.toFixed(1)
        }
        return evt.duration_seconds ? evt.duration_seconds.toFixed(1) : null
    }

    function getSeverityFromEvent(evt) {
        if (evt.severity_label) return evt.severity_label.toUpperCase()
        const risk = evt.risk_level
        if (risk === 'high') return 'CRITICAL'
        if (risk === 'medium') return 'HIGH'
        if (risk === 'low') return 'MEDIUM'
        return 'NORMAL'
    }

    // Download events as JSON
    function downloadLogs() {
        const data = events.map(evt => ({
            id: evt.id,
            timestamp: evt.timestamp,
            service: evt.service,
            action: evt.action,
            status: evt.status,
            severity: getSeverityFromEvent(evt),
            policy: evt.policy_matched,
            healing_time_s: getHealingTime(evt),
            error: evt.error || null,
        }))
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
        a.download = `skam-recovery-events-${ts}.json`
        a.click()
        URL.revokeObjectURL(url)
    }

    return (
        <div>
            <div className="grid-3" style={{ marginBottom: 16 }}>
                <div className="card">
                    <div className="metric-lbl">Policies Active</div>
                    <div className="metric-val" style={{ color: 'var(--accent)' }}>
                        {engine?.policies_loaded ?? 0}
                    </div>
                </div>
                <div className="card">
                    <div className="metric-lbl">Total Recoveries</div>
                    <div className="metric-val" style={{ color: 'var(--ok)' }}>
                        {engine?.total_recoveries ?? 0}
                    </div>
                </div>
                <div className="card">
                    <div className="metric-lbl">In Cooldown</div>
                    <div className="metric-val" style={{ color: 'var(--warn)' }}>
                        {cooldowns.length}
                    </div>
                </div>
            </div>

            {cooldowns.length > 0 && (
                <div className="card" style={{ marginBottom: 16 }}>
                    <div className="card-title" style={{ marginBottom: 10 }}>Cooldown</div>
                    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        {cooldowns.map(svc => (
                            <span key={svc} className="tag medium">{svc}</span>
                        ))}
                    </div>
                </div>
            )}

            <div className="card">
                <div className="card-header">
                    <div className="card-title">Recovery Events</div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span className="ws-status">
                            <span className={`ws-dot ${wsConnected ? 'connected' : 'disconnected'}`} />
                            {wsConnected ? 'Live' : 'Reconnecting'}
                        </span>
                        <button className="sel" onClick={downloadLogs}
                            style={{ cursor: 'pointer', fontSize: 11, padding: '4px 10px' }}
                            title="Download events as JSON">
                            Download Logs
                        </button>
                        <span className="card-subtitle">
                            {events.length} events
                        </span>
                    </div>
                </div>

                {events.length === 0 ? (
                    <div className="placeholder">
                        No recovery events yet. Events stream here when the decision engine acts.
                    </div>
                ) : (
                    <div className="event-list">
                        {events.map((evt, i) => {
                            const sevLabel = getSeverityFromEvent(evt)
                            const sevClass = `sev-${sevLabel.toLowerCase()}`
                            const healTime = getHealingTime(evt)

                            return (
                                <div key={`${evt.id}-${i}`} className="event-row">
                                    <div className={`evt-dot ${evt.status}`} />
                                    <span className="evt-svc">{evt.service}</span>
                                    <span className="evt-act">{evt.action}</span>
                                    <span className={`severity-badge ${sevClass}`} style={{ marginTop: 0 }}>
                                        {sevLabel}
                                    </span>
                                    <span style={{ fontSize: 11, color: 'var(--text-2)' }}>
                                        {evt.policy_matched}
                                    </span>
                                    {healTime != null && (
                                        <span className="evt-heal-time" title="Time from detection to recovery">
                                            healed {healTime}s
                                        </span>
                                    )}
                                    {evt.error && (
                                        <span style={{ fontSize: 10, color: 'var(--err)' }} title={evt.error}>
                                            {evt.error.slice(0, 40)}
                                        </span>
                                    )}
                                    <span className="evt-time">
                                        {new Date(evt.timestamp).toLocaleTimeString('en-GB', { hour12: false })}
                                    </span>
                                </div>
                            )
                        })}
                    </div>
                )}
            </div>
        </div>
    )
}
