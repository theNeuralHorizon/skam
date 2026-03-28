export default function EventLog({ events, engine }) {
    const cooldowns = engine?.services_in_cooldown || []

    // Compute healing time: time from detection (anomaly event) to recovery (completed event)
    function getHealingTime(evt) {
        if (evt.status !== 'completed' || !evt.timestamp) return null
        if (!evt.detected_at && !evt.duration_seconds) return null
        if (evt.detected_at) {
            const detected = new Date(evt.detected_at).getTime()
            const recovered = new Date(evt.timestamp).getTime()
            const diffSec = Math.round((recovered - detected) / 1000)
            if (diffSec > 0) return diffSec
        }
        return evt.duration_seconds || null
    }

    function getSeverityFromEvent(evt) {
        // Use severity_label from the event if available
        if (evt.severity_label) return evt.severity_label
        // Derive from risk_level as fallback
        if (evt.risk_level === 'high') return 'CRITICAL'
        if (evt.risk_level === 'medium') return 'HIGH'
        if (evt.risk_level === 'low') return 'MEDIUM'
        return 'NORMAL'
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
                    <div className="card-subtitle">{events.length} events &middot; live via WebSocket</div>
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
                                <div key={evt.id || i} className="event-row">
                                    <div className={`evt-dot ${evt.status}`} />
                                    <span className="evt-svc">{evt.service}</span>
                                    <span className="evt-act">{evt.action}</span>
                                    <span className={`tag ${evt.risk_level}`}>{evt.risk_level}</span>
                                    <span className={`severity-badge ${sevClass}`} style={{ marginTop: 0 }}>
                                        {sevLabel}
                                    </span>
                                    <span style={{ fontSize: 11, color: 'var(--text-2)' }}>
                                        {evt.policy_matched}
                                    </span>
                                    {evt.duration_seconds != null && (
                                        <span style={{ fontSize: 11, color: 'var(--teal)', fontFamily: "'JetBrains Mono', monospace" }}>
                                            {evt.duration_seconds}s
                                        </span>
                                    )}
                                    {healTime != null && (
                                        <span className="evt-heal-time" title="Healing time (detection to recovery)">
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
