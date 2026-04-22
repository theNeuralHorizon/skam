import { safeStr, safeNum, safeArr } from '../lib/security'

export default function EventLog({ events, engine, wsConnected }) {
  const cooldowns = safeArr(engine?.services_in_cooldown)
  const evts = safeArr(events)

  function getHealingTime(evt) {
    const ms = safeNum(evt.healing_time_ms)
    if (ms > 0) return (ms / 1000).toFixed(1)
    if (evt.status !== 'completed' || !evt.timestamp) return null
    if (evt.detected_at) {
      const detected = Date.parse(evt.detected_at)
      const recovered = Date.parse(evt.timestamp)
      const diffSec = (recovered - detected) / 1000
      if (Number.isFinite(diffSec) && diffSec > 0) return diffSec.toFixed(1)
    }
    return evt.duration_seconds ? Number(evt.duration_seconds).toFixed(1) : null
  }

  function getSeverityFromEvent(evt) {
    const label = safeStr(evt.severity_label, 16).toUpperCase()
    if (label) return label
    const risk = safeStr(evt.risk_level, 16)
    if (risk === 'high') return 'CRITICAL'
    if (risk === 'medium') return 'HIGH'
    if (risk === 'low') return 'MEDIUM'
    return 'NORMAL'
  }

  function downloadLogs() {
    const data = evts.map((evt) => ({
      id: safeStr(evt.id, 128),
      timestamp: safeStr(evt.timestamp, 64),
      service: safeStr(evt.service, 64),
      action: safeStr(evt.action, 64),
      status: safeStr(evt.status, 32),
      severity: getSeverityFromEvent(evt),
      policy: safeStr(evt.policy_matched, 128),
      healing_time_s: getHealingTime(evt),
      error: evt.error ? safeStr(evt.error, 512) : null,
    }))
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.rel = 'noopener'
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
          <div className="metric-val" style={{ color: 'var(--brand-bright)' }}>
            {safeNum(engine?.policies_loaded)}
          </div>
        </div>
        <div className="card">
          <div className="metric-lbl">Total Recoveries</div>
          <div className="metric-val" style={{ color: 'var(--ok)' }}>
            {safeNum(engine?.total_recoveries)}
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
          <div className="card-title" style={{ marginBottom: 10 }}>
            Cooldown
          </div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {cooldowns.map((svc) => (
              <span key={safeStr(svc, 64)} className="severity-badge sev-medium" style={{ marginTop: 0 }}>
                {safeStr(svc, 64)}
              </span>
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
            <button
              type="button"
              className="mode-toggle"
              onClick={downloadLogs}
              style={{ padding: '6px 12px', fontSize: 11 }}
              title="Download events as JSON"
            >
              ⬇ Export
            </button>
            <span className="card-subtitle">{evts.length} events</span>
          </div>
        </div>

        {evts.length === 0 ? (
          <div className="placeholder">
            No recovery events yet. Events stream here when the decision engine acts.
          </div>
        ) : (
          <div className="event-list">
            {evts.map((evt, i) => {
              const sevLabel = getSeverityFromEvent(evt)
              const sevClass = `sev-${sevLabel.toLowerCase()}`
              const healTime = getHealingTime(evt)
              const tsLabel = (() => {
                const t = Date.parse(evt.timestamp)
                return Number.isFinite(t)
                  ? new Date(t).toLocaleTimeString('en-GB', { hour12: false })
                  : '--:--:--'
              })()

              return (
                <div key={`${safeStr(evt.id, 128)}-${i}`} className="event-row">
                  <div className={`evt-dot ${safeStr(evt.status, 16)}`} />
                  <span className="evt-svc">{safeStr(evt.service, 64)}</span>
                  <span className="evt-act">{safeStr(evt.action, 64)}</span>
                  <span className={`severity-badge ${sevClass}`} style={{ marginTop: 0 }}>
                    {sevLabel}
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-2)' }}>
                    {safeStr(evt.policy_matched, 128)}
                  </span>
                  {healTime != null && (
                    <span className="evt-heal-time" title="Time from detection to recovery">
                      healed {healTime}s
                    </span>
                  )}
                  {evt.error && (
                    <span style={{ fontSize: 10, color: 'var(--critical)' }} title={safeStr(evt.error, 512)}>
                      {safeStr(evt.error, 40)}
                    </span>
                  )}
                  <span className="evt-time">{tsLabel}</span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
