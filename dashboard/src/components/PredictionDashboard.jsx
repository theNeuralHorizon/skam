import { useState, useEffect } from 'react'

const TYPE_LABELS = {
  score_trajectory: 'Score Trajectory',
  capacity_exhaustion: 'Capacity Exhaustion',
  repeat_failure: 'Repeat Failure',
}

const EVENT_LABELS = {
  threshold_breach: 'Threshold Breach',
  oom_kill: 'OOM Kill',
  recurring_anomaly: 'Recurring Anomaly',
}

function confidenceColor(c) {
  if (c >= 0.85) return 'pred-critical'
  if (c >= 0.7) return 'pred-high'
  return 'pred-medium'
}

function velocityArrow(v) {
  if (v > 0.001) return { symbol: '\u2191', cls: 'vel-up' }      // ↑
  if (v < -0.001) return { symbol: '\u2193', cls: 'vel-down' }   // ↓
  return { symbol: '\u2192', cls: 'vel-stable' }                  // →
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return 'Now'
  if (seconds < 60) return `${Math.round(seconds)}s`
  return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

export default function PredictionDashboard({ scores, predictions }) {
  const [now, setNow] = useState(Date.now())

  // Tick every second for countdown updates
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  const activePredictions = predictions.filter(p => p.confidence >= 0.4)
  const highConfidence = activePredictions.filter(p => p.confidence >= 0.6)

  return (
    <div className="prediction-dashboard">
      {/* Summary stats */}
      <div className="pred-summary">
        <div className="pred-stat-card">
          <div className="pred-stat-value">{activePredictions.length}</div>
          <div className="pred-stat-label">Active Predictions</div>
        </div>
        <div className="pred-stat-card">
          <div className="pred-stat-value pred-high-text">{highConfidence.length}</div>
          <div className="pred-stat-label">High Confidence</div>
        </div>
        <div className="pred-stat-card">
          <div className="pred-stat-value">
            {new Set(activePredictions.map(p => p.service)).size}
          </div>
          <div className="pred-stat-label">Services at Risk</div>
        </div>
      </div>

      {/* Active predictions */}
      <div className="card pred-section">
        <div className="card-header">
          <span className="card-title">Active Predictions</span>
          <span className="card-subtitle">{activePredictions.length} prediction{activePredictions.length !== 1 ? 's' : ''}</span>
        </div>

        {activePredictions.length === 0 ? (
          <div className="placeholder">No active predictions — all services are trending stable</div>
        ) : (
          <div className="pred-cards">
            {activePredictions.map((p, i) => (
              <div key={`${p.service}-${p.prediction_type}-${i}`} className={`pred-card ${confidenceColor(p.confidence)}`}>
                <div className="pred-card-header">
                  <span className="pred-card-service">{p.service}</span>
                  <span className={`pred-card-type tag ${p.confidence >= 0.7 ? 'high' : 'medium'}`}>
                    {TYPE_LABELS[p.prediction_type] || p.prediction_type}
                  </span>
                </div>

                <div className="pred-card-event">
                  {EVENT_LABELS[p.predicted_event] || p.predicted_event}
                </div>

                <div className="pred-card-details">
                  <div className="pred-detail">
                    <span className="pred-detail-label">Time to Event</span>
                    <span className="pred-detail-value pred-countdown">
                      {formatTime(p.time_to_event_seconds)}
                    </span>
                  </div>
                  <div className="pred-detail">
                    <span className="pred-detail-label">Confidence</span>
                    <div className="pred-confidence-bar">
                      <div
                        className={`pred-confidence-fill ${confidenceColor(p.confidence)}`}
                        style={{ width: `${Math.round((p.confidence || 0) * 100)}%` }}
                      />
                    </div>
                    <span className="pred-detail-value">{(p.confidence * 100).toFixed(0)}%</span>
                  </div>
                  {p.recommended_action && (
                    <div className="pred-detail">
                      <span className="pred-detail-label">Recommended</span>
                      <span className="pred-detail-value pred-action">{p.recommended_action}</span>
                    </div>
                  )}
                </div>

                {p.details && Object.keys(p.details).length > 0 && (
                  <div className="pred-card-extra">
                    {Object.entries(p.details).map(([k, v]) => (
                      <span key={k} className="pred-extra-item">
                        {k.replace(/_/g, ' ')}: {typeof v === 'number' ? v.toFixed(3) : String(v)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Score velocity panel */}
      <div className="card pred-section">
        <div className="card-header">
          <span className="card-title">Score Velocity</span>
          <span className="card-subtitle">Rate of anomaly score change per service</span>
        </div>

        <div className="velocity-grid">
          {scores.map(s => {
            const vel = s.score_velocity || 0
            const arrow = velocityArrow(vel)
            const hasPrediction = predictions.some(p => p.service === s.service)
            return (
              <div key={s.service} className={`velocity-card ${hasPrediction ? 'has-prediction' : ''}`}>
                <div className="velocity-service">{s.service}</div>
                <div className="velocity-row">
                  <span className={`velocity-arrow ${arrow.cls}`}>{arrow.symbol}</span>
                  <span className="velocity-value">{vel.toFixed(6)}/s</span>
                </div>
                <div className="velocity-score">
                  Score: <span className={s.ensemble_score > 0.7 ? 'score-crit' : s.ensemble_score > 0.5 ? 'score-warn' : 'score-safe'}>
                    {s.ensemble_score?.toFixed(4)}
                  </span>
                </div>
                {hasPrediction && <div className="velocity-warning-dot" />}
              </div>
            )
          })}
        </div>
      </div>

      {/* Prediction history */}
      <div className="card pred-section">
        <div className="card-header">
          <span className="card-title">Prediction History</span>
          <span className="card-subtitle">Recent predictions log</span>
        </div>

        {predictions.length === 0 ? (
          <div className="placeholder">No predictions recorded yet</div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Service</th>
                <th>Type</th>
                <th>Event</th>
                <th>Time to Event</th>
                <th>Confidence</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {predictions.slice(0, 20).map((p, i) => (
                <tr key={i}>
                  <td className="svc-col">{p.service}</td>
                  <td>{TYPE_LABELS[p.prediction_type] || p.prediction_type}</td>
                  <td>{EVENT_LABELS[p.predicted_event] || p.predicted_event}</td>
                  <td>{formatTime(p.time_to_event_seconds)}</td>
                  <td>
                    <span className={`tag ${p.confidence >= 0.7 ? 'high' : p.confidence >= 0.5 ? 'medium' : 'low'}`}>
                      {(p.confidence * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="pred-action">{p.recommended_action || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
