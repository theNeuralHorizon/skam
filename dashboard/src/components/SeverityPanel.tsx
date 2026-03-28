import type { AnomalyEvent, SeverityLevel } from '../types'

interface Props {
  anomalies: AnomalyEvent[]
}

const SEVERITY_CONFIG: Record<SeverityLevel, {
  color: string
  bgColor: string
  borderColor: string
  icon: string
  responseTarget: string
}> = {
  normal:   { color: 'text-green-400',  bgColor: 'bg-green-500/10',  borderColor: 'border-green-500/30', icon: '●', responseTarget: '—' },
  low:      { color: 'text-blue-400',   bgColor: 'bg-blue-500/10',   borderColor: 'border-blue-500/30',  icon: '▲', responseTarget: '< 2 min' },
  medium:   { color: 'text-yellow-400', bgColor: 'bg-yellow-500/10', borderColor: 'border-yellow-500/30', icon: '▲', responseTarget: '< 1 min' },
  high:     { color: 'text-orange-400', bgColor: 'bg-orange-500/10', borderColor: 'border-orange-500/30', icon: '▲▲', responseTarget: '< 30s' },
  critical: { color: 'text-red-400',    bgColor: 'bg-red-500/10',    borderColor: 'border-red-500/30',    icon: '▲▲▲', responseTarget: '< 15s' },
}

export function SeverityPanel({ anomalies }: Props) {
  // Get latest anomaly per service
  const latestByService = new Map<string, AnomalyEvent>()
  for (const a of anomalies) {
    const existing = latestByService.get(a.service)
    if (!existing || a.timestamp > existing.timestamp) {
      latestByService.set(a.service, a)
    }
  }

  // Count by severity
  const counts: Record<SeverityLevel, number> = { normal: 0, low: 0, medium: 0, high: 0, critical: 0 }
  for (const a of latestByService.values()) {
    counts[a.severity || 'normal']++
  }

  // Services with active anomalies sorted by severity (highest first)
  const activeAnomalies = Array.from(latestByService.values())
    .filter((a) => (a.severityLevel || 0) > 0)
    .sort((a, b) => (b.severityLevel || 0) - (a.severityLevel || 0))

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3">Severity Overview</h2>

      {/* Severity count badges */}
      <div className="flex gap-2 mb-3">
        {(['critical', 'high', 'medium', 'low', 'normal'] as SeverityLevel[]).map((sev) => {
          const cfg = SEVERITY_CONFIG[sev]
          return (
            <div
              key={sev}
              className={`flex items-center gap-1.5 px-2 py-1 rounded border ${cfg.borderColor} ${cfg.bgColor}`}
            >
              <span className={`text-xs font-bold ${cfg.color}`}>{counts[sev]}</span>
              <span className="text-[10px] text-gray-500 uppercase">{sev}</span>
            </div>
          )
        })}
      </div>

      {/* Active anomaly details */}
      {activeAnomalies.length === 0 ? (
        <p className="text-xs text-gray-600 text-center py-3">All services operating normally</p>
      ) : (
        <div className="space-y-1.5">
          {activeAnomalies.map((a) => {
            const cfg = SEVERITY_CONFIG[a.severity || 'low']
            return (
              <div
                key={a.service}
                className={`flex items-center justify-between p-2 rounded border ${cfg.borderColor} ${cfg.bgColor}`}
              >
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-bold uppercase ${cfg.color}`}>
                    {a.severity}
                  </span>
                  <span className="text-xs text-white font-medium">{a.service}</span>
                </div>
                <div className="flex items-center gap-3 text-[10px]">
                  <span className="text-gray-400">
                    score: <span className={cfg.color}>{a.combinedScore.toFixed(3)}</span>
                  </span>
                  <span className="text-gray-500">
                    {a.consecutiveWindows || 0} windows
                  </span>
                  {a.scoreVelocity > 0 && (
                    <span className="text-red-400">↑ {(a.scoreVelocity * 60).toFixed(2)}/min</span>
                  )}
                  <span className="text-gray-500">
                    target: {cfg.responseTarget}
                  </span>
                  {a.topContributors?.[0] && (
                    <span className="text-gray-400">
                      cause: {a.topContributors[0].feature}
                    </span>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
