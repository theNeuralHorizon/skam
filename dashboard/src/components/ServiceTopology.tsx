import type { ServiceHealth, AnomalyEvent, SeverityLevel } from '../types'

interface Props {
  services: ServiceHealth[]
  latestAnomalies?: Map<string, AnomalyEvent>
}

const SEVERITY_STYLES: Record<SeverityLevel, { badge: string; pulse: boolean }> = {
  normal:   { badge: 'bg-green-600 text-white', pulse: false },
  low:      { badge: 'bg-blue-600 text-white', pulse: false },
  medium:   { badge: 'bg-yellow-600 text-black', pulse: false },
  high:     { badge: 'bg-orange-500 text-white', pulse: true },
  critical: { badge: 'bg-red-600 text-white', pulse: true },
}

const SERVICE_POSITIONS: Record<string, { x: number; y: number }> = {
  'api-gateway':          { x: 250, y: 30 },
  'user-service':         { x: 80,  y: 140 },
  'product-service':      { x: 250, y: 140 },
  'order-service':        { x: 420, y: 140 },
  'payment-service':      { x: 170, y: 250 },
  'notification-service': { x: 370, y: 250 },
}

const CONNECTIONS = [
  ['api-gateway', 'user-service'],
  ['api-gateway', 'product-service'],
  ['api-gateway', 'order-service'],
  ['order-service', 'payment-service'],
  ['order-service', 'notification-service'],
]

const STATUS_COLORS = {
  healthy: { border: 'border-green-500', bg: 'bg-green-500/10', dot: 'bg-green-500' },
  degraded: { border: 'border-yellow-500', bg: 'bg-yellow-500/10', dot: 'bg-yellow-500' },
  unhealthy: { border: 'border-red-500', bg: 'bg-red-500/10', dot: 'bg-red-500' },
}

export function ServiceTopology({ services, latestAnomalies }: Props) {
  const serviceMap = Object.fromEntries(services.map((s) => [s.name, s]))

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3">Service Topology</h2>
      <div className="relative" style={{ height: 320 }}>
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 530 320">
          {CONNECTIONS.map(([from, to]) => {
            const fp = SERVICE_POSITIONS[from]
            const tp = SERVICE_POSITIONS[to]
            return (
              <line
                key={`${from}-${to}`}
                x1={fp.x + 50} y1={fp.y + 25}
                x2={tp.x + 50} y2={tp.y + 25}
                stroke="#374151"
                strokeWidth="1.5"
                strokeDasharray="4 4"
              />
            )
          })}
        </svg>

        {services.map((svc) => {
          const pos = SERVICE_POSITIONS[svc.name]
          if (!pos) return null
          const colors = STATUS_COLORS[svc.status]
          const anomaly = latestAnomalies?.get(svc.name)
          const severity = anomaly?.severity || 'normal'
          const sevStyle = SEVERITY_STYLES[severity]

          return (
            <div
              key={svc.name}
              className={`absolute rounded-lg border ${colors.border} ${colors.bg} p-2 w-[120px]`}
              style={{ left: pos.x, top: pos.y }}
            >
              <div className="flex items-center gap-1.5 mb-1">
                <div className={`w-2 h-2 rounded-full ${colors.dot} ${sevStyle.pulse ? 'animate-pulse' : ''}`} />
                <span className="text-xs font-medium text-white truncate">
                  {svc.name.replace('-service', '').replace('api-', '')}
                </span>
                {severity !== 'normal' && (
                  <span className={`text-[8px] px-1 py-px rounded font-bold uppercase ${sevStyle.badge}`}>
                    {severity}
                  </span>
                )}
              </div>
              <div className="flex justify-between text-[10px] text-gray-400">
                <span>{svc.requestRate.toFixed(0)} rps</span>
                <span className={svc.errorRate > 0.05 ? 'text-red-400' : ''}>
                  {(svc.errorRate * 100).toFixed(1)}% err
                </span>
              </div>
              {anomaly && anomaly.topContributors?.length > 0 && severity !== 'normal' && (
                <div className="mt-1 text-[9px] text-gray-500">
                  <span className="text-gray-400">cause:</span>{' '}
                  {anomaly.topContributors[0].feature} ({anomaly.topContributors[0].direction})
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
