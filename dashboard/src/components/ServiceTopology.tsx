import type { ServiceHealth } from '../types'

interface Props {
  services: ServiceHealth[]
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

export function ServiceTopology({ services }: Props) {
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

          return (
            <div
              key={svc.name}
              className={`absolute rounded-lg border ${colors.border} ${colors.bg} p-2 w-[120px]`}
              style={{ left: pos.x, top: pos.y }}
            >
              <div className="flex items-center gap-1.5 mb-1">
                <div className={`w-2 h-2 rounded-full ${colors.dot}`} />
                <span className="text-xs font-medium text-white truncate">
                  {svc.name.replace('-service', '').replace('api-', '')}
                </span>
              </div>
              <div className="flex justify-between text-[10px] text-gray-400">
                <span>{svc.requestRate.toFixed(0)} rps</span>
                <span className={svc.errorRate > 0.05 ? 'text-red-400' : ''}>
                  {(svc.errorRate * 100).toFixed(1)}% err
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
