import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { ServiceHealth } from '../types'

interface Props {
  services: ServiceHealth[]
}

export function MetricsPanel({ services }: Props) {
  const data = services.map((s) => ({
    name: s.name.replace('-service', '').replace('api-', 'gw'),
    cpu: Math.round(s.cpuUsage),
    memory: Math.round(s.memoryUsage),
    rps: Math.round(s.requestRate),
    errors: Math.round(s.errorRate * 100 * 10) / 10,
  }))

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3">Live Metrics</h2>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-gray-500 mb-1">CPU & Memory %</p>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={data}>
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} domain={[0, 100]} width={25} />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
              />
              <Bar dataKey="cpu" fill="#3b82f6" name="CPU" radius={[2, 2, 0, 0]} />
              <Bar dataKey="memory" fill="#8b5cf6" name="Memory" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div>
          <p className="text-xs text-gray-500 mb-1">Request Rate & Error %</p>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={data}>
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} width={25} />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
              />
              <Bar dataKey="rps" fill="#10b981" name="RPS" radius={[2, 2, 0, 0]} />
              <Bar dataKey="errors" fill="#ef4444" name="Error %" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
