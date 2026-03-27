import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Legend,
} from 'recharts'
import type { AnomalyEvent } from '../types'

interface Props {
  anomalies: AnomalyEvent[]
}

const SERVICE_COLORS: Record<string, string> = {
  'api-gateway': '#3b82f6',
  'user-service': '#8b5cf6',
  'product-service': '#10b981',
  'order-service': '#f59e0b',
  'payment-service': '#ef4444',
  'notification-service': '#06b6d4',
}

export function AnomalyTimeline({ anomalies }: Props) {
  // Group by timestamp, pivot services into columns
  const grouped = new Map<string, Record<string, number>>()
  for (const a of anomalies) {
    const key = a.timestamp.slice(11, 19) // HH:MM:SS
    if (!grouped.has(key)) grouped.set(key, { time: key } as any)
    grouped.get(key)![a.service] = a.combinedScore
  }
  const data = Array.from(grouped.values()).slice(-40)

  const services = [...new Set(anomalies.map((a) => a.service))]

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3">Anomaly Scores</h2>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data}>
          <XAxis
            dataKey="time"
            tick={{ fontSize: 10, fill: '#6b7280' }}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 10, fill: '#6b7280' }}
            width={30}
          />
          <Tooltip
            contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
            labelStyle={{ color: '#9ca3af' }}
          />
          <ReferenceLine
            y={0.7}
            stroke="#ef4444"
            strokeDasharray="4 4"
            label={{ value: 'Threshold', fill: '#ef4444', fontSize: 10 }}
          />
          {services.map((svc) => (
            <Line
              key={svc}
              type="monotone"
              dataKey={svc}
              stroke={SERVICE_COLORS[svc] || '#6b7280'}
              strokeWidth={1.5}
              dot={false}
              connectNulls
            />
          ))}
          <Legend
            wrapperStyle={{ fontSize: 10, color: '#9ca3af' }}
            formatter={(value: string) => value.replace('-service', '')}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
