import { useState, useCallback } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
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

const SHORT_NAMES: Record<string, string> = {
  'api-gateway': 'gateway',
  'user-service': 'user',
  'product-service': 'product',
  'order-service': 'order',
  'payment-service': 'payment',
  'notification-service': 'notif',
}

export function AnomalyTimeline({ anomalies }: Props) {
  const services = [...new Set(anomalies.map((a) => a.service))]
  const [hiddenServices, setHiddenServices] = useState<Set<string>>(new Set())

  const toggleService = useCallback((service: string) => {
    setHiddenServices((prev) => {
      const next = new Set(prev)
      if (next.has(service)) {
        next.delete(service)
      } else {
        next.add(service)
      }
      return next
    })
  }, [])

  // Group by timestamp, pivot services into columns
  const grouped = new Map<string, Record<string, number>>()
  for (const a of anomalies) {
    const key = a.timestamp.slice(11, 19) // HH:MM:SS
    if (!grouped.has(key)) grouped.set(key, { time: key } as any)
    grouped.get(key)![a.service] = a.combinedScore
  }
  const data = Array.from(grouped.values()).slice(-40)

  const visibleServices = services.filter((s) => !hiddenServices.has(s))

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3">Anomaly Scores</h2>

      {/* Clickable service toggles */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {services.map((svc) => {
          const active = !hiddenServices.has(svc)
          const color = SERVICE_COLORS[svc] || '#6b7280'
          return (
            <button
              key={svc}
              onClick={() => toggleService(svc)}
              className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-medium transition-all border"
              style={{
                borderColor: active ? color : '#374151',
                backgroundColor: active ? `${color}20` : 'transparent',
                color: active ? color : '#6b7280',
                opacity: active ? 1 : 0.5,
              }}
            >
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: active ? color : '#4b5563' }}
              />
              {SHORT_NAMES[svc] || svc}
            </button>
          )
        })}
        {hiddenServices.size > 0 && (
          <button
            onClick={() => setHiddenServices(new Set())}
            className="px-2 py-0.5 rounded text-[10px] text-gray-500 hover:text-gray-300 border border-gray-700 hover:border-gray-500 transition-colors"
          >
            show all
          </button>
        )}
      </div>

      <ResponsiveContainer width="100%" height={220}>
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
            formatter={(value: number, name: string) => [
              value.toFixed(3),
              SHORT_NAMES[name] || name,
            ]}
          />
          <ReferenceLine
            y={0.7}
            stroke="#ef4444"
            strokeDasharray="4 4"
            label={{ value: 'Threshold', fill: '#ef4444', fontSize: 10 }}
          />
          {visibleServices.map((svc) => (
            <Line
              key={svc}
              type="monotone"
              dataKey={svc}
              stroke={SERVICE_COLORS[svc] || '#6b7280'}
              strokeWidth={1.5}
              dot={false}
              connectNulls
              animationDuration={300}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
