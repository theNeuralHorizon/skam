import { useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts'
import type { EnsembleComparison as EnsembleData } from '../types'

interface Props {
  ensembles: EnsembleData[]
}

type ViewMode = 'bar' | 'radar' | 'table'

const METRIC_LABELS: Record<string, string> = {
  aucRoc: 'AUC-ROC',
  aucPr: 'AUC-PR',
  f1Best: 'F1 (best)',
  mcc: 'MCC',
  cohensKappa: "Cohen's κ",
  brierScore: 'Brier Score',
  scoreSeparation: 'Score Sep.',
}

const ENSEMBLE_COLORS = [
  '#3b82f6', // blue
  '#10b981', // green
  '#f59e0b', // amber
  '#8b5cf6', // purple
  '#ef4444', // red
  '#06b6d4', // cyan
]

export function EnsembleComparison({ ensembles }: Props) {
  const [view, setView] = useState<ViewMode>('bar')
  const [selectedMetric, setSelectedMetric] = useState<string>('aucRoc')

  if (ensembles.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <h2 className="text-sm font-semibold text-gray-300 mb-3">Ensemble Comparison</h2>
        <p className="text-xs text-gray-600 text-center py-8">
          No ensemble benchmark data available. Run the benchmark to compare models.
        </p>
      </div>
    )
  }

  // Bar chart data: one bar per ensemble for the selected metric
  const barData = ensembles.map((e) => ({
    name: e.name.replace(/_/g, ' '),
    value: (e as any)[selectedMetric] ?? 0,
  }))

  // Radar chart data: all metrics normalized to 0-1 range
  const radarMetrics = ['aucRoc', 'aucPr', 'f1Best', 'mcc', 'scoreSeparation']
  const radarData = radarMetrics.map((metric) => {
    const point: Record<string, any> = { metric: METRIC_LABELS[metric] || metric }
    ensembles.forEach((e) => {
      let val = (e as any)[metric] ?? 0
      // Normalize MCC from [-1,1] to [0,1]
      if (metric === 'mcc') val = (val + 1) / 2
      point[e.name] = Math.max(0, Math.min(1, val))
    })
    return point
  })

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-300">Ensemble Comparison</h2>
        <div className="flex gap-1">
          {(['bar', 'radar', 'table'] as ViewMode[]).map((v) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-2 py-0.5 text-[10px] rounded font-medium transition-colors ${
                view === v
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-gray-300'
              }`}
            >
              {v.charAt(0).toUpperCase() + v.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {view === 'bar' && (
        <>
          <div className="flex flex-wrap gap-1 mb-2">
            {Object.entries(METRIC_LABELS).map(([key, label]) => (
              <button
                key={key}
                onClick={() => setSelectedMetric(key)}
                className={`px-2 py-0.5 text-[10px] rounded transition-colors ${
                  selectedMetric === key
                    ? 'bg-blue-600/30 text-blue-400 border border-blue-500'
                    : 'bg-gray-800 text-gray-500 border border-gray-700 hover:text-gray-400'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={barData} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 10 }} />
              <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#e5e7eb' }}
                itemStyle={{ color: '#60a5fa' }}
              />
              <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </>
      )}

      {view === 'radar' && (
        <ResponsiveContainer width="100%" height={250}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#374151" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#9ca3af', fontSize: 9 }} />
            <PolarRadiusAxis domain={[0, 1]} tick={{ fill: '#6b7280', fontSize: 8 }} />
            {ensembles.slice(0, 4).map((e, i) => (
              <Radar
                key={e.name}
                name={e.name.replace(/_/g, ' ')}
                dataKey={e.name}
                stroke={ENSEMBLE_COLORS[i]}
                fill={ENSEMBLE_COLORS[i]}
                fillOpacity={0.1}
              />
            ))}
            <Legend wrapperStyle={{ fontSize: 10, color: '#9ca3af' }} />
          </RadarChart>
        </ResponsiveContainer>
      )}

      {view === 'table' && (
        <div className="overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="border-b border-gray-800">
                <th className="text-left text-gray-400 py-1 px-1">Ensemble</th>
                <th className="text-right text-gray-400 py-1 px-1">AUC-ROC</th>
                <th className="text-right text-gray-400 py-1 px-1">F1</th>
                <th className="text-right text-gray-400 py-1 px-1">MCC</th>
                <th className="text-right text-gray-400 py-1 px-1">κ</th>
                <th className="text-right text-gray-400 py-1 px-1">Brier</th>
                <th className="text-right text-gray-400 py-1 px-1">Sep.</th>
                <th className="text-right text-gray-400 py-1 px-1">Cold</th>
                <th className="text-right text-gray-400 py-1 px-1">Speed</th>
              </tr>
            </thead>
            <tbody>
              {ensembles.map((e, i) => (
                <tr key={e.name} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="py-1 px-1 text-gray-300 font-medium">{e.name.replace(/_/g, ' ')}</td>
                  <td className="text-right py-1 px-1 text-blue-400">{e.aucRoc.toFixed(3)}</td>
                  <td className="text-right py-1 px-1 text-green-400">{e.f1Best.toFixed(3)}</td>
                  <td className="text-right py-1 px-1 text-purple-400">{e.mcc.toFixed(3)}</td>
                  <td className="text-right py-1 px-1 text-yellow-400">{e.cohensKappa.toFixed(3)}</td>
                  <td className="text-right py-1 px-1 text-cyan-400">{e.brierScore.toFixed(3)}</td>
                  <td className="text-right py-1 px-1 text-gray-400">{e.scoreSeparation.toFixed(3)}</td>
                  <td className="text-right py-1 px-1 text-gray-500">{e.coldStartSamples}</td>
                  <td className="text-right py-1 px-1 text-gray-500">{e.throughput.toFixed(0)}/s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
