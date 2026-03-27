import { useState } from 'react'
import { Zap, Play } from 'lucide-react'
import type { ChaosExperiment } from '../types'
import { triggerExperiment } from '../utils/api'

interface Props {
  experiments: ChaosExperiment[]
  onRefresh: () => void
}

const FAULT_TYPES = [
  'pod_kill',
  'pod_crash_loop',
  'cpu_stress',
  'memory_pressure',
  'network_partition',
  'latency_injection',
]

const SERVICES = [
  'api-gateway',
  'user-service',
  'product-service',
  'order-service',
  'payment-service',
  'notification-service',
]

const STATUS_BADGE: Record<string, string> = {
  running: 'bg-yellow-500/20 text-yellow-400',
  completed: 'bg-green-500/20 text-green-400',
  failed: 'bg-red-500/20 text-red-400',
  rolled_back: 'bg-blue-500/20 text-blue-400',
}

export function ChaosPanel({ experiments, onRefresh }: Props) {
  const [showForm, setShowForm] = useState(false)
  const [target, setTarget] = useState(SERVICES[3])
  const [faultType, setFaultType] = useState(FAULT_TYPES[0])
  const [duration, setDuration] = useState(60)
  const [submitting, setSubmitting] = useState(false)

  const handleTrigger = async () => {
    setSubmitting(true)
    try {
      await triggerExperiment({
        name: `${faultType}-${target}-${Date.now()}`,
        faultType,
        target,
        duration,
      })
      setShowForm(false)
      onRefresh()
    } catch {
      // API might be unreachable
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Zap className="w-4 h-4 text-orange-500" />
          Chaos Experiments
        </h2>
        <button
          onClick={() => setShowForm(!showForm)}
          className="text-xs px-2 py-1 rounded bg-orange-600 hover:bg-orange-500 text-white transition-colors"
        >
          {showForm ? 'Cancel' : 'New Experiment'}
        </button>
      </div>

      {showForm && (
        <div className="mb-4 p-3 rounded bg-gray-800 space-y-2">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Target Service</label>
            <select
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              className="w-full text-xs bg-gray-700 text-white rounded px-2 py-1.5 border border-gray-600"
            >
              {SERVICES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Fault Type</label>
            <select
              value={faultType}
              onChange={(e) => setFaultType(e.target.value)}
              className="w-full text-xs bg-gray-700 text-white rounded px-2 py-1.5 border border-gray-600"
            >
              {FAULT_TYPES.map((f) => (
                <option key={f} value={f}>{f.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Duration: {duration}s</label>
            <input
              type="range"
              min={10}
              max={300}
              value={duration}
              onChange={(e) => setDuration(Number(e.target.value))}
              className="w-full"
            />
          </div>
          <button
            onClick={handleTrigger}
            disabled={submitting}
            className="flex items-center gap-1 text-xs px-3 py-1.5 rounded bg-red-600 hover:bg-red-500 text-white disabled:opacity-50 transition-colors"
          >
            <Play className="w-3 h-3" />
            {submitting ? 'Injecting...' : 'Inject Fault'}
          </button>
        </div>
      )}

      <div className="space-y-2 max-h-[300px] overflow-y-auto">
        {experiments.length === 0 ? (
          <p className="text-xs text-gray-500 text-center py-4">No experiments yet</p>
        ) : (
          experiments.map((exp) => (
            <div key={exp.id} className="flex items-center justify-between p-2 rounded bg-gray-800">
              <div>
                <span className="text-xs font-medium text-white">{exp.faultType.replace(/_/g, ' ')}</span>
                <span className="text-xs text-gray-500 ml-2">{exp.targetService}</span>
              </div>
              <span className={`text-xs px-2 py-0.5 rounded ${STATUS_BADGE[exp.status] || 'bg-gray-700 text-gray-400'}`}>
                {exp.status}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
