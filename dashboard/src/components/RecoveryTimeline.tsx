import type { RecoveryAction } from '../types'
import { Shield } from 'lucide-react'

interface Props {
  actions: RecoveryAction[]
}

const ACTION_COLORS: Record<string, string> = {
  restart_pod: 'bg-blue-500',
  scale_up: 'bg-purple-500',
  rolling_restart: 'bg-cyan-500',
  remove_network_policy: 'bg-orange-500',
  increase_resources: 'bg-yellow-500',
  restart_redis: 'bg-red-500',
}

const STATUS_STYLES: Record<string, string> = {
  pending: 'text-gray-400',
  executing: 'text-yellow-400 animate-pulse',
  validating: 'text-blue-400 animate-pulse',
  success: 'text-green-400',
  failed: 'text-red-400',
}

function timeSince(dateStr: string): string {
  const seconds = Math.floor((Date.now() - new Date(dateStr).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  return `${Math.floor(seconds / 3600)}h ago`
}

function durationStr(start: string, end: string | null): string {
  if (!end) return 'in progress'
  const ms = new Date(end).getTime() - new Date(start).getTime()
  return `${(ms / 1000).toFixed(1)}s`
}

export function RecoveryTimeline({ actions }: Props) {
  const sorted = [...actions].sort(
    (a, b) => new Date(b.startedAt).getTime() - new Date(a.startedAt).getTime(),
  ).slice(0, 10)

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        <Shield className="w-4 h-4 text-green-500" />
        Recovery Timeline
      </h2>

      {sorted.length === 0 ? (
        <p className="text-xs text-gray-500 text-center py-4">No recovery actions yet</p>
      ) : (
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {sorted.map((action) => (
            <div key={action.id} className="flex items-start gap-3 p-2 rounded bg-gray-800">
              <div className={`w-1 h-10 rounded ${ACTION_COLORS[action.actionType] || 'bg-gray-600'}`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-white">
                    {action.actionType.replace(/_/g, ' ')}
                  </span>
                  <span className={`text-xs ${STATUS_STYLES[action.status] || ''}`}>
                    {action.status}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-gray-500 mt-0.5">
                  <span>{action.targetService}</span>
                  <span>|</span>
                  <span>{action.anomalyType}</span>
                  <span>|</span>
                  <span>{durationStr(action.startedAt, action.completedAt)}</span>
                  <span>|</span>
                  <span>{timeSince(action.startedAt)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
