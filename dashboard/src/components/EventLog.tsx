import { useEffect, useRef } from 'react'
import { Terminal } from 'lucide-react'
import type { TimelineEvent } from '../types'

interface Props {
  events: TimelineEvent[]
}

const TYPE_BADGE: Record<string, string> = {
  injection: 'bg-red-500/20 text-red-400',
  detection: 'bg-yellow-500/20 text-yellow-400',
  recovery: 'bg-green-500/20 text-green-400',
}

function formatTime(dateStr: string): string {
  return new Date(dateStr).toLocaleTimeString('en-US', { hour12: false })
}

export function EventLog({ events }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [events.length])

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h2 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        <Terminal className="w-4 h-4 text-gray-400" />
        Event Log
      </h2>

      <div ref={containerRef} className="max-h-[250px] overflow-y-auto space-y-1 font-mono">
        {events.length === 0 ? (
          <p className="text-xs text-gray-600 text-center py-4">
            Waiting for events... Connect to the platform to see live data.
          </p>
        ) : (
          events.slice(-100).map((event) => (
            <div key={event.id} className="flex items-start gap-2 text-[11px] py-0.5">
              <span className="text-gray-600 shrink-0">{formatTime(event.timestamp)}</span>
              <span className={`px-1.5 py-0 rounded text-[10px] shrink-0 ${TYPE_BADGE[event.type] || 'bg-gray-700 text-gray-400'}`}>
                {event.type}
              </span>
              <span className="text-gray-400 truncate">
                <span className="text-gray-300">{event.service}</span> {event.description}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}
