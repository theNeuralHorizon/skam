import { useEffect, useRef, useState } from 'react'
import type { TimelineEvent } from '../types'

const WS_URLS = [
  `ws://${window.location.hostname}:8092/ws/events`,
  `ws://${window.location.hostname}:30092/ws/events`,
]
const RECONNECT_DELAY = 3000
const MAX_EVENTS = 200
const MOCK_INTERVAL = 4000

const MOCK_SERVICES = ['api-gateway', 'user-service', 'product-service', 'order-service', 'payment-service', 'notification-service']
const MOCK_TEMPLATES: Array<{ type: TimelineEvent['type']; desc: (svc: string) => string }> = [
  { type: 'detection', desc: () => `anomaly score 0.${72 + Math.floor(Math.random() * 20)} detected` },
  { type: 'injection', desc: () => `pod_kill experiment started` },
  { type: 'recovery', desc: () => `restart_pod action completed successfully` },
  { type: 'detection', desc: () => `latency spike: p99=${(200 + Math.random() * 800).toFixed(0)}ms` },
  { type: 'recovery', desc: () => `scale_up HPA to ${3 + Math.floor(Math.random() * 3)} replicas` },
  { type: 'injection', desc: () => `network_partition applied` },
  { type: 'recovery', desc: () => `network_policy removed, traffic restored` },
  { type: 'detection', desc: () => `error rate ${(5 + Math.random() * 20).toFixed(1)}% above threshold` },
]

function generateMockEvent(): TimelineEvent {
  const svc = MOCK_SERVICES[Math.floor(Math.random() * MOCK_SERVICES.length)]
  const tpl = MOCK_TEMPLATES[Math.floor(Math.random() * MOCK_TEMPLATES.length)]
  return {
    id: `mock-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    type: tpl.type,
    service: svc,
    description: tpl.desc(svc),
    timestamp: new Date().toISOString(),
    status: tpl.type === 'recovery' ? 'success' : 'active',
  }
}

export function useWebSocket() {
  const [events, setEvents] = useState<TimelineEvent[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>()
  const mockTimer = useRef<ReturnType<typeof setInterval>>()
  const urlIndex = useRef(0)
  const failCount = useRef(0)

  useEffect(() => {
    function stopMock() {
      if (mockTimer.current) {
        clearInterval(mockTimer.current)
        mockTimer.current = undefined
      }
    }

    function startMock() {
      if (mockTimer.current) return
      const seed = Array.from({ length: 5 }, () => generateMockEvent())
      setEvents((prev) => [...prev, ...seed].slice(-MAX_EVENTS))
      mockTimer.current = setInterval(() => {
        setEvents((prev) => [...prev, generateMockEvent()].slice(-MAX_EVENTS))
      }, MOCK_INTERVAL)
    }

    function connect() {
      const url = WS_URLS[urlIndex.current % WS_URLS.length]
      try {
        const ws = new WebSocket(url)

        ws.onopen = () => {
          setIsConnected(true)
          failCount.current = 0
          stopMock()
        }

        ws.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data) as TimelineEvent
            setEvents((prev) => [...prev, data].slice(-MAX_EVENTS))
          } catch {
            // ignore malformed
          }
        }

        ws.onclose = () => {
          setIsConnected(false)
          failCount.current++
          urlIndex.current++
          if (failCount.current >= WS_URLS.length * 2) {
            startMock()
          }
          reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY)
        }

        ws.onerror = () => {
          ws.close()
        }

        wsRef.current = ws
      } catch {
        failCount.current++
        urlIndex.current++
        if (failCount.current >= WS_URLS.length * 2) {
          startMock()
        }
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY)
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimer.current)
      stopMock()
      wsRef.current?.close()
    }
  }, [])

  return { events, isConnected }
}
