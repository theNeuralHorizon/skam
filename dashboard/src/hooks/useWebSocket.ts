import { useCallback, useEffect, useRef, useState } from 'react'
import type { TimelineEvent } from '../types'

// In dev mode, Vite proxies /api/healing to the decision engine.
// WebSocket connects directly — try local dev port first, then K8s NodePort.
const WS_URLS = [
  `ws://${window.location.hostname}:8092/ws/events`,
  `ws://${window.location.hostname}:30092/ws/events`,
]
const RECONNECT_DELAY = 3000
const MAX_EVENTS = 200
const MOCK_INTERVAL = 4000

const MOCK_SERVICES = ['api-gateway', 'user-service', 'product-service', 'order-service', 'payment-service', 'notification-service']
const MOCK_EVENTS: Array<{ type: TimelineEvent['type']; desc: (svc: string) => string }> = [
  { type: 'detection', desc: (s) => `anomaly score 0.${72 + Math.floor(Math.random() * 20)} detected` },
  { type: 'injection', desc: (s) => `pod_kill experiment started` },
  { type: 'recovery', desc: (s) => `restart_pod action completed successfully` },
  { type: 'detection', desc: (s) => `latency spike: p99=${(200 + Math.random() * 800).toFixed(0)}ms` },
  { type: 'recovery', desc: (s) => `scale_up HPA to ${3 + Math.floor(Math.random() * 3)} replicas` },
  { type: 'injection', desc: (s) => `network_partition applied` },
  { type: 'recovery', desc: (s) => `network_policy removed, traffic restored` },
  { type: 'detection', desc: (s) => `error rate ${(5 + Math.random() * 20).toFixed(1)}% above threshold` },
]

function generateMockEvent(): TimelineEvent {
  const svc = MOCK_SERVICES[Math.floor(Math.random() * MOCK_SERVICES.length)]
  const mock = MOCK_EVENTS[Math.floor(Math.random() * MOCK_EVENTS.length)]
  return {
    id: `mock-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    type: mock.type,
    service: svc,
    description: mock.desc(svc),
    timestamp: new Date().toISOString(),
    status: mock.type === 'recovery' ? 'success' : 'active',
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

  const stopMockEvents = useCallback(() => {
    if (mockTimer.current) {
      clearInterval(mockTimer.current)
      mockTimer.current = undefined
    }
  }, [])

  const startMockEvents = useCallback(() => {
    if (mockTimer.current) return
    // Seed a few initial events
    const seed = Array.from({ length: 5 }, () => generateMockEvent())
    setEvents((prev) => [...prev, ...seed].slice(-MAX_EVENTS))

    mockTimer.current = setInterval(() => {
      setEvents((prev) => [...prev, generateMockEvent()].slice(-MAX_EVENTS))
    }, MOCK_INTERVAL)
  }, [])

  const connect = useCallback(() => {
    const url = WS_URLS[urlIndex.current % WS_URLS.length]
    try {
      const ws = new WebSocket(url)

      ws.onopen = () => {
        setIsConnected(true)
        failCount.current = 0
        stopMockEvents()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TimelineEvent
          setEvents((prev) => [...prev, data].slice(-MAX_EVENTS))
        } catch {
          // ignore malformed messages
        }
      }

      ws.onclose = () => {
        setIsConnected(false)
        failCount.current++
        // Try next URL on failure
        urlIndex.current++
        // After trying all URLs twice, fall back to mock events
        if (failCount.current >= WS_URLS.length * 2) {
          startMockEvents()
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
        startMockEvents()
      }
      reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY)
    }
  }, [startMockEvents, stopMockEvents])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(reconnectTimer.current)
      stopMockEvents()
      wsRef.current?.close()
    }
  }, [connect, stopMockEvents])

  return { events, isConnected }
}
