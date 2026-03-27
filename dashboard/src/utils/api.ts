import type {
  AnomalyEvent,
  ChaosExperiment,
  RecoveryAction,
  ServiceHealth,
  SystemStatus,
} from '../types'

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

// ── Service Health ─────────────────────────────────────────

const SERVICES = [
  'api-gateway',
  'user-service',
  'product-service',
  'order-service',
  'payment-service',
  'notification-service',
]

export async function fetchServiceHealth(): Promise<ServiceHealth[]> {
  try {
    const status = await fetchJson<any>('/api/healing/status')
    return SERVICES.map((name) => ({
      name,
      status: status.unhealthy_services?.includes(name) ? 'unhealthy' : 'healthy',
      requestRate: Math.random() * 100,
      errorRate: status.unhealthy_services?.includes(name) ? Math.random() * 0.3 : Math.random() * 0.02,
      p99Latency: Math.random() * 500,
      cpuUsage: Math.random() * 80,
      memoryUsage: Math.random() * 70,
      podCount: 2,
    }))
  } catch {
    return mockServiceHealth()
  }
}

// ── Anomalies ──────────────────────────────────────────────

export async function fetchAnomalies(): Promise<AnomalyEvent[]> {
  try {
    return await fetchJson<AnomalyEvent[]>('/api/anomaly/anomalies/history')
  } catch {
    return mockAnomalies()
  }
}

// ── Experiments ────────────────────────────────────────────

export async function fetchExperiments(): Promise<ChaosExperiment[]> {
  try {
    return await fetchJson<ChaosExperiment[]>('/api/chaos/experiments')
  } catch {
    return mockExperiments()
  }
}

export async function triggerExperiment(config: {
  name: string
  faultType: string
  target: string
  duration: number
}): Promise<ChaosExperiment> {
  const res = await fetch('/api/chaos/experiments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: config.name,
      target: { namespace: 'default', label_selector: `app=${config.target}` },
      fault_type: config.faultType,
      parameters: {},
      duration_seconds: config.duration,
    }),
  })
  return res.json()
}

// ── Recovery Actions ───────────────────────────────────────

export async function fetchRecoveryActions(): Promise<RecoveryAction[]> {
  try {
    return await fetchJson<RecoveryAction[]>('/api/healing/actions')
  } catch {
    return mockRecoveryActions()
  }
}

// ── System Status ──────────────────────────────────────────

export async function fetchSystemStatus(): Promise<SystemStatus> {
  try {
    return await fetchJson<SystemStatus>('/api/healing/status')
  } catch {
    return {
      healthyServices: SERVICES.slice(0, 5),
      unhealthyServices: [SERVICES[5]],
      activeRecoveries: 0,
      recentActions: [],
    }
  }
}

// ── Mock Data (fallback when APIs are unreachable) ─────────

function mockServiceHealth(): ServiceHealth[] {
  return SERVICES.map((name, i) => ({
    name,
    status: i === 3 ? 'degraded' : 'healthy',
    requestRate: 20 + Math.random() * 80,
    errorRate: i === 3 ? 0.12 : Math.random() * 0.02,
    p99Latency: i === 3 ? 850 : 50 + Math.random() * 200,
    cpuUsage: 20 + Math.random() * 50,
    memoryUsage: 30 + Math.random() * 40,
    podCount: 2,
  }))
}

function mockAnomalies(): AnomalyEvent[] {
  const now = Date.now()
  return Array.from({ length: 30 }, (_, i) => ({
    service: SERVICES[i % SERVICES.length],
    timestamp: new Date(now - (30 - i) * 15000).toISOString(),
    combinedScore: i > 20 && i % 6 === 3 ? 0.6 + Math.random() * 0.35 : Math.random() * 0.4,
    isolationForestScore: Math.random() * 0.5,
    lstmScore: Math.random() * 0.5,
    isAnomaly: i > 20 && i % 6 === 3,
    anomalyType: i > 20 && i % 6 === 3 ? 'latency' : null,
  }))
}

function mockExperiments(): ChaosExperiment[] {
  return [
    {
      id: 'exp-001',
      name: 'pod-kill-order-svc',
      faultType: 'pod_kill',
      status: 'completed',
      startedAt: new Date(Date.now() - 300000).toISOString(),
      endedAt: new Date(Date.now() - 240000).toISOString(),
      targetService: 'order-service',
    },
    {
      id: 'exp-002',
      name: 'latency-product-svc',
      faultType: 'latency_injection',
      status: 'running',
      startedAt: new Date(Date.now() - 60000).toISOString(),
      endedAt: null,
      targetService: 'product-service',
    },
  ]
}

function mockRecoveryActions(): RecoveryAction[] {
  return [
    {
      id: 'rec-001',
      actionType: 'restart_pod',
      targetService: 'order-service',
      status: 'success',
      startedAt: new Date(Date.now() - 270000).toISOString(),
      completedAt: new Date(Date.now() - 240000).toISOString(),
      anomalyType: 'availability',
    },
    {
      id: 'rec-002',
      actionType: 'scale_up',
      targetService: 'product-service',
      status: 'executing',
      startedAt: new Date(Date.now() - 30000).toISOString(),
      completedAt: null,
      anomalyType: 'latency',
    },
  ]
}
