export interface ServiceHealth {
  name: string
  status: 'healthy' | 'degraded' | 'unhealthy'
  requestRate: number
  errorRate: number
  p99Latency: number
  cpuUsage: number
  memoryUsage: number
  podCount: number
}

export interface AnomalyEvent {
  service: string
  timestamp: string
  combinedScore: number
  isolationForestScore: number
  lstmScore: number
  isAnomaly: boolean
  anomalyType: string | null
}

export interface ChaosExperiment {
  id: string
  name: string
  faultType: string
  status: 'running' | 'completed' | 'failed' | 'rolled_back'
  startedAt: string
  endedAt: string | null
  targetService: string
}

export interface RecoveryAction {
  id: string
  actionType: string
  targetService: string
  status: 'pending' | 'executing' | 'validating' | 'success' | 'failed'
  startedAt: string
  completedAt: string | null
  anomalyType: string
}

export interface TimelineEvent {
  id: string
  type: 'injection' | 'detection' | 'recovery'
  service: string
  description: string
  timestamp: string
  status: string
}

export interface SystemStatus {
  healthyServices: string[]
  unhealthyServices: string[]
  activeRecoveries: number
  recentActions: RecoveryAction[]
}
