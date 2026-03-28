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

export type SeverityLevel = 'normal' | 'low' | 'medium' | 'high' | 'critical'

export interface FeatureContribution {
  feature: string
  z_score: number
  contribution_pct: number
  direction: 'high' | 'low'
}

export interface AnomalyEvent {
  service: string
  timestamp: string
  combinedScore: number
  isolationForestScore: number
  lstmScore: number
  isAnomaly: boolean
  anomalyType: string | null
  severity: SeverityLevel
  severityLevel: number  // 0-4 numeric
  consecutiveWindows: number
  scoreVelocity: number
  maxResponseTimeS: number
  topContributors: FeatureContribution[]
}

export interface EnsembleComparison {
  name: string
  aucRoc: number
  aucPr: number
  f1Best: number
  mcc: number
  cohensKappa: number
  brierScore: number
  scoreSeparation: number
  coldStartSamples: number
  throughput: number
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
  severity: SeverityLevel
  healingTimeMs: number | null  // time from detection to recovery confirmation
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
