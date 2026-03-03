import type { JobStatus } from './common'

export type { JobStatus }
export type ModelType = 'threshold' | 'vol_quantile' | 'hmm'

export interface ModelConfig {
  model_id: string
  model_type: ModelType
  params: Record<string, unknown>
}

export interface RegimeDetectionRequest {
  assets: string[]
  freq: string
  start_date?: string
  end_date?: string
  models: ModelConfig[]
  prob_window: number
}

export interface RegimeModelResult {
  model_id: string
  model_type: string
  unique_regimes: string[]
  labels: [string, string][]
  probabilities: Record<string, [string, number][]>
}

export interface RegimeDetectionResult {
  assets: string[]
  freq: string
  date_range: [string, string]
  prices: Record<string, [string, number][]>
  volumes: Record<string, [string, number][]>
  models: RegimeModelResult[]
}

export interface RegimeDetectionJobResponse {
  job_id: string
  status: JobStatus
  result?: RegimeDetectionResult
  error?: string
}
