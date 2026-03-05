import type { TimeSeries } from './models'
import type { JobStatus } from './common'

export interface CovModelSeries {
  var_BTC: TimeSeries
  var_ETH: TimeSeries
  cov_BTC_ETH: TimeSeries
}

export interface CovEvalResult {
  models: Record<string, CovModelSeries>
  realised: CovModelSeries
  comparison: Record<string, Record<string, number>>  // model → {rmse_var_BTC, rmse_var_ETH, corr_cov}
}

export interface ERModelMetrics {
  ic: number
  hit_rate: number
}

export interface ERAssetResult {
  mu: Record<string, TimeSeries>
  metrics: Record<string, ERModelMetrics>
}

export interface ERCovEvalResult {
  per_asset: Record<string, ERAssetResult>
}

export interface MarkowitzJobResponse<T> {
  job_id: string
  status: JobStatus
  result?: T
  error?: string
}
