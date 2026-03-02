import type { JobStatus } from './common'

export type TimeSeries = [string, number][]

// ── Vol Eval ──────────────────────────────────────────────────────────────

export interface VolEvalRequest {
  asset: 'BTC' | 'ETH'
  freq: string
  min_train_bars: number
  test_size: number
  models: string[]
}

export interface VolModelMetrics {
  qlike: number
  mse: number
  mae: number
}

export interface VolEvalResult {
  comparison_table: Record<string, VolModelMetrics>
  forecasts: Record<string, TimeSeries>
  realised: TimeSeries
  asset: string
  freq: string
}

// ── Price Forecast ────────────────────────────────────────────────────────

export interface PriceForecastRequest {
  asset: 'BTC' | 'ETH'
  forecast_horizon: string
  min_train_bars: number
  models: string[]
  model_resolutions: Record<string, string>
  n_calibration_bins?: number
}

export interface ConfusionMatrix {
  tp: number
  fp: number
  tn: number
  fn: number
}

export interface DirectionMetrics {
  hit_rate: number
  precision: number
  recall: number
  f1: number
  confusion: ConfusionMatrix
  roc_curve: [number, number][]
  auc: number
}

export interface CalibrationBin {
  bin: number
  mean_abs_signal: number
  mean_abs_return: number
  count: number
}

export interface ReliabilityBin {
  prob_bin: number
  actual_rate: number
  count: number
}

export interface AmplitudeMetrics {
  ic: number
  rank_ic: number
  mae: number
  mse: number
  calibration_bins: CalibrationBin[]
  reliability_diagram?: ReliabilityBin[]
  brier_score?: number
  log_loss?: number
}

export interface HistogramBin {
  bin_center: number
  count: number
}

export interface AcfPoint {
  lag: number
  acf: number
}

export interface AdditionalMetrics {
  signal_histogram: HistogramBin[]
  turnover: number
  acf: AcfPoint[]
}

export interface PriceForecastModelMetrics {
  direction: DirectionMetrics
  amplitude: AmplitudeMetrics
  additional: AdditionalMetrics
  output_type: 'direction' | 'amplitude' | 'probability'
}

export interface PriceForecastResult {
  signals: Record<string, TimeSeries>
  metrics: Record<string, PriceForecastModelMetrics>
  prices: TimeSeries
  asset: string
  forecast_horizon: string
  model_resolutions: Record<string, string>
  warnings?: string[]
}

// ── Co-Movement ───────────────────────────────────────────────────────────

export interface CoMovRequest {
  freq: string
  rolling_window: number
  copula_type: 'gaussian' | 'student_t' | 'clayton'
}

export interface CoMovCurrentStats {
  dcc_correlation: number
  dcc_lower_tail_dep: number
  copula_lower_tail_dep: number
}

export interface CoMovResult {
  dcc_correlation: TimeSeries
  dcc_lower_tail_dep: TimeSeries
  copula_lower_tail_dep: TimeSeries
  copula_upper_tail_dep: TimeSeries
  current_stats: CoMovCurrentStats
  freq: string
  rolling_window: number
  copula_type: string
}

// ── Generic job response ──────────────────────────────────────────────────

export interface ModelJobResponse<T> {
  job_id: string
  status: JobStatus
  result?: T
  error?: string
}
