export type StrategyName = 'momentum' | 'mean_reversion' | 'orchestrator'

export interface BacktestRequest {
  strategy: StrategyName
  min_train_bars: number
  rebalance_every: number
  fee_rate: number
  slippage: number
}

export interface BacktestMetrics {
  sharpe_ratio: number
  max_drawdown: number
  calmar_ratio: number
  historical_var_5pct: number
  expected_shortfall_5pct: number
  win_rate?: number
}

export interface BacktestResult {
  equity_curve: [string, number][]
  signals: [string, Record<string, number>][]
  metrics: BacktestMetrics
  trade_count: number
}

export type JobStatus = 'pending' | 'running' | 'done' | 'failed'

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  result?: BacktestResult
  error?: string
}
