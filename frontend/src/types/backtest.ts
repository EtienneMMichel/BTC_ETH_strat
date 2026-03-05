export type StrategyName = 'momentum' | 'mean_reversion' | 'orchestrator' | 'markowitz'

export interface MarkowitzParams {
  objective: 'max_sharpe' | 'min_variance' | 'mean_variance' | 'max_diversification'
  cov_model: 'rolling' | 'diagonal' | 'bekk'
  er_model: 'rolling_mean' | 'signal_tsmom'
  gamma: number
  long_only: boolean
  max_weight: number
  min_weight: number
  risk_free_rate: number
  target_vol: number | null
  min_history: number
}

export interface BacktestRequest {
  strategy: StrategyName
  freq?: string
  min_train_bars: number
  rebalance_every: number
  fee_rate: number
  slippage: number
  markowitz?: MarkowitzParams
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
  benchmarks: Record<string, [string, number][]>
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
