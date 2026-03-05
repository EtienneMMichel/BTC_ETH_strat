import type { StrategyName, MarkowitzParams } from './backtest'

export interface StrategyConfig {
  label: string
  strategy: StrategyName
  markowitz?: MarkowitzParams
}

export interface CompareRequest {
  freq?: string
  min_train_bars: number
  rebalance_every: number
  fee_rate: number
  slippage: number
  strategies: StrategyConfig[]
  include_frontier: boolean
  frontier_points: number
  frontier_cov_model: 'rolling' | 'diagonal' | 'bekk'
  frontier_er_model: 'rolling_mean' | 'signal_tsmom'
  frontier_long_only: boolean
}

export interface StrategyResultEntry {
  label: string
  strategy: string
  equity_curve: [string, number][]
  metrics: Record<string, number>
  trade_count: number
  signals: [string, Record<string, number>][]
  ann_return: number
  ann_vol: number
}

export interface FrontierPoint {
  gamma: number
  ann_return: number
  ann_vol: number
  sharpe: number
  weights: Record<string, number>
}

export interface CompareResult {
  strategies: StrategyResultEntry[]
  benchmarks: Record<string, [string, number][]>
  frontier: FrontierPoint[] | null
  frontier_special: Record<string, FrontierPoint> | null
}

export type CompareJobStatus = 'pending' | 'running' | 'done' | 'failed'

export interface CompareJobResponse {
  job_id: string
  status: CompareJobStatus
  result?: CompareResult
  error?: string
}
