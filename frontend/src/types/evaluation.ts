export interface VarTestRequest {
  returns: number[]
  var_series: number[]
  alpha: number
}

export interface VarTestResult {
  kupiec: { test_stat: number; p_value: number; violations: number; expected: number }
  christoffersen: { independence_stat: number; p_value: number }
}

export interface DMTestRequest {
  errors1: number[]
  errors2: number[]
}

export interface DMTestResult {
  dm_stat: number
  p_value: number
}

export type EvalJobStatus = 'pending' | 'running' | 'done' | 'failed'

export interface EvalJobResponse<T> {
  job_id: string
  status: EvalJobStatus
  result?: T
  error?: string
}
