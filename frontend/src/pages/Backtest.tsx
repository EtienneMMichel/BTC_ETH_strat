import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ConfigForm } from '@/components/backtest/ConfigForm'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import { MetricsGrid } from '@/components/backtest/MetricsGrid'
import { EquityChart } from '@/components/charts/EquityChart'
import { WeightsChart } from '@/components/charts/WeightsChart'
import { submitBacktest, getBacktestJob } from '@/lib/api/backtest'
import { useBacktestStore } from '@/lib/store'
import type { BacktestRequest } from '@/types/backtest'

export function BacktestPage() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const setResult = useBacktestStore((s) => s.setResult)
  const lastRequest = useBacktestStore((s) => s.lastRequest)

  const { data: job } = useQuery({
    queryKey: ['backtest', jobId],
    queryFn: () => getBacktestJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'done' || status === 'failed' ? false : 2000
    },
  })

  // Persist result to store when job completes — kept in useEffect to avoid render-phase side effects
  useEffect(() => {
    if (job?.status === 'done' && job.result && lastRequest) {
      setResult(job.result, lastRequest)
    }
  }, [job, lastRequest, setResult])

  const handleSubmit = async (req: BacktestRequest) => {
    setIsSubmitting(true)
    // Store request immediately so it's available when job completes
    useBacktestStore.setState({ lastRequest: req })
    try {
      const { job_id } = await submitBacktest(req)
      setJobId(job_id)
    } finally {
      setIsSubmitting(false)
    }
  }

  const result = job?.result

  return (
    <div className="flex gap-6 h-full">
      {/* Left panel */}
      <div className="w-80 shrink-0">
        <h1 className="text-lg font-semibold mb-4 text-slate-100">Configure Backtest</h1>
        <ConfigForm onSubmit={handleSubmit} isLoading={isSubmitting || job?.status === 'running' || job?.status === 'pending'} />
        {job && (
          <div className="mt-4 space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Status:</span>
              <StatusBadge status={job.status} />
            </div>
            <JobProgressBar status={job.status} />
          </div>
        )}
        {job?.error && <p className="mt-2 text-sm text-rose-400">{job.error}</p>}
      </div>

      {/* Right panel */}
      <div className="flex-1 flex flex-col gap-4 min-w-0">
        {result ? (
          <>
            <div>
              <h2 className="text-sm font-medium text-slate-400 mb-2">Equity Curve</h2>
              <EquityChart equityCurve={result.equity_curve} label="Strategy" benchmarks={result.benchmarks} />
            </div>
            <div>
              <h2 className="text-sm font-medium text-slate-400 mb-2">Portfolio Weights</h2>
              <WeightsChart signals={result.signals} />
            </div>
            <div>
              <h2 className="text-sm font-medium text-slate-400 mb-2">Performance Metrics</h2>
              <MetricsGrid metrics={result.metrics} />
            </div>
            {result.trade_count != null && (
              <p className="text-xs text-slate-500">Total trades: {result.trade_count}</p>
            )}
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
            Configure and submit a backtest to see results.
          </div>
        )}
      </div>
    </div>
  )
}
