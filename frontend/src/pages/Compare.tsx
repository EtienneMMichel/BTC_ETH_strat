import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { CompareConfigForm, COMPARE_PALETTE } from '@/components/compare/CompareConfigForm'
import { CompareMetricsTable } from '@/components/compare/CompareMetricsTable'
import { OverlayEquityChart } from '@/components/charts/OverlayEquityChart'
import { EfficientFrontierChart } from '@/components/charts/EfficientFrontierChart'
import { EyeButton } from '@/components/ui/EyeButton'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import { submitCompare, getCompareJob } from '@/lib/api/compare'
import type { CompareRequest } from '@/types/compare'

export function ComparePage() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [hidden, setHidden] = useState<Set<string>>(new Set())

  const { data: job } = useQuery({
    queryKey: ['compare', jobId],
    queryFn: () => getCompareJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'done' || status === 'failed' ? false : 2000
    },
  })

  const handleSubmit = async (req: CompareRequest) => {
    setIsSubmitting(true)
    setHidden(new Set())
    try {
      const { job_id } = await submitCompare(req)
      setJobId(job_id)
    } finally {
      setIsSubmitting(false)
    }
  }

  const result = job?.result
  const isRunning = job?.status === 'running' || job?.status === 'pending'

  const toggleVisibility = (label: string) => {
    setHidden((prev) => {
      const next = new Set(prev)
      if (next.has(label)) next.delete(label)
      else next.add(label)
      return next
    })
  }

  const curves = result?.strategies.map((s, i) => ({
    label: s.label,
    color: COMPARE_PALETTE[i % COMPARE_PALETTE.length],
    data: s.equity_curve,
    visible: !hidden.has(s.label),
  })) ?? []

  return (
    <div className="flex gap-6 h-full">
      {/* Left panel */}
      <div className="w-80 shrink-0 overflow-y-auto">
        <h1 className="text-lg font-semibold mb-4 text-slate-100">Strategy Comparison</h1>
        <CompareConfigForm onSubmit={handleSubmit} isLoading={isSubmitting || isRunning} />
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
      <div className="flex-1 flex flex-col gap-4 min-w-0 overflow-y-auto">
        {result ? (
          <>
            {/* Equity curves */}
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h2 className="text-sm font-medium text-slate-400">Equity Curves</h2>
                <div className="flex gap-2 flex-wrap">
                  {result.strategies.map((s, i) => (
                    <div key={s.label} className="flex items-center gap-1">
                      <EyeButton
                        visible={!hidden.has(s.label)}
                        onToggle={() => toggleVisibility(s.label)}
                      />
                      <span
                        className="inline-block w-2 h-2 rounded-full"
                        style={{ background: COMPARE_PALETTE[i % COMPARE_PALETTE.length] }}
                      />
                      <span className="text-xs text-slate-400">{s.label}</span>
                    </div>
                  ))}
                </div>
              </div>
              <OverlayEquityChart curves={curves} benchmarks={result.benchmarks} />
            </div>

            {/* Efficient Frontier */}
            {result.frontier && result.frontier.length > 0 && (
              <div>
                <h2 className="text-sm font-medium text-slate-400 mb-2">Efficient Frontier</h2>
                <EfficientFrontierChart
                  frontier={result.frontier}
                  special={result.frontier_special}
                  strategies={result.strategies}
                  colors={result.strategies.map((_, i) => COMPARE_PALETTE[i % COMPARE_PALETTE.length])}
                />
              </div>
            )}

            {/* Metrics table */}
            <div>
              <h2 className="text-sm font-medium text-slate-400 mb-2">Performance Metrics</h2>
              <CompareMetricsTable
                strategies={result.strategies}
                colors={result.strategies.map((_, i) => COMPARE_PALETTE[i % COMPARE_PALETTE.length])}
              />
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
            Configure strategies and run a comparison to see results.
          </div>
        )}
      </div>
    </div>
  )
}
