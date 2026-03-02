import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { submitCoMov, getCoMovJob } from '@/lib/api/models'
import { MultiLineChart } from '@/components/charts/MultiLineChart'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import type { CoMovRequest, CoMovResult } from '@/types/models'

type CopulaType = 'gaussian' | 'student_t' | 'clayton'

export function CoMovPage() {
  const [freq, setFreq] = useState('1D')
  const [rollingWindow, setRollingWindow] = useState(126)
  const [copulaType, setCopulaType] = useState<CopulaType>('clayton')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)

  const { data: job } = useQuery({
    queryKey: ['co-mov', jobId],
    queryFn: () => getCoMovJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000,
  })

  const handleSubmit = async () => {
    setIsSubmitting(true)
    try {
      const req: CoMovRequest = { freq, rolling_window: rollingWindow, copula_type: copulaType }
      const { job_id } = await submitCoMov(req)
      setJobId(job_id)
    } finally {
      setIsSubmitting(false)
    }
  }

  const result: CoMovResult | undefined = job?.result

  const dccSeries = useMemo(() => {
    if (!result) return []
    return [
      { data: result.dcc_correlation, color: '#38bdf8', title: 'DCC Correlation' },
      { data: result.dcc_lower_tail_dep, color: '#a78bfa', title: 'DCC Lower Tail Dep' },
    ]
  }, [result])

  const copulaSeries = useMemo(() => {
    if (!result) return []
    return [
      { data: result.copula_lower_tail_dep, color: '#34d399', title: 'Copula Lower Tail Dep' },
      { data: result.copula_upper_tail_dep, color: '#fbbf24', title: 'Copula Upper Tail Dep' },
    ]
  }, [result])

  const fmt = (v: number) =>
    isNaN(v) ? '—' : v.toFixed(4)

  return (
    <div className="flex flex-col gap-6 max-w-5xl">
      {/* Top form bar */}
      <div className="flex flex-wrap items-end gap-6">
        <h1 className="text-lg font-semibold text-slate-100 w-full">Co-Movement</h1>

        <div>
          <Label className="mb-2 block">Frequency</Label>
          <div className="flex gap-3">
            {['1D', '4h', '1h'].map((f) => (
              <label key={f} className="flex items-center gap-1.5 cursor-pointer">
                <input type="radio" name="freq" checked={freq === f} onChange={() => setFreq(f)} className="accent-sky-500" />
                <span className="text-sm text-slate-300">{f}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="w-48">
          <Label className="mb-1 block">Rolling window: {rollingWindow}</Label>
          <input
            type="range" min={30} max={252} step={1}
            value={rollingWindow}
            onChange={(e) => setRollingWindow(Number(e.target.value))}
            className="w-full accent-sky-500"
          />
        </div>

        <div>
          <Label className="mb-2 block">Copula type</Label>
          <div className="flex gap-3">
            {(['gaussian', 'student_t', 'clayton'] as CopulaType[]).map((c) => (
              <label key={c} className="flex items-center gap-1.5 cursor-pointer">
                <input type="radio" name="copula" checked={copulaType === c} onChange={() => setCopulaType(c)} className="accent-sky-500" />
                <span className="text-sm text-slate-300">{c}</span>
              </label>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Button
            onClick={handleSubmit}
            disabled={isSubmitting || job?.status === 'running' || job?.status === 'pending'}
          >
            Run
          </Button>
          {job && <StatusBadge status={job.status} />}
        </div>
      </div>

      {job && <JobProgressBar status={job.status} />}
      {job?.error && <p className="text-sm text-rose-400">{job.error}</p>}

      {result && (
        <>
          {/* Current stats row */}
          <div className="grid grid-cols-3 gap-4">
            <Card>
              <CardHeader><CardTitle className="text-sm">DCC Correlation</CardTitle></CardHeader>
              <CardContent>
                <div className="text-2xl font-semibold text-sky-400">
                  {fmt(result.current_stats.dcc_correlation)}
                </div>
                <div className="text-xs text-slate-500 mt-1">Latest estimate</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle className="text-sm">DCC Lower Tail Dep</CardTitle></CardHeader>
              <CardContent>
                <div className="text-2xl font-semibold text-violet-400">
                  {fmt(result.current_stats.dcc_lower_tail_dep)}
                </div>
                <div className="text-xs text-slate-500 mt-1">λ_L (DCC proxy)</div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle className="text-sm">Copula Lower Tail Dep</CardTitle></CardHeader>
              <CardContent>
                <div className="text-2xl font-semibold text-emerald-400">
                  {fmt(result.current_stats.copula_lower_tail_dep)}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  λ_L ({result.copula_type}, w={result.rolling_window})
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Chart 1: DCC */}
          <div>
            <h2 className="text-sm font-medium text-slate-400 mb-2">
              DCC Correlation &amp; Lower Tail Dependence
            </h2>
            <MultiLineChart series={dccSeries} height={260} />
          </div>

          {/* Chart 2: Copula */}
          <div>
            <h2 className="text-sm font-medium text-slate-400 mb-2">
              Copula Tail Dependence ({result.copula_type}, window={result.rolling_window})
            </h2>
            <MultiLineChart series={copulaSeries} height={260} />
          </div>
        </>
      )}

      {!result && !job && (
        <div className="flex items-center justify-center h-48 text-slate-600 text-sm">
          Configure and submit to view BTC-ETH co-movement analysis.
        </div>
      )}
    </div>
  )
}
