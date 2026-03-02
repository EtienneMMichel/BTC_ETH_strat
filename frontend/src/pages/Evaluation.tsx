import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useBacktestStore } from '@/lib/store'
import { equityToReturns } from '@/lib/utils'
import { submitVarTest, submitDMTest, getEvalJob } from '@/lib/api/evaluation'
import { submitBacktest, getBacktestJob } from '@/lib/api/backtest'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import { EquityChart } from '@/components/charts/EquityChart'
import type { VarTestResult, DMTestResult } from '@/lib/api/evaluation'
import type { StrategyName } from '@/types/backtest'

type Alpha = 0.01 | 0.05 | 0.10

export function EvaluationPage() {
  const { lastResult, lastRequest } = useBacktestStore()
  const [alpha, setAlpha] = useState<Alpha>(0.05)
  const [varJobId, setVarJobId] = useState<string | null>(null)
  const [dmJobId, setDmJobId] = useState<string | null>(null)
  const [compareStrategy, setCompareStrategy] = useState<StrategyName>('momentum')
  const [compareJobId, setCompareJobId] = useState<string | null>(null)
  const [isSubmittingVar, setIsSubmittingVar] = useState(false)
  const [isSubmittingCompare, setIsSubmittingCompare] = useState(false)

  const { data: varJob } = useQuery({
    queryKey: ['eval-var', varJobId],
    queryFn: () => getEvalJob<VarTestResult>(varJobId!),
    enabled: !!varJobId,
    refetchInterval: (q) => (q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000),
  })

  const { data: compareJob } = useQuery({
    queryKey: ['backtest-compare', compareJobId],
    queryFn: () => getBacktestJob(compareJobId!),
    enabled: !!compareJobId,
    refetchInterval: (q) => (q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000),
  })

  const { data: dmJob } = useQuery({
    queryKey: ['eval-dm', dmJobId],
    queryFn: () => getEvalJob<DMTestResult>(dmJobId!),
    enabled: !!dmJobId,
    refetchInterval: (q) => (q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000),
  })

  // Auto-trigger DM test once comparison backtest completes
  useEffect(() => {
    if (compareJob?.status === 'done' && compareJob.result && !dmJobId && lastResult) {
      const returns1 = equityToReturns(lastResult.equity_curve)
      const returns2 = equityToReturns(compareJob.result.equity_curve)
      submitDMTest({ errors1: returns1, errors2: returns2 })
        .then(({ job_id }) => setDmJobId(job_id))
        .catch(console.error)
    }
  }, [compareJob, dmJobId, lastResult])

  if (!lastResult || !lastRequest) {
    return (
      <div className="flex items-center justify-center h-full">
        <Card className="max-w-sm text-center p-8">
          <p className="text-slate-400">Run a backtest first to enable evaluation.</p>
        </Card>
      </div>
    )
  }

  const returns = equityToReturns(lastResult.equity_curve)
  const varValue = lastResult.metrics.historical_var_5pct

  const handleVarTest = async () => {
    setIsSubmittingVar(true)
    try {
      const { job_id } = await submitVarTest({
        returns,
        var_series: Array(returns.length).fill(varValue) as number[],
        alpha,
      })
      setVarJobId(job_id)
    } finally {
      setIsSubmittingVar(false)
    }
  }

  const handleDMTest = async () => {
    if (!lastRequest) return
    setIsSubmittingCompare(true)
    // Reset dm job so the effect can re-trigger
    setDmJobId(null)
    try {
      const { job_id } = await submitBacktest({ ...lastRequest, strategy: compareStrategy })
      setCompareJobId(job_id)
    } finally {
      setIsSubmittingCompare(false)
    }
  }

  const varResult = varJob?.result
  const dmResult = dmJob?.result
  const compareResult = compareJob?.result

  const strategyOptions: StrategyName[] = (['momentum', 'mean_reversion', 'orchestrator'] as StrategyName[])
    .filter((s) => s !== lastRequest.strategy)

  return (
    <div className="flex flex-col gap-8 max-w-4xl">
      <div>
        <h1 className="text-lg font-semibold text-slate-100 mb-1">Evaluation</h1>
        <p className="text-sm text-slate-400">
          Based on last backtest:{' '}
          <span className="text-slate-300 font-medium">{lastRequest.strategy}</span>
        </p>
      </div>

      {/* Panel A: VaR Backtest */}
      <section>
        <h2 className="text-base font-semibold text-slate-200 mb-3">Panel A — VaR Backtest</h2>
        <div className="flex items-center gap-4 mb-3">
          <Label>Alpha:</Label>
          {([0.01, 0.05, 0.10] as Alpha[]).map((a) => (
            <label key={a} className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="alpha"
                checked={alpha === a}
                onChange={() => setAlpha(a)}
                className="accent-sky-500"
              />
              <span className="text-sm text-slate-300">{(a * 100).toFixed(0)}%</span>
            </label>
          ))}
          <Button onClick={handleVarTest} disabled={isSubmittingVar || varJob?.status === 'running'} size="sm">
            Run VaR Test
          </Button>
          {varJob && <StatusBadge status={varJob.status} />}
        </div>
        {varJob && <JobProgressBar status={varJob.status} />}
        {varResult && (
          <div className="grid grid-cols-2 gap-3">
            <Card>
              <CardHeader><CardTitle>Kupiec Test</CardTitle></CardHeader>
              <CardContent>
                <div className="text-base">Stat: {varResult.kupiec.test_stat.toFixed(4)}</div>
                <div className={`text-sm mt-1 ${varResult.kupiec.p_value >= 0.05 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  p-value: {varResult.kupiec.p_value.toFixed(4)}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Violations: {varResult.kupiec.violations} / expected: {varResult.kupiec.expected.toFixed(1)}
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>Christoffersen Test</CardTitle></CardHeader>
              <CardContent>
                <div className="text-base">Stat: {varResult.christoffersen.independence_stat.toFixed(4)}</div>
                <div className={`text-sm mt-1 ${varResult.christoffersen.p_value >= 0.05 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  p-value: {varResult.christoffersen.p_value.toFixed(4)}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </section>

      {/* Panel B: DM Test */}
      <section>
        <h2 className="text-base font-semibold text-slate-200 mb-3">Panel B — Strategy Comparison (DM Test)</h2>
        <div className="flex items-center gap-4 mb-3">
          <Label>Compare with:</Label>
          {strategyOptions.map((s) => (
            <label key={s} className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="compare"
                checked={compareStrategy === s}
                onChange={() => setCompareStrategy(s)}
                className="accent-sky-500"
              />
              <span className="text-sm text-slate-300">{s}</span>
            </label>
          ))}
          <Button
            onClick={handleDMTest}
            disabled={isSubmittingCompare || compareJob?.status === 'running' || compareJob?.status === 'pending'}
            size="sm"
          >
            Run Comparison
          </Button>
          {compareJob && <StatusBadge status={compareJob.status} />}
        </div>
        {compareJob && <JobProgressBar status={compareJob.status} />}

        {dmResult && compareResult && (
          <>
            <Card className="mb-3">
              <CardHeader><CardTitle>Diebold-Mariano Test</CardTitle></CardHeader>
              <CardContent>
                <span>DM Stat: {dmResult.dm_stat.toFixed(4)}</span>
                <span className={`ml-4 text-base ${dmResult.p_value < 0.05 ? 'text-emerald-400' : 'text-slate-400'}`}>
                  p = {dmResult.p_value.toFixed(4)}
                </span>
              </CardContent>
            </Card>

            <div className="grid grid-cols-2 gap-4 mb-4 text-sm">
              {[
                { label: lastRequest.strategy, m: lastResult.metrics },
                { label: compareStrategy, m: compareResult.metrics },
              ].map(({ label, m }) => (
                <Card key={label}>
                  <CardHeader><CardTitle>{label}</CardTitle></CardHeader>
                  <div className="space-y-1 text-slate-300 text-sm">
                    <div>Sharpe: {m.sharpe_ratio.toFixed(3)}</div>
                    <div>MDD: {(m.max_drawdown * 100).toFixed(2)}%</div>
                    <div>Calmar: {m.calmar_ratio.toFixed(3)}</div>
                  </div>
                </Card>
              ))}
            </div>

            <h3 className="text-sm text-slate-400 mb-2">Equity Curve Overlay</h3>
            <EquityChart
              equityCurve={lastResult.equity_curve}
              label={lastRequest.strategy}
              equityCurve2={compareResult.equity_curve}
              label2={compareStrategy}
            />
          </>
        )}
      </section>
    </div>
  )
}
