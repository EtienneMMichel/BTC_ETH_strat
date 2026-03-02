import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { submitVolEval, getVolEvalJob } from '@/lib/api/models'
import { MultiLineChart } from '@/components/charts/MultiLineChart'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import type { VolEvalRequest, VolEvalResult } from '@/types/models'

const ALL_MODELS = ['garch', 'gjr_garch', 'egarch', 'ewma', 'rogers_satchell', 'yang_zhang']

const MODEL_COLORS: Record<string, string> = {
  garch: '#38bdf8',          // sky-400
  gjr_garch: '#a78bfa',      // violet-400
  egarch: '#34d399',         // emerald-400
  ewma: '#fbbf24',           // amber-400
  rogers_satchell: '#f87171',// red-400
  yang_zhang: '#e879f9',     // fuchsia-400
  realised: '#94a3b8',       // slate-400
}

type SortKey = 'qlike' | 'mse' | 'mae'

export function VolEvalPage() {
  const [asset, setAsset] = useState<'BTC' | 'ETH'>('BTC')
  const [freq, setFreq] = useState('1D')
  const [minTrainBars, setMinTrainBars] = useState(252)
  const [testSize, setTestSize] = useState(21)
  const [selectedModels, setSelectedModels] = useState<string[]>(ALL_MODELS)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<SortKey>('qlike')

  const { data: job } = useQuery({
    queryKey: ['vol-eval', jobId],
    queryFn: () => getVolEvalJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000,
  })

  const handleSubmit = async () => {
    setIsSubmitting(true)
    try {
      const req: VolEvalRequest = {
        asset,
        freq,
        min_train_bars: minTrainBars,
        test_size: testSize,
        models: selectedModels,
      }
      const { job_id } = await submitVolEval(req)
      setJobId(job_id)
    } finally {
      setIsSubmitting(false)
    }
  }

  const toggleModel = (m: string) => {
    setSelectedModels((prev) =>
      prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]
    )
  }

  const result: VolEvalResult | undefined = job?.result

  const chartSeries = useMemo(() => {
    if (!result) return []
    const series = Object.entries(result.forecasts).map(([name, data]) => ({
      data,
      color: MODEL_COLORS[name] ?? '#ffffff',
      title: name,
    }))
    if (result.realised.length > 0) {
      series.push({ data: result.realised, color: MODEL_COLORS.realised, title: 'Realised' })
    }
    return series
  }, [result])

  const sortedModels = useMemo(() => {
    if (!result) return []
    return Object.entries(result.comparison_table).sort(
      ([, a], [, b]) => (a[sortKey] ?? Infinity) - (b[sortKey] ?? Infinity)
    )
  }, [result, sortKey])

  return (
    <div className="flex gap-6 h-full">
      {/* Left panel */}
      <div className="w-80 shrink-0 flex flex-col gap-4">
        <h1 className="text-lg font-semibold text-slate-100">Volatility Evaluation</h1>

        <div>
          <Label className="mb-2 block">Asset</Label>
          {(['BTC', 'ETH'] as const).map((a) => (
            <label key={a} className="flex items-center gap-2 cursor-pointer mb-1">
              <input type="radio" name="asset" checked={asset === a} onChange={() => setAsset(a)} className="accent-sky-500" />
              <span className="text-sm text-slate-300">{a}</span>
            </label>
          ))}
        </div>

        <div>
          <Label className="mb-2 block">Frequency</Label>
          {['1D', '4h', '1h'].map((f) => (
            <label key={f} className="flex items-center gap-2 cursor-pointer mb-1">
              <input type="radio" name="freq" checked={freq === f} onChange={() => setFreq(f)} className="accent-sky-500" />
              <span className="text-sm text-slate-300">{f}</span>
            </label>
          ))}
        </div>

        <div>
          <Label className="mb-1 block">Min train bars: {minTrainBars}</Label>
          <input
            type="range" min={50} max={500} step={10}
            value={minTrainBars}
            onChange={(e) => setMinTrainBars(Number(e.target.value))}
            className="w-full accent-sky-500"
          />
        </div>

        <div>
          <Label className="mb-1 block">Test size: {testSize}</Label>
          <input
            type="range" min={5} max={63} step={1}
            value={testSize}
            onChange={(e) => setTestSize(Number(e.target.value))}
            className="w-full accent-sky-500"
          />
        </div>

        <div>
          <Label className="mb-2 block">Models</Label>
          {ALL_MODELS.map((m) => (
            <label key={m} className="flex items-center gap-2 cursor-pointer mb-1">
              <input
                type="checkbox"
                checked={selectedModels.includes(m)}
                onChange={() => toggleModel(m)}
                className="accent-sky-500"
              />
              <span className="text-sm text-slate-300 flex items-center gap-2">
                <span
                  className="inline-block w-2.5 h-2.5 rounded-full"
                  style={{ background: MODEL_COLORS[m] }}
                />
                {m}
              </span>
            </label>
          ))}
        </div>

        <Button
          onClick={handleSubmit}
          disabled={isSubmitting || job?.status === 'running' || job?.status === 'pending' || selectedModels.length === 0}
        >
          Run
        </Button>

        {job && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-sm text-slate-400">Status:</span>
              <StatusBadge status={job.status} />
            </div>
            <JobProgressBar status={job.status} />
          </div>
        )}
        {job?.error && <p className="text-sm text-rose-400">{job.error}</p>}
      </div>

      {/* Right panel */}
      <div className="flex-1 flex flex-col gap-6 min-w-0">
        {result ? (
          <>
            <div>
              <h2 className="text-sm font-medium text-slate-400 mb-2">Volatility Forecasts</h2>
              <MultiLineChart series={chartSeries} height={300} />
            </div>

            <div>
              <div className="flex items-center gap-4 mb-3">
                <h2 className="text-sm font-medium text-slate-400">Comparison Table</h2>
                <div className="flex gap-2">
                  {(['qlike', 'mse', 'mae'] as SortKey[]).map((k) => (
                    <button
                      key={k}
                      onClick={() => setSortKey(k)}
                      className={`text-xs px-2 py-1 rounded ${sortKey === k ? 'bg-sky-700 text-white' : 'text-slate-400 hover:text-slate-200'}`}
                    >
                      {k.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-slate-300 border-collapse">
                  <thead>
                    <tr className="border-b border-slate-700 text-slate-500 text-xs">
                      <th className="text-left py-2 pr-4">Model</th>
                      <th className="text-right py-2 px-3">QLIKE</th>
                      <th className="text-right py-2 px-3">MSE</th>
                      <th className="text-right py-2 px-3">MAE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedModels.map(([name, m], i) => (
                      <tr key={name} className={`border-b border-slate-800 ${i === 0 ? 'text-emerald-400' : ''}`}>
                        <td className="py-2 pr-4 flex items-center gap-2">
                          <span
                            className="inline-block w-2.5 h-2.5 rounded-full"
                            style={{ background: MODEL_COLORS[name] ?? '#ffffff' }}
                          />
                          {name}
                        </td>
                        <td className="text-right px-3">{m.qlike?.toFixed(6) ?? '—'}</td>
                        <td className="text-right px-3">{m.mse?.toFixed(8) ?? '—'}</td>
                        <td className="text-right px-3">{m.mae?.toFixed(6) ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
            Configure and submit to compare volatility models.
          </div>
        )}
      </div>
    </div>
  )
}
