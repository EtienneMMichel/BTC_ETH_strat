import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { submitVolEval, getVolEvalJob } from '@/lib/api/models'
import { submitCovEval, getCovEvalJob } from '@/lib/api/markowitz'
import { MultiLineChart } from '@/components/charts/MultiLineChart'
import { EyeButton } from '@/components/ui/EyeButton'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import { COV_COLORS } from '@/lib/constants/markowitz-colors'
import type { VolEvalRequest, VolEvalMultiResult } from '@/types/models'
import type { CovEvalResult } from '@/types/markowitz'

const ALL_MODELS = ['garch', 'gjr_garch', 'egarch', 'ewma', 'rogers_satchell', 'yang_zhang']
const ALL_ASSETS = ['BTC', 'ETH']

const MODEL_COLORS: Record<string, string> = {
  garch: '#38bdf8',
  gjr_garch: '#a78bfa',
  egarch: '#34d399',
  ewma: '#fbbf24',
  rogers_satchell: '#f87171',
  yang_zhang: '#e879f9',
  realised: '#94a3b8',
}

type SortKey = 'qlike' | 'mse' | 'mae'
type TopTab = 'volatility' | 'covariance'

export function VolEvalPage() {
  const [topTab, setTopTab] = useState<TopTab>('volatility')

  // ── Volatility state ──
  const [selectedAssets, setSelectedAssets] = useState<string[]>(['BTC'])
  const [freq, setFreq] = useState('1D')
  const [minTrainBars, setMinTrainBars] = useState(252)
  const [testSize, setTestSize] = useState(21)
  const [selectedModels, setSelectedModels] = useState<string[]>(ALL_MODELS)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [sortKey, setSortKey] = useState<SortKey>('qlike')
  const [activeAsset, setActiveAsset] = useState('BTC')
  const [hiddenModels, setHiddenModels] = useState<Set<string>>(new Set())

  // ── Covariance state ──
  const [covModels, setCovModels] = useState<string[]>(['rolling', 'diagonal', 'bekk'])
  const [covFreq, setCovFreq] = useState('1D')
  const [covWindow, setCovWindow] = useState(63)
  const [covMinHistory, setCovMinHistory] = useState(252)
  const [covJobId, setCovJobId] = useState<string | null>(null)
  const [isCovSubmitting, setIsCovSubmitting] = useState(false)
  const [hiddenCov, setHiddenCov] = useState<Set<string>>(new Set())

  // ── Volatility query ──
  const { data: job } = useQuery({
    queryKey: ['vol-eval', jobId],
    queryFn: () => getVolEvalJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000,
  })

  // ── Covariance query ──
  const { data: covJob } = useQuery({
    queryKey: ['cov-eval', covJobId],
    queryFn: () => getCovEvalJob(covJobId!),
    enabled: !!covJobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000,
  })

  // ── Volatility handlers ──
  const handleSubmit = async () => {
    setIsSubmitting(true)
    setHiddenModels(new Set())
    try {
      const req: VolEvalRequest = {
        target_assets: selectedAssets,
        freq,
        min_train_bars: minTrainBars,
        test_size: testSize,
        models: selectedModels,
      }
      const { job_id } = await submitVolEval(req)
      setJobId(job_id)
      setActiveAsset(selectedAssets[0] ?? 'BTC')
    } finally {
      setIsSubmitting(false)
    }
  }

  const toggleModel = (m: string) =>
    setSelectedModels((prev) => prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m])

  const toggleAsset = (a: string) =>
    setSelectedAssets((prev) => prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a])

  const toggleHidden = (name: string) => {
    setHiddenModels((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  // ── Covariance handlers ──
  const handleCovSubmit = async () => {
    setIsCovSubmitting(true)
    setHiddenCov(new Set())
    try {
      const { job_id } = await submitCovEval({
        freq: covFreq,
        models: covModels,
        window: covWindow,
        min_history: covMinHistory,
      })
      setCovJobId(job_id)
    } finally {
      setIsCovSubmitting(false)
    }
  }

  const toggleCovModel = (m: string) =>
    setCovModels((prev) => prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m])

  const toggleHiddenCov = (name: string) =>
    setHiddenCov((prev) => { const s = new Set(prev); s.has(name) ? s.delete(name) : s.add(name); return s })

  // ── Volatility derived ──
  const result: VolEvalMultiResult | undefined = job?.result
  const availableAssets = result ? Object.keys(result.per_asset) : []
  const currentAsset = availableAssets.includes(activeAsset) ? activeAsset : (availableAssets[0] ?? 'BTC')
  const assetData = result?.per_asset?.[currentAsset]

  const chartSeries = useMemo(() => {
    if (!assetData) return []
    const series = Object.entries(assetData.forecasts)
      .filter(([name]) => !hiddenModels.has(name))
      .map(([name, data]) => ({
        data,
        color: MODEL_COLORS[name] ?? '#ffffff',
        title: name,
      }))
    if (assetData.realised.length > 0 && !hiddenModels.has('realised')) {
      series.push({ data: assetData.realised, color: MODEL_COLORS.realised, title: 'Realised' })
    }
    return series
  }, [assetData, hiddenModels])

  const rollingQLikeSeries = useMemo(() => {
    if (!assetData?.rolling_metrics) return []
    return Object.entries(assetData.rolling_metrics)
      .filter(([name]) => !hiddenModels.has(name))
      .map(([name, rm]) => ({
        data: rm.qlike,
        color: MODEL_COLORS[name] ?? '#ffffff',
        title: name,
      }))
  }, [assetData, hiddenModels])

  const sortedModels = useMemo(() => {
    if (!assetData) return []
    return Object.entries(assetData.comparison_table).sort(
      ([, a], [, b]) => (a[sortKey] ?? Infinity) - (b[sortKey] ?? Infinity)
    )
  }, [assetData, sortKey])

  // ── Covariance derived ──
  const covResult: CovEvalResult | undefined = covJob?.result as CovEvalResult | undefined
  const isCovRunning = covJob?.status === 'running' || covJob?.status === 'pending'

  const covSeriesVarBTC = useMemo(() => {
    if (!covResult) return []
    const out = Object.entries(covResult.models)
      .filter(([name]) => !hiddenCov.has(name))
      .map(([name, s]) => ({ data: s.var_BTC, color: COV_COLORS[name] ?? '#fff', title: name }))
    if (!hiddenCov.has('realised'))
      out.push({ data: covResult.realised.var_BTC, color: COV_COLORS.realised, title: 'realised' })
    return out
  }, [covResult, hiddenCov])

  const covSeriesVarETH = useMemo(() => {
    if (!covResult) return []
    const out = Object.entries(covResult.models)
      .filter(([name]) => !hiddenCov.has(name))
      .map(([name, s]) => ({ data: s.var_ETH, color: COV_COLORS[name] ?? '#fff', title: name }))
    if (!hiddenCov.has('realised'))
      out.push({ data: covResult.realised.var_ETH, color: COV_COLORS.realised, title: 'realised' })
    return out
  }, [covResult, hiddenCov])

  const covSeriesCov = useMemo(() => {
    if (!covResult) return []
    const out = Object.entries(covResult.models)
      .filter(([name]) => !hiddenCov.has(name))
      .map(([name, s]) => ({ data: s.cov_BTC_ETH, color: COV_COLORS[name] ?? '#fff', title: name }))
    if (!hiddenCov.has('realised'))
      out.push({ data: covResult.realised.cov_BTC_ETH, color: COV_COLORS.realised, title: 'realised' })
    return out
  }, [covResult, hiddenCov])

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* ── Top-level tab bar ── */}
      <div className="flex gap-1 shrink-0">
        <button
          onClick={() => setTopTab('volatility')}
          className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
            topTab === 'volatility' ? 'bg-sky-700 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          Volatility
        </button>
        <button
          onClick={() => setTopTab('covariance')}
          className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
            topTab === 'covariance' ? 'bg-violet-700 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          Covariance
        </button>
      </div>

      {/* ── Volatility tab ── */}
      {topTab === 'volatility' && (
        <div className="flex gap-6 flex-1 min-h-0">
          {/* Left panel */}
          <div className="w-80 shrink-0 flex flex-col gap-4">
            <h2 className="text-lg font-semibold text-slate-100">Volatility Evaluation</h2>

            <div>
              <Label className="mb-2 block">Assets</Label>
              {ALL_ASSETS.map((a) => (
                <label key={a} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input
                    type="checkbox"
                    checked={selectedAssets.includes(a)}
                    onChange={() => toggleAsset(a)}
                    className="accent-sky-500"
                  />
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
              disabled={isSubmitting || job?.status === 'running' || job?.status === 'pending' || selectedModels.length === 0 || selectedAssets.length === 0}
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
          <div className="flex-1 flex flex-col gap-6 min-w-0 overflow-y-auto">
            {result ? (
              <>
                {availableAssets.length > 1 && (
                  <div className="flex gap-1">
                    {availableAssets.map((a) => (
                      <button
                        key={a}
                        onClick={() => setActiveAsset(a)}
                        className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                          currentAsset === a
                            ? 'bg-sky-700 text-white'
                            : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                        }`}
                      >
                        {a}
                      </button>
                    ))}
                  </div>
                )}

                {assetData && (
                  <>
                    <div>
                      <h2 className="text-sm font-medium text-slate-400 mb-2">Volatility Forecasts — {currentAsset}</h2>
                      <MultiLineChart series={chartSeries} height={300} />
                    </div>

                    {rollingQLikeSeries.length > 0 && (
                      <div>
                        <h2 className="text-sm font-medium text-slate-400 mb-2">Rolling QLIKE (63-bar) — {currentAsset}</h2>
                        <MultiLineChart series={rollingQLikeSeries} height={200} />
                      </div>
                    )}

                    <div>
                      <div className="flex items-center gap-4 mb-3">
                        <h2 className="text-sm font-medium text-slate-400">Comparison Table — {currentAsset}</h2>
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
                              <th className="py-2 px-2 w-8"></th>
                            </tr>
                          </thead>
                          <tbody>
                            {sortedModels.map(([name, m], i) => (
                              <tr key={name} className={`border-b border-slate-800 ${i === 0 ? 'text-emerald-400' : ''}`}>
                                <td className="py-2 pr-4">
                                  <div className="flex items-center gap-2">
                                    <span
                                      className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                                      style={{ background: MODEL_COLORS[name] ?? '#ffffff', opacity: hiddenModels.has(name) ? 0.3 : 1 }}
                                    />
                                    <span className={hiddenModels.has(name) ? 'opacity-40' : ''}>{name}</span>
                                  </div>
                                </td>
                                <td className="text-right px-3">{m.qlike?.toFixed(6) ?? '\u2014'}</td>
                                <td className="text-right px-3">{m.mse?.toFixed(8) ?? '\u2014'}</td>
                                <td className="text-right px-3">{m.mae?.toFixed(6) ?? '\u2014'}</td>
                                <td className="px-2 text-center">
                                  <EyeButton
                                    visible={!hiddenModels.has(name)}
                                    onToggle={() => toggleHidden(name)}
                                  />
                                </td>
                              </tr>
                            ))}
                            <tr className="border-b border-slate-800">
                              <td className="py-2 pr-4">
                                <div className="flex items-center gap-2">
                                  <span
                                    className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                                    style={{ background: MODEL_COLORS.realised, opacity: hiddenModels.has('realised') ? 0.3 : 1 }}
                                  />
                                  <span className={`text-slate-400 ${hiddenModels.has('realised') ? 'opacity-40' : ''}`}>realised</span>
                                </div>
                              </td>
                              <td className="text-right px-3 text-slate-600">{'\u2014'}</td>
                              <td className="text-right px-3 text-slate-600">{'\u2014'}</td>
                              <td className="text-right px-3 text-slate-600">{'\u2014'}</td>
                              <td className="px-2 text-center">
                                <EyeButton
                                  visible={!hiddenModels.has('realised')}
                                  onToggle={() => toggleHidden('realised')}
                                />
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                )}
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
                Configure and submit to compare volatility models.
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Covariance tab ── */}
      {topTab === 'covariance' && (
        <div className="flex gap-6 flex-1 min-h-0">
          {/* Left panel */}
          <div className="w-80 shrink-0 flex flex-col gap-4">
            <h2 className="text-lg font-semibold text-slate-100">Covariance Evaluation</h2>

            <div>
              <Label className="mb-2 block">Models</Label>
              {(['rolling', 'diagonal', 'bekk'] as const).map((m) => (
                <label key={m} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input
                    type="checkbox"
                    checked={covModels.includes(m)}
                    onChange={() => toggleCovModel(m)}
                    className="accent-violet-500"
                  />
                  <span className="text-sm text-slate-300 flex items-center gap-2">
                    <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: COV_COLORS[m] }} />
                    {m}
                  </span>
                </label>
              ))}
            </div>

            <div>
              <Label className="mb-2 block">Frequency</Label>
              {['1D', '4h', '1h'].map((f) => (
                <label key={f} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input type="radio" name="cov-freq" checked={covFreq === f} onChange={() => setCovFreq(f)} className="accent-violet-500" />
                  <span className="text-sm text-slate-300">{f}</span>
                </label>
              ))}
            </div>

            <div>
              <Label className="mb-1 block">Rolling window: {covWindow}</Label>
              <input
                type="range" min={5} max={126} step={1}
                value={covWindow} onChange={(e) => setCovWindow(Number(e.target.value))}
                className="w-full accent-violet-500"
              />
            </div>

            <div>
              <Label className="mb-1 block">Min history: {covMinHistory}</Label>
              <input
                type="range" min={10} max={500} step={10}
                value={covMinHistory} onChange={(e) => setCovMinHistory(Number(e.target.value))}
                className="w-full accent-violet-500"
              />
            </div>

            <Button
              onClick={handleCovSubmit}
              disabled={isCovSubmitting || isCovRunning || covModels.length === 0}
              className="bg-violet-700 hover:bg-violet-600 text-white"
            >
              Run
            </Button>

            {covJob && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-slate-400">Status:</span>
                  <StatusBadge status={covJob.status} />
                </div>
                <JobProgressBar status={covJob.status} />
              </div>
            )}
            {covJob?.error && <p className="text-sm text-rose-400">{covJob.error}</p>}
          </div>

          {/* Right panel */}
          <div className="flex-1 flex flex-col gap-6 min-w-0 overflow-y-auto">
            {covResult ? (
              <>
                <div>
                  <h2 className="text-sm font-medium text-slate-400 mb-2">Variance BTC</h2>
                  <MultiLineChart series={covSeriesVarBTC} height={220} />
                </div>
                <div>
                  <h2 className="text-sm font-medium text-slate-400 mb-2">Variance ETH</h2>
                  <MultiLineChart series={covSeriesVarETH} height={220} />
                </div>
                <div>
                  <h2 className="text-sm font-medium text-slate-400 mb-2">Covariance BTC-ETH</h2>
                  <MultiLineChart series={covSeriesCov} height={220} />
                </div>

                {/* Comparison table */}
                <div>
                  <h2 className="text-sm font-medium text-slate-400 mb-2">Comparison vs. Realised (rolling-21)</h2>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-slate-300 border-collapse">
                      <thead>
                        <tr className="border-b border-slate-700 text-slate-500 text-xs">
                          <th className="text-left py-2 pr-4">Model</th>
                          <th className="text-right py-2 px-3">RMSE var_BTC</th>
                          <th className="text-right py-2 px-3">RMSE var_ETH</th>
                          <th className="text-right py-2 px-3">Corr cov</th>
                          <th className="py-2 px-2 w-8"></th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(covResult.comparison).map(([name, m]) => (
                          <tr key={name} className="border-b border-slate-800">
                            <td className="py-2 pr-4">
                              <div className="flex items-center gap-2">
                                <span
                                  className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                                  style={{ background: COV_COLORS[name] ?? '#fff', opacity: hiddenCov.has(name) ? 0.3 : 1 }}
                                />
                                <span className={hiddenCov.has(name) ? 'opacity-40' : ''}>{name}</span>
                              </div>
                            </td>
                            <td className="text-right px-3">{m.rmse_var_BTC?.toExponential(3) ?? '\u2014'}</td>
                            <td className="text-right px-3">{m.rmse_var_ETH?.toExponential(3) ?? '\u2014'}</td>
                            <td className="text-right px-3">{m.corr_cov?.toFixed(4) ?? '\u2014'}</td>
                            <td className="px-2 text-center">
                              <EyeButton visible={!hiddenCov.has(name)} onToggle={() => toggleHiddenCov(name)} />
                            </td>
                          </tr>
                        ))}
                        {/* realised row */}
                        <tr className="border-b border-slate-800">
                          <td className="py-2 pr-4">
                            <div className="flex items-center gap-2">
                              <span
                                className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                                style={{ background: COV_COLORS.realised, opacity: hiddenCov.has('realised') ? 0.3 : 1 }}
                              />
                              <span className={`text-slate-400 ${hiddenCov.has('realised') ? 'opacity-40' : ''}`}>realised</span>
                            </div>
                          </td>
                          <td className="text-right px-3 text-slate-600">{'\u2014'}</td>
                          <td className="text-right px-3 text-slate-600">{'\u2014'}</td>
                          <td className="text-right px-3 text-slate-600">{'\u2014'}</td>
                          <td className="px-2 text-center">
                            <EyeButton visible={!hiddenCov.has('realised')} onToggle={() => toggleHiddenCov('realised')} />
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
                Configure and run a covariance evaluation to see results.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
