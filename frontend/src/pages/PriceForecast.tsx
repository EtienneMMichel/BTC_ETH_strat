import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { submitPriceForecast, getPriceForecastJob } from '@/lib/api/models'
import { MultiLineChart } from '@/components/charts/MultiLineChart'
import { RocCurveChart } from '@/components/charts/RocCurveChart'
import { CalibrationChart } from '@/components/charts/CalibrationChart'
import { HistogramChart } from '@/components/charts/HistogramChart'
import { AcfChart } from '@/components/charts/AcfChart'
import { ReliabilityChart } from '@/components/charts/ReliabilityChart'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import type {
  PriceForecastRequest,
  PriceForecastResult,
  ConfusionMatrix,
  PriceForecastModelMetrics,
} from '@/types/models'

// ── Constants ─────────────────────────────────────────────────────────────

const ALL_MODELS = ['tsmom', 'momentum', 'ema_crossover', 'hp_filter', 'kalman', 'logistic']

const MODEL_COLORS: Record<string, string> = {
  tsmom:        '#38bdf8',
  momentum:     '#a78bfa',
  ema_crossover:'#34d399',
  hp_filter:    '#fbbf24',
  kalman:       '#f87171',
  logistic:     '#fb923c',
}

const OUTPUT_TYPE_BADGE: Record<string, string> = {
  direction:   'bg-sky-900 text-sky-300',
  amplitude:   'bg-violet-900 text-violet-300',
  probability: 'bg-orange-900 text-orange-300',
}

const DEFAULT_RESOLUTIONS: Record<string, string> = {
  tsmom:        '1D',
  momentum:     '1D',
  ema_crossover:'1h',
  hp_filter:    '1D',
  kalman:       '1h',
  logistic:     '1h',
}

const RESOLUTION_OPTIONS = ['1min', '5min', '15min', '1h', '4h', '1D']
const HORIZON_OPTIONS = ['1h', '4h', '8h', '1D', '3D', '1W']

// ── Confusion matrix components ───────────────────────────────────────────

function ConfusionBar({ cm }: { cm: ConfusionMatrix }) {
  const total = cm.tp + cm.fp + cm.tn + cm.fn
  if (total === 0) return null
  const bars = [
    { label: 'TP', value: cm.tp, color: '#34d399', title: 'True Positive' },
    { label: 'FP', value: cm.fp, color: '#f87171', title: 'False Positive' },
    { label: 'TN', value: cm.tn, color: '#38bdf8', title: 'True Negative' },
    { label: 'FN', value: cm.fn, color: '#fbbf24', title: 'False Negative' },
  ]
  const maxVal = Math.max(...bars.map((b) => b.value))
  return (
    <div className="mt-2 space-y-1">
      {bars.map(({ label, value, color, title }) => (
        <div key={label} className="flex items-center gap-2" title={title}>
          <span className="w-5 text-xs text-slate-400 shrink-0">{label}</span>
          <div className="flex-1 h-2.5 bg-slate-800 rounded-sm overflow-hidden">
            <div className="h-full rounded-sm" style={{ width: `${(value / maxVal) * 100}%`, background: color }} />
          </div>
          <span className="w-8 text-right text-xs text-slate-400 shrink-0">{value}</span>
        </div>
      ))}
    </div>
  )
}

function ConfusionGrid({ cm }: { cm: ConfusionMatrix }) {
  const cell = (label: string, value: number, color: string, desc: string) => (
    <div
      className="flex flex-col items-center justify-center rounded p-1 text-center"
      style={{ background: color + '22', border: `1px solid ${color}44` }}
      title={desc}
    >
      <span className="text-[9px] text-slate-400 leading-none">{label}</span>
      <span className="text-xs font-semibold mt-0.5" style={{ color }}>{value}</span>
    </div>
  )
  return (
    <div className="mt-1.5">
      <div className="grid grid-cols-2 gap-[2px] text-[8px] text-slate-500 mb-0.5">
        <span className="text-center">Pred +</span>
        <span className="text-center">Pred −</span>
      </div>
      <div className="grid grid-cols-2 gap-[2px]">
        {cell('TP', cm.tp, '#34d399', 'True Positive')}
        {cell('FN', cm.fn, '#fbbf24', 'False Negative')}
        {cell('FP', cm.fp, '#f87171', 'False Positive')}
        {cell('TN', cm.tn, '#38bdf8', 'True Negative')}
      </div>
    </div>
  )
}

// ── Metric number helpers ─────────────────────────────────────────────────

function pct(v: number | null | undefined) {
  if (v == null || isNaN(v)) return '—'
  return (v * 100).toFixed(1) + '%'
}

function fmt4(v: number | null | undefined) {
  if (v == null || isNaN(v)) return '—'
  return v.toFixed(4)
}

function fmt3(v: number | null | undefined) {
  if (v == null || isNaN(v)) return '—'
  return v.toFixed(3)
}

// ── Page ──────────────────────────────────────────────────────────────────

export function PriceForecastPage() {
  const [asset, setAsset] = useState<'BTC' | 'ETH'>('BTC')
  const [horizon, setHorizon] = useState('1D')
  const [minTrainBars, setMinTrainBars] = useState(252)
  const [selectedModels, setSelectedModels] = useState<string[]>(ALL_MODELS.filter((m) => m !== 'logistic'))
  const [modelResolutions, setModelResolutions] = useState<Record<string, string>>(DEFAULT_RESOLUTIONS)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)

  const { data: job } = useQuery({
    queryKey: ['price-forecast', jobId],
    queryFn: () => getPriceForecastJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000,
  })

  const handleSubmit = async () => {
    setIsSubmitting(true)
    try {
      const req: PriceForecastRequest = {
        asset,
        forecast_horizon: horizon,
        min_train_bars: minTrainBars,
        models: selectedModels,
        model_resolutions: modelResolutions,
      }
      const { job_id } = await submitPriceForecast(req)
      setJobId(job_id)
    } finally {
      setIsSubmitting(false)
    }
  }

  const toggleModel = (m: string) =>
    setSelectedModels((prev) => prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m])

  const setRes = (model: string, res: string) =>
    setModelResolutions((prev) => ({ ...prev, [model]: res }))

  const result: PriceForecastResult | undefined = job?.result

  // Signal series for MultiLineChart (at model resolution, downsampled to daily for display)
  const signalSeries = useMemo(() => {
    if (!result) return []
    return Object.entries(result.signals).map(([name, data]) => ({
      data,
      color: MODEL_COLORS[name] ?? '#ffffff',
      title: `${name}@${result.model_resolutions[name] ?? '?'}`,
    }))
  }, [result])

  // ROC curves
  const rocCurves = useMemo(() => {
    if (!result) return []
    return Object.entries(result.metrics).map(([name, m]) => ({
      name,
      color: MODEL_COLORS[name] ?? '#ffffff',
      points: m.direction.roc_curve,
      auc: m.direction.auc,
    }))
  }, [result])

  // Models with logistic output
  const logisticModels = useMemo(() => {
    if (!result) return []
    return Object.entries(result.metrics)
      .filter(([, m]) => m.output_type === 'probability')
      .map(([name]) => name)
  }, [result])

  const headerTag = result
    ? `${result.asset} | Horizon: ${result.forecast_horizon} | ${
        Object.entries(result.model_resolutions)
          .map(([k, v]) => `${k}@${v}`)
          .join(' ')
      }`
    : null

  return (
    <div className="flex gap-6 h-full">
      {/* ── Left panel ── */}
      <div className="w-72 shrink-0 flex flex-col gap-4 overflow-y-auto pr-1">
        <h1 className="text-lg font-semibold text-slate-100">Price Forecast</h1>

        {/* Asset */}
        <div>
          <Label className="mb-2 block text-xs text-slate-400">Asset</Label>
          <div className="flex gap-2">
            {(['BTC', 'ETH'] as const).map((a) => (
              <button
                key={a}
                onClick={() => setAsset(a)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  asset === a
                    ? 'bg-sky-600 text-white'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {a}
              </button>
            ))}
          </div>
        </div>

        {/* Forecast horizon pills */}
        <div>
          <Label className="mb-2 block text-xs text-slate-400">Forecast Horizon</Label>
          <div className="flex flex-wrap gap-1">
            {HORIZON_OPTIONS.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  horizon === h
                    ? 'bg-sky-600 text-white'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        </div>

        {/* Min train bars */}
        <div>
          <Label className="mb-1 block text-xs text-slate-400">Min train bars: {minTrainBars}</Label>
          <input
            type="range" min={50} max={500} step={10}
            value={minTrainBars}
            onChange={(e) => setMinTrainBars(Number(e.target.value))}
            className="w-full accent-sky-500"
          />
        </div>

        {/* Models + resolution dropdowns */}
        <div>
          <Label className="mb-2 block text-xs text-slate-400">Models</Label>
          <div className="space-y-2">
            {ALL_MODELS.map((m) => {
              const isHpFilter = m === 'hp_filter'
              return (
                <div key={m} className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(m)}
                    onChange={() => toggleModel(m)}
                    className="accent-sky-500 shrink-0"
                  />
                  <span
                    className="inline-block w-2 h-2 rounded-full shrink-0"
                    style={{ background: MODEL_COLORS[m] }}
                  />
                  <span className="text-xs text-slate-300 flex-1 truncate">{m}</span>
                  <select
                    value={modelResolutions[m] ?? '1D'}
                    onChange={(e) => setRes(m, e.target.value)}
                    disabled={isHpFilter}
                    className="text-xs bg-slate-800 text-slate-300 border border-slate-700 rounded px-1 py-0.5 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    {RESOLUTION_OPTIONS.map((r) => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </div>
              )
            })}
          </div>
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
              <span className="text-xs text-slate-400">Status:</span>
              <StatusBadge status={job.status} />
            </div>
            <JobProgressBar status={job.status} />
          </div>
        )}
        {job?.error && <p className="text-xs text-rose-400">{job.error}</p>}
      </div>

      {/* ── Right panel ── */}
      <div className="flex-1 flex flex-col gap-6 min-w-0 overflow-y-auto">
        {result ? (
          <>
            {/* Header bar */}
            <div className="text-sm text-slate-400 font-mono">{headerTag}</div>

            {/* Warnings */}
            {result.warnings && result.warnings.length > 0 && (
              <div className="bg-amber-950 border border-amber-700 rounded-lg px-3 py-2 text-xs text-amber-300 space-y-0.5">
                {result.warnings.map((w, i) => <div key={i}>{w}</div>)}
              </div>
            )}

            {/* ── Section A: Direction ── */}
            <section>
              <h2 className="text-sm font-semibold text-sky-400 mb-3">A — Direction</h2>

              {/* Metric cards */}
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                {Object.entries(result.metrics).map(([name, m]) => (
                  <ModelDirectionCard key={name} name={name} m={m} />
                ))}
              </div>

              {/* ROC chart */}
              {rocCurves.length > 0 && (
                <div className="rounded-lg overflow-hidden bg-slate-900 p-3">
                  <RocCurveChart curves={rocCurves} height={280} />
                  <p className="text-xs text-slate-600 mt-1">
                    Dashed diagonal = random classifier (AUC 0.5). Higher/left = better.
                  </p>
                </div>
              )}
            </section>

            {/* ── Section B: Amplitude ── */}
            <section>
              <h2 className="text-sm font-semibold text-violet-400 mb-3">B — Amplitude</h2>

              {/* IC cards */}
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                {Object.entries(result.metrics).map(([name, m]) => (
                  <AmplitudeCard key={name} name={name} m={m} />
                ))}
              </div>

              {/* Calibration charts */}
              <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                {Object.entries(result.metrics).map(([name, m]) => (
                  <Card key={name}>
                    <CardHeader className="pb-1">
                      <CardTitle className="text-xs flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full inline-block" style={{ background: MODEL_COLORS[name] }} />
                        {name} — Calibration
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <CalibrationChart
                        data={m.amplitude.calibration_bins}
                        color={MODEL_COLORS[name] ?? '#fff'}
                        height={160}
                      />
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Probability sub-section (logistic only) */}
              {logisticModels.length > 0 && (
                <div>
                  <h3 className="text-xs font-medium text-orange-400 mb-2">Probability metrics (logistic)</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {logisticModels.map((name) => {
                      const m = result.metrics[name]
                      return (
                        <Card key={name}>
                          <CardHeader className="pb-1">
                            <CardTitle className="text-xs flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full inline-block" style={{ background: MODEL_COLORS[name] }} />
                              {name} — Reliability
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pt-0 space-y-2">
                            <ReliabilityChart
                              data={m.amplitude.reliability_diagram ?? []}
                              color={MODEL_COLORS[name] ?? '#fff'}
                              height={200}
                            />
                            <div className="flex gap-4 text-xs text-slate-400">
                              <span>Brier: <span className="text-slate-200">{fmt4(m.amplitude.brier_score)}</span></span>
                              <span>Log-Loss: <span className="text-slate-200">{fmt4(m.amplitude.log_loss)}</span></span>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                </div>
              )}
            </section>

            {/* ── Section C: Additional Data ── */}
            <section>
              <h2 className="text-sm font-semibold text-emerald-400 mb-3">C — Additional Data</h2>

              {/* Signals overlay */}
              {signalSeries.length > 0 && (
                <div className="mb-4">
                  <p className="text-xs text-slate-500 mb-1">Signals (at model resolution)</p>
                  <MultiLineChart series={signalSeries} height={220} />
                </div>
              )}

              {/* Per-model histogram + ACF */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {Object.entries(result.metrics).map(([name, m]) => {
                  const turn = m.additional.turnover
                  const turnHigh = !isNaN(turn) && turn > 0.3
                  const nObs = m.additional.signal_histogram.reduce((s, b) => s + b.count, 0)
                  return (
                    <Card key={name}>
                      <CardHeader className="pb-1">
                        <CardTitle className="text-xs flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full inline-block" style={{ background: MODEL_COLORS[name] }} />
                          <span>{name}</span>
                          <span className={`text-[10px] ${turnHigh ? 'text-amber-400' : 'text-slate-500'}`}>
                            Turnover: {isNaN(turn) ? '—' : (turn * 100).toFixed(1) + '%'}
                          </span>
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="pt-0 flex gap-3">
                        <div className="flex-1 min-w-0">
                          <p className="text-[9px] text-slate-600 mb-0.5">Signal distribution</p>
                          <HistogramChart
                            data={m.additional.signal_histogram}
                            color={MODEL_COLORS[name] ?? '#fff'}
                            height={140}
                          />
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="text-[9px] text-slate-600 mb-0.5">ACF (±1.96/√n)</p>
                          <AcfChart
                            data={m.additional.acf}
                            color={MODEL_COLORS[name] ?? '#fff'}
                            n_obs={nObs}
                            height={140}
                          />
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </section>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
            Configure and submit to compare price forecast models.
          </div>
        )}
      </div>
    </div>
  )
}

// ── Sub-components ────────────────────────────────────────────────────────

function ModelDirectionCard({ name, m }: { name: string; m: PriceForecastModelMetrics }) {
  const d = m.direction
  const hitOk = d.hit_rate > 0.5
  const aucOk = d.auc > 0.5
  return (
    <Card>
      <CardHeader className="pb-1">
        <CardTitle className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full inline-block" style={{ background: MODEL_COLORS[name] ?? '#fff' }} />
          {name}
          <span className={`text-[9px] px-1.5 py-0.5 rounded-full ${OUTPUT_TYPE_BADGE[m.output_type] ?? ''}`}>
            {m.output_type}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1 pt-0">
        <div className="grid grid-cols-2 gap-x-2 text-xs">
          <span className={hitOk ? 'text-emerald-400' : 'text-rose-400'}>
            Hit: {pct(d.hit_rate)}
          </span>
          <span className="text-slate-400">
            Prec: {pct(d.precision)}
          </span>
          <span className="text-slate-400">
            Rec: {pct(d.recall)}
          </span>
          <span className="text-slate-400">
            F1: {pct(d.f1)}
          </span>
        </div>
        <div className={`text-xs ${aucOk ? 'text-emerald-400' : 'text-rose-400'}`}>
          AUC: {fmt3(d.auc)}
        </div>
        {d.confusion && (
          <>
            <ConfusionBar cm={d.confusion} />
            <ConfusionGrid cm={d.confusion} />
          </>
        )}
      </CardContent>
    </Card>
  )
}

function AmplitudeCard({ name, m }: { name: string; m: PriceForecastModelMetrics }) {
  const a = m.amplitude
  const icOk = !isNaN(a.ic) && a.ic > 0
  const rankOk = !isNaN(a.rank_ic) && a.rank_ic > 0
  return (
    <Card>
      <CardHeader className="pb-1">
        <CardTitle className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full inline-block" style={{ background: MODEL_COLORS[name] ?? '#fff' }} />
          {name}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1 pt-0">
        <div className="grid grid-cols-2 gap-x-2 text-xs">
          <span className={icOk ? 'text-emerald-400' : 'text-rose-400'}>
            IC: {fmt4(a.ic)}
          </span>
          <span className={rankOk ? 'text-emerald-400' : 'text-rose-400'}>
            Rank IC: {fmt4(a.rank_ic)}
          </span>
          <span className="text-slate-400">
            MAE: {fmt4(a.mae)}
          </span>
          <span className="text-slate-400">
            MSE: {fmt4(a.mse)}
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
