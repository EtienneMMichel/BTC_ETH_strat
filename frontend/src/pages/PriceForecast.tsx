import { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { submitPriceForecast, getPriceForecastJob, getPriceForecastDataRange } from '@/lib/api/models'
import { submitERCovEval, getERCovEvalJob } from '@/lib/api/markowitz'
import { MultiLineChart } from '@/components/charts/MultiLineChart'
import { RocCurveChart } from '@/components/charts/RocCurveChart'
import { CalibrationChart } from '@/components/charts/CalibrationChart'
import { HistogramChart } from '@/components/charts/HistogramChart'
import { AcfChart } from '@/components/charts/AcfChart'
import { ReliabilityChart } from '@/components/charts/ReliabilityChart'
import { RegimeTimelineChart } from '@/components/charts/RegimeTimelineChart'
import { DateSplitPanel } from '@/components/price-forecast/DateSplitPanel'
import { AssetTabs } from '@/components/price-forecast/AssetTabs'
import { SaveRunDialog } from '@/components/price-forecast/SaveRunDialog'
import { SavedRunsPanel } from '@/components/price-forecast/SavedRunsPanel'
import { RegimeConditioningPanel } from '@/components/price-forecast/RegimeConditioningPanel'
import { EyeButton } from '@/components/ui/EyeButton'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import { ER_COLORS } from '@/lib/constants/markowitz-colors'
import type {
  PriceForecastRequest,
  PriceForecastResult,
  PerAssetForecastData,
  ConfusionMatrix,
  PriceForecastModelMetrics,
  AmplitudeMetrics,
  DirectionMetrics,
  AdditionalMetrics,
  RegimeClassifierType,
} from '@/types/models'
import type { ERCovEvalResult } from '@/types/markowitz'

// ── Constants ─────────────────────────────────────────────────────────────

const ALL_MODELS = ['tsmom', 'momentum', 'ema_crossover', 'hp_filter', 'kalman', 'logistic']

const MODEL_COLORS: Record<string, string> = {
  tsmom:          '#38bdf8',
  momentum:       '#a78bfa',
  ema_crossover:  '#34d399',
  hp_filter:      '#fbbf24',
  kalman:         '#f87171',
  logistic:       '#fb923c',
  cross_tsmom:    '#e879f9',
  cross_momentum: '#fb7185',
}

const REGIME_PALETTE = ['#38bdf8', '#34d399', '#fbbf24', '#f87171', '#a78bfa', '#fb923c']

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

// ── Normalize legacy flat result to per_asset shape ───────────────────────

function normalizeResult(res: PriceForecastResult): PriceForecastResult {
  if (res.per_asset) return res
  const assetName = res.asset ?? 'BTC'
  return {
    assets: [assetName],
    per_asset: {
      [assetName]: {
        signals: res.signals ?? {},
        metrics: res.metrics ?? {},
        prices: res.prices ?? [],
        model_resolutions: res.model_resolutions ?? {},
      },
    },
    forecast_horizon: res.forecast_horizon,
    warnings: res.warnings,
    train_period: res.train_period,
    test_period: res.test_period,
    regime_labels: res.regime_labels,
    regime_metrics: res.regime_metrics,
  }
}

// ── averageMetrics: average PriceForecastModelMetrics across regimes ───────

function averageMetrics(slices: PriceForecastModelMetrics[]): PriceForecastModelMetrics {
  if (slices.length === 1) return slices[0]
  const n = slices.length
  const mean = (vals: number[]) => vals.reduce((s, v) => s + v, 0) / n

  const direction: DirectionMetrics = {
    hit_rate:  mean(slices.map((s) => s.direction.hit_rate)),
    precision: mean(slices.map((s) => s.direction.precision)),
    recall:    mean(slices.map((s) => s.direction.recall)),
    f1:        mean(slices.map((s) => s.direction.f1)),
    auc:       mean(slices.map((s) => s.direction.auc)),
    roc_curve: slices[0].direction.roc_curve,
    confusion: {
      tp: slices.reduce((s, m) => s + m.direction.confusion.tp, 0),
      fp: slices.reduce((s, m) => s + m.direction.confusion.fp, 0),
      tn: slices.reduce((s, m) => s + m.direction.confusion.tn, 0),
      fn: slices.reduce((s, m) => s + m.direction.confusion.fn, 0),
    },
  }

  const hasBrier = slices[0].amplitude.brier_score != null
  const hasLogLoss = slices[0].amplitude.log_loss != null

  const amplitude: AmplitudeMetrics = {
    ic:       mean(slices.map((s) => s.amplitude.ic)),
    rank_ic:  mean(slices.map((s) => s.amplitude.rank_ic)),
    mae:      mean(slices.map((s) => s.amplitude.mae)),
    mse:      mean(slices.map((s) => s.amplitude.mse)),
    calibration_bins: slices[0].amplitude.calibration_bins,
    reliability_diagram: slices[0].amplitude.reliability_diagram,
    brier_score: hasBrier ? mean(slices.map((s) => s.amplitude.brier_score ?? 0)) : undefined,
    log_loss:    hasLogLoss ? mean(slices.map((s) => s.amplitude.log_loss ?? 0)) : undefined,
  }

  const additional: AdditionalMetrics = {
    turnover:         mean(slices.map((s) => s.additional.turnover)),
    signal_histogram: slices[0].additional.signal_histogram,
    acf:              slices[0].additional.acf,
  }

  return { direction, amplitude, additional, output_type: slices[0].output_type }
}

// ── Helpers ────────────────────────────────────────────────────────────────

function buildRegimeColorMap(labels: [string, string][]): Record<string, string> {
  const unique = [...new Set(labels.map(([, l]) => l))].sort()
  return Object.fromEntries(unique.map((l, i) => [l, REGIME_PALETTE[i % REGIME_PALETTE.length]]))
}

// ── Confusion matrix ───────────────────────────────────────────────────────

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

// ── Metric helpers ─────────────────────────────────────────────────────────

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

// ── Regime filter ──────────────────────────────────────────────────────────

function RegimeFilter({
  labels,
  selected,
  colorMap,
  onChange,
}: {
  labels: [string, string][]
  selected: string[]
  colorMap: Record<string, string>
  onChange: (v: string[]) => void
}) {
  const uniqueLabels = useMemo(() => [...new Set(labels.map(([, l]) => l))].sort(), [labels])
  const toggle = (l: string) =>
    onChange(selected.includes(l) ? selected.filter((x) => x !== l) : [...selected, l])

  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-[10px] text-slate-500 shrink-0">Regime filter:</span>
      <button
        onClick={() => onChange([])}
        className={`text-[10px] px-2 py-0.5 rounded-full transition-colors ${
          selected.length === 0
            ? 'bg-slate-600 text-white'
            : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
        }`}
      >
        All
      </button>
      {uniqueLabels.map((l) => (
        <button
          key={l}
          onClick={() => toggle(l)}
          className="text-[10px] px-2 py-0.5 rounded-full transition-colors border"
          style={
            selected.includes(l)
              ? { background: colorMap[l], borderColor: colorMap[l], color: '#fff' }
              : { background: 'transparent', borderColor: colorMap[l], color: colorMap[l] }
          }
        >
          {l}
        </button>
      ))}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────

type AssetMode = 'BTC' | 'ETH' | 'Both'
type TopTab = 'forecast' | 'er'

export function PriceForecastPage() {
  const [topTab, setTopTab] = useState<TopTab>('forecast')

  // ── Expected Returns state ──
  const [erModels, setErModels] = useState<string[]>(['rolling_mean', 'signal_tsmom'])
  const [erFreq, setErFreq] = useState('1D')
  const [erWindow, setErWindow] = useState(63)
  const [erMinHistory, setErMinHistory] = useState(100)
  const [erJobId, setErJobId] = useState<string | null>(null)
  const [isErSubmitting, setIsErSubmitting] = useState(false)
  const [hiddenEr, setHiddenEr] = useState<Set<string>>(new Set())
  const [activeErAsset, setActiveErAsset] = useState<'BTC' | 'ETH'>('BTC')

  const [assetMode, setAssetMode] = useState<AssetMode>('BTC')
  const [crossAsset, setCrossAsset] = useState(false)
  const [horizon, setHorizon] = useState('1D')
  const [selectedModels, setSelectedModels] = useState<string[]>(
    ALL_MODELS.filter((m) => m !== 'logistic')
  )
  const [modelResolutions, setModelResolutions] = useState<Record<string, string>>(DEFAULT_RESOLUTIONS)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [savedResult, setSavedResult] = useState<PriceForecastResult | null>(null)

  // Feature 1 — date split
  const [dateEnabled, setDateEnabled] = useState(false)
  const [trainStart, setTrainStart] = useState('')
  const [trainEnd, setTrainEnd] = useState('')
  const [testStart, setTestStart] = useState('')
  const [testEnd, setTestEnd] = useState('')

  // Feature 4 — regime conditioning
  const [regimeEnabled, setRegimeEnabled] = useState(false)
  const [regimeType, setRegimeType] = useState<RegimeClassifierType>('vol_quantile')
  const [regimeParams, setRegimeParams] = useState<Record<string, unknown>>({})

  // Regime filter (right panel)
  const [selectedRegimes, setSelectedRegimes] = useState<string[]>([])

  const [activeAsset, setActiveAsset] = useState<string>('BTC')

  // Feature 2 — hidden signals
  const [hiddenSignals, setHiddenSignals] = useState<Set<string>>(new Set())

  const { data: dataRanges } = useQuery({
    queryKey: ['price-forecast-data-range'],
    queryFn: getPriceForecastDataRange,
    staleTime: Infinity,
  })
  const dataRange = dataRanges?.['BTC']

  const { data: job } = useQuery({
    queryKey: ['price-forecast', jobId],
    queryFn: () => getPriceForecastJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000,
  })

  const assetList: ('BTC' | 'ETH')[] =
    assetMode === 'Both' ? ['BTC', 'ETH'] : [assetMode]

  const handleSubmit = async () => {
    setSavedResult(null)
    setSelectedRegimes([])
    setHiddenSignals(new Set())
    setIsSubmitting(true)
    try {
      const req: PriceForecastRequest = {
        assets: assetList,
        cross_asset: assetMode === 'Both' && crossAsset,
        forecast_horizon: horizon,
        models: selectedModels,
        model_resolutions: modelResolutions,
        ...(dateEnabled && trainStart && trainEnd && testStart && testEnd
          ? { train_start: trainStart, train_end: trainEnd, test_start: testStart, test_end: testEnd }
          : {}),
        ...(regimeEnabled
          ? {
              regime_conditioning: true,
              regime_classifier_type: regimeType,
              regime_classifier_params: regimeParams,
            }
          : {}),
      }
      const { job_id } = await submitPriceForecast(req)
      setJobId(job_id)
      setActiveAsset(assetList[0])
    } finally {
      setIsSubmitting(false)
    }
  }

  const toggleModel = (m: string) =>
    setSelectedModels((prev) => prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m])
  const setRes = (model: string, res: string) =>
    setModelResolutions((prev) => ({ ...prev, [model]: res }))
  const handleDateChange = (
    field: 'trainStart' | 'trainEnd' | 'testStart' | 'testEnd',
    value: string
  ) => {
    if (field === 'trainStart') setTrainStart(value)
    else if (field === 'trainEnd') setTrainEnd(value)
    else if (field === 'testStart') setTestStart(value)
    else setTestEnd(value)
  }
  const handleSetAllDates = (ts: string, te: string, ss: string, se: string) => {
    setTrainStart(ts); setTrainEnd(te); setTestStart(ss); setTestEnd(se)
  }

  // ── ER query ──
  const { data: erJob } = useQuery({
    queryKey: ['er-eval', erJobId],
    queryFn: () => getERCovEvalJob(erJobId!),
    enabled: !!erJobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000,
  })

  const handleErSubmit = async () => {
    setIsErSubmitting(true)
    setHiddenEr(new Set())
    try {
      const { job_id } = await submitERCovEval({
        freq: erFreq,
        models: erModels,
        er_window: erWindow,
        min_history: erMinHistory,
      })
      setErJobId(job_id)
    } finally {
      setIsErSubmitting(false)
    }
  }

  const toggleErModel = (m: string) =>
    setErModels((prev) => prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m])

  const toggleHiddenEr = (name: string) =>
    setHiddenEr((prev) => { const s = new Set(prev); s.has(name) ? s.delete(name) : s.add(name); return s })

  const erResult: ERCovEvalResult | undefined = erJob?.result as ERCovEvalResult | undefined
  const erAssetData = erResult?.per_asset?.[activeErAsset]
  const isErRunning = erJob?.status === 'running' || erJob?.status === 'pending'

  const erMuSeries = useMemo(() => {
    if (!erAssetData) return []
    return Object.entries(erAssetData.mu)
      .filter(([name]) => !hiddenEr.has(name))
      .map(([name, data]) => ({ data, color: ER_COLORS[name] ?? '#fff', title: name }))
  }, [erAssetData, hiddenEr])

  const toggleHiddenSignal = (name: string) => {
    setHiddenSignals((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const rawResult = savedResult ?? (job?.result ?? undefined)
  const result = useMemo(
    () => (rawResult ? normalizeResult(rawResult) : undefined),
    [rawResult]
  )

  const displayedAssets = result?.assets ?? []
  const currentAsset = displayedAssets.includes(activeAsset) ? activeAsset : (displayedAssets[0] ?? 'BTC')
  const assetData: PerAssetForecastData | undefined = result?.per_asset?.[currentAsset]

  // Regime color map (stable across filter changes)
  const regimeColorMap = useMemo(
    () => (result?.regime_labels ? buildRegimeColorMap(result.regime_labels) : {}),
    [result?.regime_labels]
  )

  // Active metric groups: averaged single group for multi-regime, otherwise per-regime or all
  const activeMetricGroups = useMemo(() => {
    if (!assetData) return []
    if (selectedRegimes.length === 0 || !result?.regime_metrics) {
      return [{ label: null as string | null, metrics: assetData.metrics }]
    }
    const assetRM = result.regime_metrics[currentAsset] ?? {}
    const modelNames = Object.keys(assetData.metrics)

    if (selectedRegimes.length === 1) {
      const regime = selectedRegimes[0]
      const metrics = Object.fromEntries(
        modelNames
          .map((model) => [model, assetRM[model]?.[regime]])
          .filter(([, m]) => m != null)
      ) as Record<string, PriceForecastModelMetrics>
      return Object.keys(metrics).length > 0
        ? [{ label: regime, metrics }]
        : []
    }

    // Multiple regimes: average across all selected
    const avgMetrics: Record<string, PriceForecastModelMetrics> = {}
    for (const model of modelNames) {
      const slices = selectedRegimes
        .map((regime) => assetRM[model]?.[regime])
        .filter(Boolean) as PriceForecastModelMetrics[]
      if (slices.length > 0) {
        avgMetrics[model] = averageMetrics(slices)
      }
    }
    const label = `avg(${selectedRegimes.join(', ')})`
    return Object.keys(avgMetrics).length > 0
      ? [{ label, metrics: avgMetrics }]
      : []
  }, [assetData, result, selectedRegimes, currentAsset])

  // Signal series (always aggregate) — filtered by hiddenSignals
  const signalSeries = useMemo(() => {
    if (!assetData) return []
    return Object.entries(assetData.signals)
      .filter(([name]) => !hiddenSignals.has(name))
      .map(([name, data]) => ({
        data,
        color: MODEL_COLORS[name] ?? '#ffffff',
        title: `${name}@${assetData.model_resolutions[name] ?? '?'}`,
      }))
  }, [assetData, hiddenSignals])

  // ROC curves — use active metric groups
  const rocCurves = useMemo(() => {
    if (!assetData) return []
    return activeMetricGroups.flatMap((group) =>
      Object.entries(group.metrics).map(([name, m]) => ({
        name: group.label ? `${name}·${group.label}` : name,
        color: group.label ? regimeColorMap[group.label] ?? (MODEL_COLORS[name] ?? '#fff') : (MODEL_COLORS[name] ?? '#fff'),
        points: m.direction.roc_curve,
        auc: m.direction.auc,
      }))
    )
  }, [activeMetricGroups, assetData, regimeColorMap])

  // Logistic models in active groups
  const logisticModels = useMemo(() => {
    if (!assetData) return []
    return Object.entries(assetData.metrics)
      .filter(([, m]) => m.output_type === 'probability')
      .map(([name]) => name)
  }, [assetData])

  const headerTag = result
    ? `${displayedAssets.join('+')} | ${result.forecast_horizon}${
        result.test_period?.[0] ? ` | Test: ${result.test_period[0]} → ${result.test_period[1]}` : ''
      }`
    : null

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* ── Top-level tab bar ── */}
      <div className="flex gap-1 shrink-0">
        <button
          onClick={() => setTopTab('forecast')}
          className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
            topTab === 'forecast' ? 'bg-sky-700 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          Price Forecast
        </button>
        <button
          onClick={() => setTopTab('er')}
          className={`px-4 py-1.5 rounded text-sm font-medium transition-colors ${
            topTab === 'er' ? 'bg-violet-700 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          Expected Returns
        </button>
      </div>

      {/* ── Forecast tab ── */}
      {topTab === 'forecast' && (
      <div className="flex gap-6 flex-1 min-h-0">
      {/* ── Left panel ── */}
      <div className="w-72 shrink-0 flex flex-col gap-4 overflow-y-auto pr-1">
        <h2 className="text-lg font-semibold text-slate-100">Price Forecast</h2>

        {/* Asset */}
        <div>
          <Label className="mb-2 block text-xs text-slate-400">Asset</Label>
          <div className="flex gap-2">
            {(['BTC', 'ETH', 'Both'] as AssetMode[]).map((a) => (
              <button
                key={a}
                onClick={() => setAssetMode(a)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  assetMode === a ? 'bg-sky-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {a}
              </button>
            ))}
          </div>
          {assetMode === 'Both' && (
            <div className="flex items-center gap-2 mt-2">
              <input
                type="checkbox"
                id="cross-asset"
                checked={crossAsset}
                onChange={(e) => setCrossAsset(e.target.checked)}
                className="accent-sky-500"
              />
              <Label htmlFor="cross-asset" className="text-xs text-slate-300 cursor-pointer">
                Cross-asset signals
              </Label>
            </div>
          )}
        </div>

        {/* Forecast horizon */}
        <div>
          <Label className="mb-2 block text-xs text-slate-400">Forecast Horizon</Label>
          <div className="flex flex-wrap gap-1">
            {HORIZON_OPTIONS.map((h) => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                  horizon === h ? 'bg-sky-600 text-white' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        </div>

        {/* Models */}
        <div>
          <Label className="mb-2 block text-xs text-slate-400">Models</Label>
          <div className="space-y-2">
            {ALL_MODELS.map((m) => (
              <div key={m} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={selectedModels.includes(m)}
                  onChange={() => toggleModel(m)}
                  className="accent-sky-500 shrink-0"
                />
                <span className="inline-block w-2 h-2 rounded-full shrink-0" style={{ background: MODEL_COLORS[m] }} />
                <span className="text-xs text-slate-300 flex-1 truncate">{m}</span>
                <select
                  value={modelResolutions[m] ?? '1D'}
                  onChange={(e) => setRes(m, e.target.value)}
                  disabled={m === 'hp_filter'}
                  className="text-xs bg-slate-800 text-slate-300 border border-slate-700 rounded px-1 py-0.5 disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  {RESOLUTION_OPTIONS.map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>
            ))}
          </div>
        </div>

        <DateSplitPanel
          enabled={dateEnabled}
          onToggle={setDateEnabled}
          trainStart={trainStart}
          trainEnd={trainEnd}
          testStart={testStart}
          testEnd={testEnd}
          dataRange={dataRange}
          onChange={handleDateChange}
          onSetAll={handleSetAllDates}
        />

        <RegimeConditioningPanel
          enabled={regimeEnabled}
          classifierType={regimeType}
          params={regimeParams}
          hasBothAssets={assetMode === 'Both'}
          onChange={(en, type, params) => { setRegimeEnabled(en); setRegimeType(type); setRegimeParams(params) }}
        />

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
        {job?.error && (
          <pre className="text-[10px] text-rose-400 whitespace-pre-wrap break-all bg-rose-950/30 rounded p-2">
            {job.error}
          </pre>
        )}

        {job?.status === 'done' && jobId && <SaveRunDialog jobId={jobId} />}
        <SavedRunsPanel onLoad={(r) => { setSavedResult(r); setSelectedRegimes([]); setActiveAsset((r.assets ?? ['BTC'])[0]) }} />
      </div>

      {/* ── Right panel ── */}
      <div className="flex-1 flex flex-col gap-6 min-w-0 overflow-y-auto">
        {result && assetData ? (
          <>
            {/* Header + asset tabs */}
            <div className="flex items-center justify-between flex-wrap gap-2">
              <span className="text-sm text-slate-400 font-mono">{headerTag}</span>
              <AssetTabs assets={displayedAssets} active={currentAsset} onSelect={setActiveAsset} />
            </div>

            {/* Warnings */}
            {result.warnings && result.warnings.length > 0 && (
              <div className="bg-amber-950 border border-amber-700 rounded-lg px-3 py-2 text-xs text-amber-300 space-y-0.5">
                {result.warnings.map((w, i) => <div key={i}>{w}</div>)}
              </div>
            )}

            {/* Regime timeline + filter (shown whenever regime data is present) */}
            {result.regime_labels && result.regime_labels.length > 0 && (
              <div className="space-y-2 bg-slate-900/50 rounded-lg p-3">
                <RegimeTimelineChart data={result.regime_labels} height={24} />
                <RegimeFilter
                  labels={result.regime_labels}
                  selected={selectedRegimes}
                  colorMap={regimeColorMap}
                  onChange={setSelectedRegimes}
                />
                {selectedRegimes.length > 1 && (
                  <p className="text-[10px] text-slate-500">
                    Multiple regimes selected — metrics averaged across selected regimes.
                  </p>
                )}
              </div>
            )}

            {/* ── Section A: Direction ── */}
            <section>
              <h2 className="text-sm font-semibold text-sky-400 mb-3">A — Direction</h2>
              {activeMetricGroups.map((group) => (
                <div key={group.label ?? '__all__'} className="mb-4">
                  {group.label && (
                    <h3 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: regimeColorMap[group.label] ?? '#94a3b8' }}>
                      <span className="w-2 h-2 rounded-sm inline-block" style={{ background: regimeColorMap[group.label] ?? '#94a3b8' }} />
                      {group.label}
                    </h3>
                  )}
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                    {Object.entries(group.metrics).map(([name, m]) => (
                      <ModelDirectionCard key={name} name={name} m={m} />
                    ))}
                  </div>
                </div>
              ))}
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
              {activeMetricGroups.map((group) => (
                <div key={group.label ?? '__all__'} className="mb-4">
                  {group.label && (
                    <h3 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: regimeColorMap[group.label] ?? '#94a3b8' }}>
                      <span className="w-2 h-2 rounded-sm inline-block" style={{ background: regimeColorMap[group.label] ?? '#94a3b8' }} />
                      {group.label}
                    </h3>
                  )}
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3 mb-3">
                    {Object.entries(group.metrics).map(([name, m]) => (
                      <AmplitudeCard key={name} name={name} m={m} />
                    ))}
                  </div>
                  <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
                    {Object.entries(group.metrics).map(([name, m]) => (
                      <Card key={name}>
                        <CardHeader className="pb-1">
                          <CardTitle className="text-xs flex items-center gap-1.5">
                            <span className="w-2 h-2 rounded-full inline-block" style={{ background: MODEL_COLORS[name] }} />
                            {name} — Calibration
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="pt-0">
                          <CalibrationChart data={m.amplitude.calibration_bins} color={MODEL_COLORS[name] ?? '#fff'} height={160} />
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </div>
              ))}
              {logisticModels.length > 0 && selectedRegimes.length === 0 && (
                <div>
                  <h3 className="text-xs font-medium text-orange-400 mb-2">Probability metrics (logistic)</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {logisticModels.map((name) => {
                      const m = assetData.metrics[name]
                      return (
                        <Card key={name}>
                          <CardHeader className="pb-1">
                            <CardTitle className="text-xs flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full inline-block" style={{ background: MODEL_COLORS[name] }} />
                              {name} — Reliability
                            </CardTitle>
                          </CardHeader>
                          <CardContent className="pt-0 space-y-2">
                            <ReliabilityChart data={m.amplitude.reliability_diagram ?? []} color={MODEL_COLORS[name] ?? '#fff'} height={200} />
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
              {/* Header with per-signal eye toggles */}
              <div className="flex items-center gap-3 mb-3 flex-wrap">
                <h2 className="text-sm font-semibold text-emerald-400">C — Additional Data</h2>
                {Object.keys(assetData.signals).map((name) => (
                  <button
                    key={name}
                    onClick={() => toggleHiddenSignal(name)}
                    className="flex items-center gap-1 text-[10px] text-slate-400 hover:text-slate-200 transition-colors"
                    title={hiddenSignals.has(name) ? `Show ${name}` : `Hide ${name}`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{ background: MODEL_COLORS[name] ?? '#fff', opacity: hiddenSignals.has(name) ? 0.3 : 1 }}
                    />
                    <span className={hiddenSignals.has(name) ? 'opacity-40' : ''}>{name}</span>
                    <span className="text-slate-600">{hiddenSignals.has(name) ? '👁‍🗨' : '👁'}</span>
                  </button>
                ))}
              </div>

              {signalSeries.length > 0 && selectedRegimes.length === 0 && (
                <div className="mb-4">
                  <p className="text-xs text-slate-500 mb-1">Signals (at model resolution)</p>
                  <MultiLineChart series={signalSeries} height={220} />
                </div>
              )}
              {activeMetricGroups.map((group) => (
                <div key={group.label ?? '__all__'} className="mb-4">
                  {group.label && (
                    <h3 className="text-xs font-medium mb-2 flex items-center gap-1.5" style={{ color: regimeColorMap[group.label] ?? '#94a3b8' }}>
                      <span className="w-2 h-2 rounded-sm inline-block" style={{ background: regimeColorMap[group.label] ?? '#94a3b8' }} />
                      {group.label}
                    </h3>
                  )}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {Object.entries(group.metrics).map(([name, m]) => {
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
                              <HistogramChart data={m.additional.signal_histogram} color={MODEL_COLORS[name] ?? '#fff'} height={140} />
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-[9px] text-slate-600 mb-0.5">ACF (±1.96/√n)</p>
                              <AcfChart data={m.additional.acf} color={MODEL_COLORS[name] ?? '#fff'} n_obs={nObs} height={140} />
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                </div>
              ))}
            </section>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
            Configure and submit to compare price forecast models.
          </div>
        )}
      </div>
      </div>
      )}

      {/* ── Expected Returns tab ── */}
      {topTab === 'er' && (
        <div className="flex gap-6 flex-1 min-h-0">
          {/* Left panel */}
          <div className="w-72 shrink-0 flex flex-col gap-4 overflow-y-auto pr-1">
            <h2 className="text-lg font-semibold text-slate-100">Expected Returns Evaluation</h2>

            <div>
              <Label className="mb-2 block">Models</Label>
              {(['rolling_mean', 'signal_tsmom'] as const).map((m) => (
                <label key={m} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input
                    type="checkbox"
                    checked={erModels.includes(m)}
                    onChange={() => toggleErModel(m)}
                    className="accent-violet-500"
                  />
                  <span className="text-sm text-slate-300 flex items-center gap-2">
                    <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: ER_COLORS[m] }} />
                    {m}
                  </span>
                </label>
              ))}
            </div>

            <div>
              <Label className="mb-2 block">Frequency</Label>
              {['1D', '4h', '1h'].map((f) => (
                <label key={f} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input type="radio" name="er-freq" checked={erFreq === f} onChange={() => setErFreq(f)} className="accent-violet-500" />
                  <span className="text-sm text-slate-300">{f}</span>
                </label>
              ))}
            </div>

            <div>
              <Label className="mb-1 block">ER window: {erWindow}</Label>
              <input
                type="range" min={5} max={126} step={1}
                value={erWindow} onChange={(e) => setErWindow(Number(e.target.value))}
                className="w-full accent-violet-500"
              />
            </div>

            <div>
              <Label className="mb-1 block">Min history: {erMinHistory}</Label>
              <input
                type="range" min={10} max={500} step={10}
                value={erMinHistory} onChange={(e) => setErMinHistory(Number(e.target.value))}
                className="w-full accent-violet-500"
              />
            </div>

            <Button
              onClick={handleErSubmit}
              disabled={isErSubmitting || isErRunning || erModels.length === 0}
              className="bg-violet-700 hover:bg-violet-600 text-white"
            >
              Run
            </Button>

            {erJob && (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-slate-400">Status:</span>
                  <StatusBadge status={erJob.status} />
                </div>
                <JobProgressBar status={erJob.status} />
              </div>
            )}
            {erJob?.error && <p className="text-sm text-rose-400">{erJob.error}</p>}
          </div>

          {/* Right panel */}
          <div className="flex-1 flex flex-col gap-6 min-w-0 overflow-y-auto">
            {erResult ? (
              <>
                {/* Asset tabs */}
                <div className="flex gap-1">
                  {(['BTC', 'ETH'] as const).map((a) => (
                    <button
                      key={a}
                      onClick={() => setActiveErAsset(a)}
                      className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                        activeErAsset === a
                          ? 'bg-violet-700 text-white'
                          : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                      }`}
                    >
                      {a}
                    </button>
                  ))}
                </div>

                {erAssetData && (
                  <>
                    <div>
                      <h2 className="text-sm font-medium text-slate-400 mb-2">Expected Returns — {activeErAsset}</h2>
                      <MultiLineChart series={erMuSeries} height={280} />
                    </div>

                    {/* Metrics table */}
                    <div>
                      <h2 className="text-sm font-medium text-slate-400 mb-2">Model Metrics — {activeErAsset}</h2>
                      <div className="overflow-x-auto">
                        <table className="w-full text-sm text-slate-300 border-collapse">
                          <thead>
                            <tr className="border-b border-slate-700 text-slate-500 text-xs">
                              <th className="text-left py-2 pr-4">Model</th>
                              <th className="text-right py-2 px-3">IC</th>
                              <th className="text-right py-2 px-3">Hit Rate</th>
                              <th className="py-2 px-2 w-8"></th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(erAssetData.metrics).map(([name, m]) => (
                              <tr key={name} className="border-b border-slate-800">
                                <td className="py-2 pr-4">
                                  <div className="flex items-center gap-2">
                                    <span
                                      className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                                      style={{ background: ER_COLORS[name] ?? '#fff', opacity: hiddenEr.has(name) ? 0.3 : 1 }}
                                    />
                                    <span className={hiddenEr.has(name) ? 'opacity-40' : ''}>{name}</span>
                                  </div>
                                </td>
                                <td className={`text-right px-3 ${m.ic > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                  {m.ic.toFixed(4)}
                                </td>
                                <td className={`text-right px-3 ${m.hit_rate >= 0.5 ? 'text-emerald-400' : 'text-rose-400'}`}>
                                  {(m.hit_rate * 100).toFixed(1)}%
                                </td>
                                <td className="px-2 text-center">
                                  <EyeButton visible={!hiddenEr.has(name)} onToggle={() => toggleHiddenEr(name)} />
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </>
                )}
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-600 text-sm">
                Configure and run an expected-returns evaluation to see results.
              </div>
            )}
          </div>
        </div>
      )}
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
          <span className={hitOk ? 'text-emerald-400' : 'text-rose-400'}>Hit: {pct(d.hit_rate)}</span>
          <span className="text-slate-400">Prec: {pct(d.precision)}</span>
          <span className="text-slate-400">Rec: {pct(d.recall)}</span>
          <span className="text-slate-400">F1: {pct(d.f1)}</span>
        </div>
        <div className={`text-xs ${aucOk ? 'text-emerald-400' : 'text-rose-400'}`}>AUC: {fmt3(d.auc)}</div>
        {d.confusion && <><ConfusionBar cm={d.confusion} /><ConfusionGrid cm={d.confusion} /></>}
      </CardContent>
    </Card>
  )
}

function AmplitudeCard({ name, m }: { name: string; m: PriceForecastModelMetrics }) {
  const a = m.amplitude
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
          <span className={!isNaN(a.ic) && a.ic > 0 ? 'text-emerald-400' : 'text-rose-400'}>IC: {fmt4(a.ic)}</span>
          <span className={!isNaN(a.rank_ic) && a.rank_ic > 0 ? 'text-emerald-400' : 'text-rose-400'}>Rank IC: {fmt4(a.rank_ic)}</span>
          <span className="text-slate-400">MAE: {fmt4(a.mae)}</span>
          <span className="text-slate-400">MSE: {fmt4(a.mse)}</span>
        </div>
      </CardContent>
    </Card>
  )
}
