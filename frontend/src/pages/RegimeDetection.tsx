import { useCallback, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  submitRegimeDetection,
  getRegimeDetectionJob,
  listSavedModels,
  deleteSavedModel,
} from '@/lib/api/regime_detection'
import { PriceVolumeChart } from '@/components/charts/PriceVolumeChart'
import { RegimeRow } from '@/components/charts/RegimeRow'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { StatusBadge } from '@/components/backtest/StatusBadge'
import { JobProgressBar } from '@/components/ui/progress'
import type { ModelConfig, ModelType, RegimeDetectionResult } from '@/types/regime_detection'

// ---------------------------------------------------------------------------
// Default model configs
// ---------------------------------------------------------------------------

const DEFAULT_MODELS: ModelConfig[] = [
  { model_id: 'threshold_default', model_type: 'threshold', params: {} },
  { model_id: 'hmm_2', model_type: 'hmm', params: { n_components: 2 } },
]

const MODEL_TYPE_OPTIONS: { value: ModelType; label: string }[] = [
  { value: 'threshold', label: 'Threshold' },
  { value: 'vol_quantile', label: 'Vol Quantile' },
  { value: 'hmm', label: 'HMM' },
]

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function RegimeDetectionPage() {
  // Config state
  const [freq, setFreq] = useState('1D')
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [models, setModels] = useState<ModelConfig[]>(DEFAULT_MODELS)
  const [probWindow, setProbWindow] = useState(20)
  const [saveModels, setSaveModels] = useState(false)

  // UI state
  const [mode, setMode] = useState<'max' | 'prob'>('max')
  const [displayMode, setDisplayMode] = useState<'price' | 'return'>('return')
  const [chartMargins, setChartMargins] = useState({ left: 0, right: 0 })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [savedModelsOpen, setSavedModelsOpen] = useState(false)

  const queryClient = useQueryClient()

  // Polling
  const { data: job } = useQuery({
    queryKey: ['regime-detection', jobId],
    queryFn: () => getRegimeDetectionJob(jobId!),
    enabled: !!jobId,
    refetchInterval: (q) =>
      q.state.data?.status === 'done' || q.state.data?.status === 'failed' ? false : 2000,
  })

  // Saved models list
  const { data: savedModelsList } = useQuery({
    queryKey: ['regime-detection-saved-models'],
    queryFn: listSavedModels,
    enabled: savedModelsOpen,
  })

  const deleteMutation = useMutation({
    mutationFn: (name: string) => deleteSavedModel(name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['regime-detection-saved-models'] }),
  })

  const result: RegimeDetectionResult | undefined = job?.result

  const handleSubmit = async () => {
    if (isSubmitting) return
    setIsSubmitting(true)
    try {
      const { job_id } = await submitRegimeDetection({
        assets: ['BTC', 'ETH'],
        freq,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        models,
        prob_window: probWindow,
        save_models: saveModels,
      })
      setJobId(job_id)
      if (saveModels) {
        queryClient.invalidateQueries({ queryKey: ['regime-detection-saved-models'] })
      }
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleChartMargins = useCallback((left: number, right: number) => {
    setChartMargins({ left, right })
  }, [])

  // Model list editing helpers
  const addModel = () => {
    setModels((prev) => [
      ...prev,
      { model_id: `model_${prev.length + 1}`, model_type: 'hmm', params: {} },
    ])
  }

  const removeModel = (idx: number) => {
    setModels((prev) => prev.filter((_, i) => i !== idx))
  }

  const updateModel = (idx: number, patch: Partial<ModelConfig>) => {
    setModels((prev) => prev.map((m, i) => (i === idx ? { ...m, ...patch } : m)))
  }

  const updateModelParam = (idx: number, key: string, rawValue: string) => {
    const value = rawValue === '' ? undefined : isNaN(Number(rawValue)) ? rawValue : Number(rawValue)
    setModels((prev) =>
      prev.map((m, i) =>
        i === idx
          ? { ...m, params: value === undefined ? { ...m.params } : { ...m.params, [key]: value } }
          : m
      )
    )
  }

  // Default param fields per model type
  const defaultParamFields: Record<ModelType, { key: string; label: string; placeholder: string }[]> = {
    threshold: [
      { key: 'vol_window', label: 'Vol window', placeholder: '63' },
      { key: 'vol_threshold_pct', label: 'Vol threshold pct', placeholder: '0.6' },
      { key: 'min_holding_bars', label: 'Min holding bars', placeholder: '5' },
    ],
    vol_quantile: [
      { key: 'n_regimes', label: 'N regimes', placeholder: '2' },
      { key: 'vol_window', label: 'Vol window', placeholder: '21' },
    ],
    hmm: [
      { key: 'n_components', label: 'N states', placeholder: '2' },
      { key: 'n_iter', label: 'N iter', placeholder: '100' },
    ],
  }

  const [expandedModel, setExpandedModel] = useState<number | null>(null)

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Left panel ── */}
      <aside className="w-80 shrink-0 border-r border-slate-800 bg-slate-950 overflow-y-auto flex flex-col gap-4 p-4">
        <div className="text-sm font-semibold text-slate-300">Regime Detection</div>

        {/* Frequency */}
        <div>
          <Label className="text-xs text-slate-400 mb-1 block">Frequency</Label>
          <div className="flex gap-1">
            {['1D', '4h', '1h'].map((f) => (
              <button
                key={f}
                onClick={() => setFreq(f)}
                className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                  freq === f
                    ? 'bg-sky-700 text-white'
                    : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {/* Date range */}
        <div>
          <Label className="text-xs text-slate-400 mb-1 block">Date range (optional)</Label>
          <div className="space-y-1">
            <input
              type="text"
              placeholder="Start YYYY-MM-DD"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-sky-600"
            />
            <input
              type="text"
              placeholder="End YYYY-MM-DD"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-sky-600"
            />
          </div>
        </div>

        {/* Models list */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <Label className="text-xs text-slate-400">Models</Label>
            <button
              onClick={addModel}
              className="text-[10px] px-2 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-300"
            >
              + Add model
            </button>
          </div>

          <div className="space-y-2">
            {models.map((m, idx) => (
              <div key={idx} className="border border-slate-700 rounded p-2 space-y-1.5">
                <div className="flex gap-1 items-center">
                  <input
                    type="text"
                    value={m.model_id}
                    onChange={(e) => updateModel(idx, { model_id: e.target.value })}
                    className="flex-1 min-w-0 bg-slate-800 border border-slate-700 rounded px-2 py-0.5 text-xs text-slate-200 focus:outline-none focus:border-sky-600"
                    placeholder="model_id"
                  />
                  <button
                    onClick={() => setExpandedModel(expandedModel === idx ? null : idx)}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 hover:bg-slate-600 text-slate-400"
                    title="Edit params"
                  >
                    ⚙
                  </button>
                  <button
                    onClick={() => removeModel(idx)}
                    className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 hover:bg-red-900 text-slate-500 hover:text-red-300"
                  >
                    ×
                  </button>
                </div>

                <select
                  value={m.model_type}
                  onChange={(e) =>
                    updateModel(idx, { model_type: e.target.value as ModelType, params: {} })
                  }
                  className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-0.5 text-xs text-slate-300 focus:outline-none focus:border-sky-600"
                >
                  {MODEL_TYPE_OPTIONS.map(({ value, label }) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>

                {expandedModel === idx && (
                  <div className="pt-1 space-y-1">
                    {defaultParamFields[m.model_type].map(({ key, label, placeholder }) => (
                      <div key={key} className="flex items-center gap-2">
                        <span className="text-[10px] text-slate-500 w-28 shrink-0">{label}</span>
                        <input
                          type="text"
                          placeholder={placeholder}
                          value={(m.params[key] as string | number | undefined) ?? ''}
                          onChange={(e) => updateModelParam(idx, key, e.target.value)}
                          className="flex-1 bg-slate-800 border border-slate-700 rounded px-1.5 py-0.5 text-xs text-slate-200 focus:outline-none focus:border-sky-600"
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Prob window (shown always, relevant in prob mode) */}
        <div>
          <Label className="text-xs text-slate-400 mb-1 block">
            Prob window <span className="text-slate-500">(rolling, for threshold/vol_quantile)</span>
          </Label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={5}
              max={63}
              value={probWindow}
              onChange={(e) => setProbWindow(Number(e.target.value))}
              className="flex-1"
            />
            <span className="text-xs text-slate-300 w-8 text-right">{probWindow}</span>
          </div>
        </div>

        {/* Save models checkbox */}
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="save-models"
            checked={saveModels}
            onChange={(e) => setSaveModels(e.target.checked)}
            className="accent-sky-500"
          />
          <Label htmlFor="save-models" className="text-xs text-slate-300 cursor-pointer">
            Save fitted models to disk
          </Label>
        </div>

        {/* Run button */}
        <Button
          onClick={handleSubmit}
          disabled={isSubmitting || models.length === 0}
          className="w-full"
        >
          {isSubmitting ? 'Submitting…' : 'Run'}
        </Button>

        {/* Job status */}
        {job && (
          <div className="space-y-1">
            <StatusBadge status={job.status} />
            {(job.status === 'pending' || job.status === 'running') && <JobProgressBar status={job.status} />}
            {job.status === 'failed' && (
              <p className="text-xs text-red-400 break-all">{job.error}</p>
            )}
          </div>
        )}

        {/* Saved Models section */}
        <div className="border border-slate-700 rounded">
          <button
            onClick={() => setSavedModelsOpen((o) => !o)}
            className="w-full flex items-center justify-between px-3 py-2 text-xs text-slate-400 hover:text-slate-200"
          >
            <span className="font-medium">Saved Models</span>
            <span>{savedModelsOpen ? '▲' : '▼'}</span>
          </button>
          {savedModelsOpen && (
            <div className="border-t border-slate-700 p-2 space-y-1 max-h-48 overflow-y-auto">
              {!savedModelsList || savedModelsList.length === 0 ? (
                <p className="text-[10px] text-slate-600 text-center py-2">No saved models</p>
              ) : (
                savedModelsList.map((sm) => (
                  <div
                    key={sm.name}
                    className="flex items-center justify-between gap-1 text-[10px] text-slate-400 py-0.5"
                  >
                    <div className="min-w-0">
                      <span className="text-slate-300 font-medium truncate block">{sm.name}</span>
                      <span className="text-slate-600">
                        {sm.model_type} · {sm.saved_at.slice(0, 10)}
                      </span>
                    </div>
                    <button
                      onClick={() => deleteMutation.mutate(sm.name)}
                      className="shrink-0 text-[10px] px-1.5 py-0.5 rounded hover:bg-red-900 text-slate-500 hover:text-red-300"
                      title="Delete"
                    >
                      ×
                    </button>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </aside>

      {/* ── Right panel ── */}
      <main className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Mode toggle */}
        <div className="flex items-center justify-between">
          <div className="text-sm font-semibold text-slate-300">
            {result ? `${result.freq} · ${result.date_range[0]?.slice(0, 10)} – ${result.date_range[1]?.slice(0, 10)}` : 'Regime Timeline'}
          </div>
          <div className="flex gap-1 items-center">
            {/* Price / Return toggle */}
            <button
              onClick={() => setDisplayMode(displayMode === 'price' ? 'return' : 'price')}
              className="px-2 py-1 text-xs rounded transition-colors bg-slate-800 text-slate-400 hover:bg-slate-700 mr-2"
              title="Toggle between raw price and cumulated returns"
            >
              {displayMode === 'return' ? 'Return %' : 'Price'}
            </button>

            {mode === 'prob' && (
              <span className="text-[10px] text-slate-500 mr-2">prob window: {probWindow}</span>
            )}
            <button
              onClick={() => setMode('max')}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                mode === 'max'
                  ? 'bg-sky-700 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              Max
            </button>
            <button
              onClick={() => setMode('prob')}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                mode === 'prob'
                  ? 'bg-sky-700 text-white'
                  : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              Prob
            </button>
          </div>
        </div>

        {/* Price + Volume chart */}
        {result ? (
          <>
            <PriceVolumeChart
              btcClose={result.prices['BTC'] ?? []}
              ethClose={result.prices['ETH'] ?? []}
              btcVolume={result.volumes['BTC'] ?? []}
              ethVolume={result.volumes['ETH'] ?? []}
              displayMode={displayMode}
              onChartMargins={handleChartMargins}
            />

            {/* Regime strips */}
            <div className="space-y-3">
              {result.models.map((m, idx) => (
                <div
                  key={m.model_id}
                  className="bg-slate-900 rounded-lg p-3 border border-slate-800"
                >
                  <RegimeRow
                    modelId={m.model_id}
                    modelType={m.model_type}
                    labels={m.labels}
                    probabilities={m.probabilities}
                    uniqueRegimes={m.unique_regimes}
                    dateRange={result.date_range}
                    mode={mode}
                    showTimeAxis={idx === result.models.length - 1}
                    paddingLeft={chartMargins.left}
                    paddingRight={chartMargins.right}
                  />
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="flex items-center justify-center h-64 text-slate-500 text-sm">
            Configure models in the left panel and click Run.
          </div>
        )}
      </main>
    </div>
  )
}
