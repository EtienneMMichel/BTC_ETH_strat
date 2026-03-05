import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import type { CompareRequest, StrategyConfig } from '@/types/compare'
import type { StrategyName, MarkowitzParams } from '@/types/backtest'

interface CompareConfigFormProps {
  onSubmit: (req: CompareRequest) => void
  isLoading: boolean
}

const STRATEGY_OPTIONS: { value: StrategyName; label: string }[] = [
  { value: 'orchestrator', label: 'Orchestrator' },
  { value: 'momentum', label: 'Momentum' },
  { value: 'mean_reversion', label: 'Mean Reversion' },
  { value: 'markowitz', label: 'Markowitz' },
]

const DEFAULT_MARKOWITZ: MarkowitzParams = {
  objective: 'max_sharpe',
  cov_model: 'diagonal',
  er_model: 'signal_tsmom',
  gamma: 1.0,
  long_only: false,
  max_weight: 1.0,
  min_weight: -1.0,
  risk_free_rate: 0.0,
  target_vol: null,
  min_history: 252,
}

const COMPARE_PALETTE = [
  '#38bdf8', '#a78bfa', '#34d399', '#fbbf24',
  '#f87171', '#e879f9', '#fb923c', '#4ade80',
]

function defaultEntry(idx: number): StrategyConfig {
  const strats: StrategyName[] = ['momentum', 'orchestrator', 'mean_reversion']
  return {
    label: '',
    strategy: strats[idx % strats.length],
  }
}

export function CompareConfigForm({ onSubmit, isLoading }: CompareConfigFormProps) {
  const [freq, setFreq] = useState('1D')
  const [minTrainBars, setMinTrainBars] = useState(252)
  const [rebalanceEvery, setRebalanceEvery] = useState(21)
  const [feeRate, setFeeRate] = useState(0.001)
  const [slippage, setSlippage] = useState(0.0005)

  const [strategies, setStrategies] = useState<StrategyConfig[]>([
    { label: 'Momentum', strategy: 'momentum' },
    { label: 'Orchestrator', strategy: 'orchestrator' },
  ])

  // Frontier options
  const [includeFrontier, setIncludeFrontier] = useState(true)
  const [frontierPoints, setFrontierPoints] = useState(20)
  const [frontierCov, setFrontierCov] = useState<'rolling' | 'diagonal' | 'bekk'>('diagonal')
  const [frontierER, setFrontierER] = useState<'rolling_mean' | 'signal_tsmom'>('signal_tsmom')
  const [frontierLongOnly, setFrontierLongOnly] = useState(false)

  const updateStrategy = (idx: number, patch: Partial<StrategyConfig>) => {
    setStrategies((prev) => prev.map((s, i) => (i === idx ? { ...s, ...patch } : s)))
  }

  const removeStrategy = (idx: number) => {
    setStrategies((prev) => prev.filter((_, i) => i !== idx))
  }

  const addStrategy = () => {
    if (strategies.length >= 10) return
    setStrategies((prev) => [...prev, defaultEntry(prev.length)])
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Auto-label unnamed strategies
    const labeled = strategies.map((s, i) => ({
      ...s,
      label: s.label || `${s.strategy}_${i}`,
      markowitz: s.strategy === 'markowitz' ? (s.markowitz ?? DEFAULT_MARKOWITZ) : undefined,
    }))
    onSubmit({
      freq,
      min_train_bars: minTrainBars,
      rebalance_every: rebalanceEvery,
      fee_rate: feeRate,
      slippage,
      strategies: labeled,
      include_frontier: includeFrontier,
      frontier_points: frontierPoints,
      frontier_cov_model: frontierCov,
      frontier_er_model: frontierER,
      frontier_long_only: frontierLongOnly,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      {/* Shared params */}
      <div>
        <Label className="mb-2 block">Frequency</Label>
        <div className="flex gap-3">
          {['1D', '4h', '1h'].map((f) => (
            <label key={f} className="flex items-center gap-1.5 cursor-pointer">
              <input type="radio" name="cmp-freq" checked={freq === f} onChange={() => setFreq(f)} className="accent-sky-500" />
              <span className="text-sm text-slate-300">{f}</span>
            </label>
          ))}
        </div>
      </div>

      <div>
        <div className="flex justify-between mb-1">
          <Label>Min Train Bars</Label>
          <span className="text-sm text-slate-400">{minTrainBars}</span>
        </div>
        <Slider value={minTrainBars} onChange={setMinTrainBars} min={50} max={500} step={10} />
      </div>

      <div>
        <div className="flex justify-between mb-1">
          <Label>Rebalance Every</Label>
          <span className="text-sm text-slate-400">{rebalanceEvery}</span>
        </div>
        <Slider value={rebalanceEvery} onChange={setRebalanceEvery} min={1} max={63} />
      </div>

      <div>
        <div className="flex justify-between mb-1">
          <Label>Fee Rate</Label>
          <span className="text-sm text-slate-400">{(feeRate * 100).toFixed(3)}%</span>
        </div>
        <Slider value={Math.round(feeRate * 10000)} onChange={(v) => setFeeRate(v / 10000)} min={0} max={100} step={1} />
      </div>

      <div>
        <div className="flex justify-between mb-1">
          <Label>Slippage</Label>
          <span className="text-sm text-slate-400">{(slippage * 100).toFixed(3)}%</span>
        </div>
        <Slider value={Math.round(slippage * 10000)} onChange={(v) => setSlippage(v / 10000)} min={0} max={100} step={1} />
      </div>

      {/* Strategy list */}
      <div>
        <Label className="mb-2 block">Strategies</Label>
        <div className="flex flex-col gap-3">
          {strategies.map((s, i) => (
            <div key={i} className="border border-slate-700 rounded-lg p-3 flex flex-col gap-2">
              <div className="flex items-center gap-2">
                <span
                  className="inline-block w-3 h-3 rounded-full shrink-0"
                  style={{ background: COMPARE_PALETTE[i % COMPARE_PALETTE.length] }}
                />
                <input
                  type="text"
                  placeholder={`${s.strategy}_${i}`}
                  value={s.label}
                  onChange={(e) => updateStrategy(i, { label: e.target.value })}
                  className="flex-1 bg-slate-800 text-slate-200 rounded px-2 py-1 text-sm border border-slate-700 focus:outline-none focus:border-sky-500"
                />
                {strategies.length > 1 && (
                  <button
                    type="button"
                    onClick={() => removeStrategy(i)}
                    className="text-slate-500 hover:text-rose-400 text-lg leading-none px-1"
                    title="Remove"
                  >
                    x
                  </button>
                )}
              </div>
              <select
                value={s.strategy}
                onChange={(e) => updateStrategy(i, { strategy: e.target.value as StrategyName })}
                className="bg-slate-800 text-slate-200 rounded px-2 py-1 text-sm border border-slate-700 focus:outline-none focus:border-sky-500"
              >
                {STRATEGY_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>

              {/* Markowitz sub-params */}
              {s.strategy === 'markowitz' && (
                <div className="flex flex-col gap-2 ml-2 border-l border-slate-700 pl-2">
                  <div>
                    <Label className="text-xs mb-1 block">Objective</Label>
                    <select
                      value={s.markowitz?.objective ?? 'max_sharpe'}
                      onChange={(e) =>
                        updateStrategy(i, {
                          markowitz: { ...(s.markowitz ?? DEFAULT_MARKOWITZ), objective: e.target.value as MarkowitzParams['objective'] },
                        })
                      }
                      className="bg-slate-800 text-slate-200 rounded px-2 py-1 text-xs border border-slate-700 w-full"
                    >
                      <option value="max_sharpe">Max Sharpe</option>
                      <option value="min_variance">Min Variance</option>
                      <option value="mean_variance">Mean-Variance</option>
                      <option value="max_diversification">Max Diversification</option>
                    </select>
                  </div>
                  <div>
                    <Label className="text-xs mb-1 block">Cov Model</Label>
                    <select
                      value={s.markowitz?.cov_model ?? 'diagonal'}
                      onChange={(e) =>
                        updateStrategy(i, {
                          markowitz: { ...(s.markowitz ?? DEFAULT_MARKOWITZ), cov_model: e.target.value as MarkowitzParams['cov_model'] },
                        })
                      }
                      className="bg-slate-800 text-slate-200 rounded px-2 py-1 text-xs border border-slate-700 w-full"
                    >
                      <option value="rolling">Rolling</option>
                      <option value="diagonal">Diagonal (DCC)</option>
                      <option value="bekk">BEKK</option>
                    </select>
                  </div>
                  <div>
                    <Label className="text-xs mb-1 block">ER Model</Label>
                    <select
                      value={s.markowitz?.er_model ?? 'signal_tsmom'}
                      onChange={(e) =>
                        updateStrategy(i, {
                          markowitz: { ...(s.markowitz ?? DEFAULT_MARKOWITZ), er_model: e.target.value as MarkowitzParams['er_model'] },
                        })
                      }
                      className="bg-slate-800 text-slate-200 rounded px-2 py-1 text-xs border border-slate-700 w-full"
                    >
                      <option value="rolling_mean">Rolling Mean</option>
                      <option value="signal_tsmom">Signal TSMOM</option>
                    </select>
                  </div>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={s.markowitz?.long_only ?? false}
                      onChange={(e) =>
                        updateStrategy(i, {
                          markowitz: { ...(s.markowitz ?? DEFAULT_MARKOWITZ), long_only: e.target.checked },
                        })
                      }
                      className="accent-sky-500"
                    />
                    <span className="text-xs text-slate-300">Long-only</span>
                  </label>
                </div>
              )}
            </div>
          ))}

          {strategies.length < 10 && (
            <button
              type="button"
              onClick={addStrategy}
              className="text-sm text-sky-400 hover:text-sky-300 text-left"
            >
              + Add Strategy
            </button>
          )}
        </div>
      </div>

      {/* Frontier options */}
      <details className="border border-slate-700 rounded-lg overflow-hidden">
        <summary className="px-3 py-2 bg-slate-800 text-sm font-medium text-slate-200 cursor-pointer select-none">
          Efficient Frontier
        </summary>
        <div className="flex flex-col gap-3 p-3">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={includeFrontier}
              onChange={(e) => setIncludeFrontier(e.target.checked)}
              className="accent-sky-500"
            />
            <span className="text-sm text-slate-300">Compute frontier</span>
          </label>

          {includeFrontier && (
            <>
              <div>
                <div className="flex justify-between mb-1">
                  <Label className="text-xs">Points</Label>
                  <span className="text-xs text-slate-400">{frontierPoints}</span>
                </div>
                <Slider value={frontierPoints} onChange={setFrontierPoints} min={5} max={50} />
              </div>
              <div>
                <Label className="text-xs mb-1 block">Cov Model</Label>
                <select
                  value={frontierCov}
                  onChange={(e) => setFrontierCov(e.target.value as typeof frontierCov)}
                  className="bg-slate-800 text-slate-200 rounded px-2 py-1 text-xs border border-slate-700 w-full"
                >
                  <option value="rolling">Rolling</option>
                  <option value="diagonal">Diagonal (DCC)</option>
                  <option value="bekk">BEKK</option>
                </select>
              </div>
              <div>
                <Label className="text-xs mb-1 block">ER Model</Label>
                <select
                  value={frontierER}
                  onChange={(e) => setFrontierER(e.target.value as typeof frontierER)}
                  className="bg-slate-800 text-slate-200 rounded px-2 py-1 text-xs border border-slate-700 w-full"
                >
                  <option value="rolling_mean">Rolling Mean</option>
                  <option value="signal_tsmom">Signal TSMOM</option>
                </select>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={frontierLongOnly}
                  onChange={(e) => setFrontierLongOnly(e.target.checked)}
                  className="accent-sky-500"
                />
                <span className="text-xs text-slate-300">Long-only</span>
              </label>
            </>
          )}
        </div>
      </details>

      <Button type="submit" disabled={isLoading} size="lg" className="mt-2">
        {isLoading ? 'Running...' : 'Run Comparison'}
      </Button>
    </form>
  )
}

export { COMPARE_PALETTE }
