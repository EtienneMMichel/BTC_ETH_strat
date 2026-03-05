import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { COV_COLORS, ER_COLORS } from '@/lib/constants/markowitz-colors'
import type { BacktestRequest, StrategyName, MarkowitzParams } from '@/types/backtest'

interface ConfigFormProps {
  onSubmit: (req: BacktestRequest) => void
  isLoading: boolean
}

const STRATEGIES: { value: StrategyName; label: string }[] = [
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

export function ConfigForm({ onSubmit, isLoading }: ConfigFormProps) {
  const [strategy, setStrategy] = useState<StrategyName>('orchestrator')
  const [minTrainBars, setMinTrainBars] = useState(252)
  const [rebalanceEvery, setRebalanceEvery] = useState(21)
  const [feeRate, setFeeRate] = useState(0.001)
  const [slippage, setSlippage] = useState(0.0005)

  // Markowitz-specific state
  const [freq, setFreq] = useState('1D')
  const [mwParams, setMwParams] = useState<MarkowitzParams>(DEFAULT_MARKOWITZ)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const base: BacktestRequest = {
      strategy,
      min_train_bars: minTrainBars,
      rebalance_every: rebalanceEvery,
      fee_rate: feeRate,
      slippage,
    }
    if (strategy === 'markowitz') {
      base.freq = freq
      base.markowitz = mwParams
    }
    onSubmit(base)
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      <div>
        <Label className="mb-2 block">Strategy</Label>
        <div className="flex flex-col gap-2">
          {STRATEGIES.map((s) => (
            <label key={s.value} className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="strategy"
                value={s.value}
                checked={strategy === s.value}
                onChange={() => setStrategy(s.value)}
                className="accent-sky-500"
              />
              <span className="text-sm text-slate-300">{s.label}</span>
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
          <Label>Rebalance Every (bars)</Label>
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

      {/* ── Markowitz collapsible ── */}
      {strategy === 'markowitz' && (
        <details open className="border border-slate-700 rounded-lg overflow-hidden">
          <summary className="px-3 py-2 bg-slate-800 text-sm font-medium text-slate-200 cursor-pointer select-none">
            Markowitz Parameters
          </summary>
          <div className="flex flex-col gap-4 p-3">
            {/* Frequency */}
            <div>
              <Label className="mb-2 block">Frequency</Label>
              {['1D', '4h', '1h'].map((f) => (
                <label key={f} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input type="radio" name="mw-freq" checked={freq === f} onChange={() => setFreq(f)} className="accent-violet-500" />
                  <span className="text-sm text-slate-300">{f}</span>
                </label>
              ))}
            </div>

            {/* Objective */}
            <div>
              <Label className="mb-2 block">Objective</Label>
              {(['max_sharpe', 'min_variance', 'mean_variance', 'max_diversification'] as const).map((o) => (
                <label key={o} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input
                    type="radio" name="mw-obj"
                    checked={mwParams.objective === o}
                    onChange={() => setMwParams((p) => ({ ...p, objective: o }))}
                    className="accent-violet-500"
                  />
                  <span className="text-sm text-slate-300">{o}</span>
                </label>
              ))}
            </div>

            {/* Covariance model */}
            <div>
              <Label className="mb-2 block">Covariance model</Label>
              {(['rolling', 'diagonal', 'bekk'] as const).map((c) => (
                <label key={c} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input
                    type="radio" name="mw-cov"
                    checked={mwParams.cov_model === c}
                    onChange={() => setMwParams((p) => ({ ...p, cov_model: c }))}
                    className="accent-violet-500"
                  />
                  <span className="text-sm text-slate-300 flex items-center gap-2">
                    <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: COV_COLORS[c] }} />
                    {c}
                  </span>
                </label>
              ))}
            </div>

            {/* Expected returns model */}
            <div>
              <Label className="mb-2 block">Expected returns model</Label>
              {(['rolling_mean', 'signal_tsmom'] as const).map((er) => (
                <label key={er} className="flex items-center gap-2 cursor-pointer mb-1">
                  <input
                    type="radio" name="mw-er"
                    checked={mwParams.er_model === er}
                    onChange={() => setMwParams((p) => ({ ...p, er_model: er }))}
                    className="accent-violet-500"
                  />
                  <span className="text-sm text-slate-300 flex items-center gap-2">
                    <span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: ER_COLORS[er] }} />
                    {er}
                  </span>
                </label>
              ))}
            </div>

            {/* Gamma */}
            <div>
              <div className="flex justify-between mb-1">
                <Label>Gamma</Label>
                <span className="text-sm text-slate-400">{mwParams.gamma.toFixed(2)}</span>
              </div>
              <Slider
                value={Math.round(mwParams.gamma * 100)}
                onChange={(v) => setMwParams((p) => ({ ...p, gamma: v / 100 }))}
                min={1}
                max={1000}
                step={1}
                disabled={mwParams.objective !== 'mean_variance'}
              />
            </div>

            {/* Long-only */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={mwParams.long_only}
                onChange={(e) => setMwParams((p) => ({ ...p, long_only: e.target.checked }))}
                className="accent-violet-500"
              />
              <span className="text-sm text-slate-300">Long-only</span>
            </label>

            {/* Target vol */}
            <div>
              <Label className="mb-1 block">Target vol (annualised, blank = none)</Label>
              <input
                type="number" min={0} max={5} step={0.05}
                placeholder="e.g. 0.20"
                value={mwParams.target_vol ?? ''}
                onChange={(e) =>
                  setMwParams((p) => ({
                    ...p,
                    target_vol: e.target.value === '' ? null : Number(e.target.value),
                  }))
                }
                className="w-full bg-slate-800 text-slate-200 rounded px-2 py-1 text-sm border border-slate-700 focus:outline-none focus:border-violet-500"
              />
            </div>

            {/* Min history */}
            <div>
              <div className="flex justify-between mb-1">
                <Label>Min History</Label>
                <span className="text-sm text-slate-400">{mwParams.min_history}</span>
              </div>
              <Slider
                value={mwParams.min_history}
                onChange={(v) => setMwParams((p) => ({ ...p, min_history: v }))}
                min={10}
                max={500}
                step={10}
              />
            </div>
          </div>
        </details>
      )}

      <Button type="submit" disabled={isLoading} size="lg" className="mt-2">
        {isLoading ? 'Running\u2026' : 'Run Backtest'}
      </Button>
    </form>
  )
}
