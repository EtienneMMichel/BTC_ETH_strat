import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import type { BacktestRequest, StrategyName } from '@/types/backtest'

interface ConfigFormProps {
  onSubmit: (req: BacktestRequest) => void
  isLoading: boolean
}

const STRATEGIES: { value: StrategyName; label: string }[] = [
  { value: 'orchestrator', label: 'Orchestrator' },
  { value: 'momentum', label: 'Momentum' },
  { value: 'mean_reversion', label: 'Mean Reversion' },
]

export function ConfigForm({ onSubmit, isLoading }: ConfigFormProps) {
  const [strategy, setStrategy] = useState<StrategyName>('orchestrator')
  const [minTrainBars, setMinTrainBars] = useState(252)
  const [rebalanceEvery, setRebalanceEvery] = useState(21)
  const [feeRate, setFeeRate] = useState(0.001)
  const [slippage, setSlippage] = useState(0.0005)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    onSubmit({ strategy, min_train_bars: minTrainBars, rebalance_every: rebalanceEvery, fee_rate: feeRate, slippage })
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

      <Button type="submit" disabled={isLoading} size="lg" className="mt-2">
        {isLoading ? 'Running…' : 'Run Backtest'}
      </Button>
    </form>
  )
}
