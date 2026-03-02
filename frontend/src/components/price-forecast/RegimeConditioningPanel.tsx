import { useState } from 'react'
import { Label } from '@/components/ui/label'
import type { RegimeClassifierType, ManualRegimeDateRange } from '@/types/models'

interface RegimeConditioningPanelProps {
  enabled: boolean
  classifierType: RegimeClassifierType
  params: Record<string, unknown>
  hasBothAssets: boolean
  onChange: (
    enabled: boolean,
    classifierType: RegimeClassifierType,
    params: Record<string, unknown>
  ) => void
}

export function RegimeConditioningPanel({
  enabled,
  classifierType,
  params,
  hasBothAssets,
  onChange,
}: RegimeConditioningPanelProps) {
  const [manualRanges, setManualRanges] = useState<ManualRegimeDateRange[]>(
    (params.date_ranges as ManualRegimeDateRange[] | undefined) ?? []
  )

  const update = (
    newEnabled: boolean,
    newType: RegimeClassifierType,
    newParams: Record<string, unknown>
  ) => onChange(newEnabled, newType, newParams)

  const updateParams = (patch: Record<string, unknown>) =>
    update(enabled, classifierType, { ...params, ...patch })

  const addRange = () => {
    const updated = [...manualRanges, { label: `regime_${manualRanges.length}`, start: '', end: '' }]
    setManualRanges(updated)
    updateParams({ date_ranges: updated })
  }

  const updateRange = (i: number, field: keyof ManualRegimeDateRange, value: string) => {
    const updated = manualRanges.map((r, idx) => (idx === i ? { ...r, [field]: value } : r))
    setManualRanges(updated)
    updateParams({ date_ranges: updated })
  }

  const removeRange = (i: number) => {
    const updated = manualRanges.filter((_, idx) => idx !== i)
    setManualRanges(updated)
    updateParams({ date_ranges: updated })
  }

  return (
    <details>
      <summary className="text-xs text-slate-400 cursor-pointer select-none hover:text-slate-200 transition-colors">
        Regime conditioning
      </summary>
      <div className="mt-2 space-y-2 pl-1">
        <div className="flex items-center gap-2">
          <input
            type="checkbox"
            id="regime-enable"
            checked={enabled}
            onChange={(e) => update(e.target.checked, classifierType, params)}
            className="accent-sky-500"
          />
          <Label htmlFor="regime-enable" className="text-xs text-slate-300 cursor-pointer">
            Enable
          </Label>
        </div>

        {enabled && (
          <>
            <div>
              <Label className="text-[10px] text-slate-400">Classifier</Label>
              <select
                value={classifierType}
                onChange={(e) =>
                  update(enabled, e.target.value as RegimeClassifierType, {})
                }
                className="w-full text-xs bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300 mt-0.5"
              >
                <option value="vol_quantile">Vol Quantile</option>
                <option value="threshold">Threshold (BTC+ETH)</option>
                <option value="manual">Manual date ranges</option>
              </select>
            </div>

            {classifierType === 'threshold' && !hasBothAssets && (
              <p className="text-[10px] text-amber-400">
                Threshold classifier requires Both assets to be selected.
              </p>
            )}

            {classifierType === 'vol_quantile' && (
              <div className="space-y-1">
                <div>
                  <Label className="text-[10px] text-slate-400">
                    Regimes: {(params.n_regimes as number | undefined) ?? 2}
                  </Label>
                  <input
                    type="range"
                    min={2}
                    max={4}
                    step={1}
                    value={(params.n_regimes as number | undefined) ?? 2}
                    onChange={(e) => updateParams({ n_regimes: Number(e.target.value) })}
                    className="w-full accent-sky-500"
                  />
                </div>
                <div>
                  <Label className="text-[10px] text-slate-400">
                    Vol window: {(params.vol_window as number | undefined) ?? 21}
                  </Label>
                  <input
                    type="range"
                    min={5}
                    max={63}
                    step={1}
                    value={(params.vol_window as number | undefined) ?? 21}
                    onChange={(e) => updateParams({ vol_window: Number(e.target.value) })}
                    className="w-full accent-sky-500"
                  />
                </div>
              </div>
            )}

            {classifierType === 'threshold' && (
              <div className="space-y-1 text-xs">
                <div className="flex gap-2 items-center">
                  <Label className="text-[10px] text-slate-400 w-32">Vol threshold pct</Label>
                  <input
                    type="number"
                    step={0.05}
                    min={0}
                    max={1}
                    value={(params.vol_threshold_pct as number | undefined) ?? 0.6}
                    onChange={(e) => updateParams({ vol_threshold_pct: Number(e.target.value) })}
                    className="w-20 bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
                  />
                </div>
                <div className="flex gap-2 items-center">
                  <Label className="text-[10px] text-slate-400 w-32">Drawdown threshold</Label>
                  <input
                    type="number"
                    step={0.05}
                    min={-1}
                    max={0}
                    value={(params.drawdown_threshold as number | undefined) ?? -0.2}
                    onChange={(e) => updateParams({ drawdown_threshold: Number(e.target.value) })}
                    className="w-20 bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
                  />
                </div>
                <div className="flex gap-2 items-center">
                  <Label className="text-[10px] text-slate-400 w-32">ADF p-value</Label>
                  <input
                    type="number"
                    step={0.01}
                    min={0}
                    max={0.5}
                    value={(params.adf_pvalue as number | undefined) ?? 0.05}
                    onChange={(e) => updateParams({ adf_pvalue: Number(e.target.value) })}
                    className="w-20 bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
                  />
                </div>
              </div>
            )}

            {classifierType === 'manual' && (
              <div className="space-y-1">
                {manualRanges.map((r, i) => (
                  <div key={i} className="space-y-0.5 bg-slate-800/50 rounded p-1.5">
                    <div className="flex gap-1 items-center">
                      <input
                        type="text"
                        value={r.label}
                        onChange={(e) => updateRange(i, 'label', e.target.value)}
                        placeholder="Label"
                        className="flex-1 text-xs bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
                      />
                      <button
                        onClick={() => removeRange(i)}
                        className="text-[10px] text-rose-400 hover:text-rose-300 px-1"
                      >
                        ✕
                      </button>
                    </div>
                    <div className="flex gap-1">
                      <input
                        type="date"
                        value={r.start}
                        onChange={(e) => updateRange(i, 'start', e.target.value)}
                        className="flex-1 text-[10px] bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
                      />
                      <input
                        type="date"
                        value={r.end}
                        onChange={(e) => updateRange(i, 'end', e.target.value)}
                        className="flex-1 text-[10px] bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
                      />
                    </div>
                  </div>
                ))}
                <button
                  onClick={addRange}
                  className="text-[10px] text-sky-400 hover:text-sky-300 transition-colors"
                >
                  + Add range
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </details>
  )
}
