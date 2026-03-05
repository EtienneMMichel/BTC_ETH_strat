import { useState } from 'react'
import type { StrategyResultEntry } from '@/types/compare'

interface CompareMetricsTableProps {
  strategies: StrategyResultEntry[]
  colors: string[]
}

type SortDir = 'asc' | 'desc'

const COLUMNS: { key: string; label: string; format: (v: number) => string; higherBetter: boolean }[] = [
  { key: 'ann_return', label: 'Ann. Return', format: (v) => `${(v * 100).toFixed(2)}%`, higherBetter: true },
  { key: 'ann_vol', label: 'Ann. Vol', format: (v) => `${(v * 100).toFixed(2)}%`, higherBetter: false },
  { key: 'sharpe_ratio', label: 'Sharpe', format: (v) => v.toFixed(3), higherBetter: true },
  { key: 'max_drawdown', label: 'Max DD', format: (v) => `${(v * 100).toFixed(2)}%`, higherBetter: false },
  { key: 'calmar_ratio', label: 'Calmar', format: (v) => v.toFixed(3), higherBetter: true },
  { key: 'historical_var_5pct', label: 'VaR 5%', format: (v) => `${(v * 100).toFixed(2)}%`, higherBetter: false },
  { key: 'expected_shortfall_5pct', label: 'ES 5%', format: (v) => `${(v * 100).toFixed(2)}%`, higherBetter: false },
  { key: 'trade_count', label: 'Trades', format: (v) => v.toFixed(0), higherBetter: false },
]

function getValue(s: StrategyResultEntry, key: string): number {
  if (key === 'ann_return') return s.ann_return
  if (key === 'ann_vol') return s.ann_vol
  if (key === 'trade_count') return s.trade_count
  return s.metrics[key] ?? 0
}

export function CompareMetricsTable({ strategies, colors }: CompareMetricsTableProps) {
  const [sortKey, setSortKey] = useState('ann_return')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const handleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('desc')
    }
  }

  const sorted = [...strategies].sort((a, b) => {
    const va = getValue(a, sortKey)
    const vb = getValue(b, sortKey)
    return sortDir === 'asc' ? va - vb : vb - va
  })

  // Find best/worst per column
  const bestWorst: Record<string, { best: number; worst: number }> = {}
  for (const col of COLUMNS) {
    const vals = strategies.map((s) => getValue(s, col.key))
    if (col.higherBetter) {
      bestWorst[col.key] = { best: Math.max(...vals), worst: Math.min(...vals) }
    } else {
      bestWorst[col.key] = { best: Math.min(...vals), worst: Math.max(...vals) }
    }
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700">
            <th className="text-left py-2 px-2 text-slate-400 font-medium">Strategy</th>
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className="text-right py-2 px-2 text-slate-400 font-medium cursor-pointer hover:text-slate-200 select-none whitespace-nowrap"
              >
                {col.label}
                {sortKey === col.key && (
                  <span className="ml-1 text-xs">{sortDir === 'asc' ? '\u25B2' : '\u25BC'}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((s) => {
            const origIdx = strategies.indexOf(s)
            return (
              <tr key={s.label} className="border-b border-slate-800 hover:bg-slate-800/50">
                <td className="py-2 px-2 flex items-center gap-2">
                  <span
                    className="inline-block w-2.5 h-2.5 rounded-full shrink-0"
                    style={{ background: colors[origIdx % colors.length] }}
                  />
                  <span className="text-slate-200">{s.label}</span>
                </td>
                {COLUMNS.map((col) => {
                  const v = getValue(s, col.key)
                  const bw = bestWorst[col.key]
                  const isBest = strategies.length > 1 && v === bw.best
                  const isWorst = strategies.length > 1 && v === bw.worst
                  return (
                    <td
                      key={col.key}
                      className={`text-right py-2 px-2 tabular-nums ${
                        isBest ? 'text-emerald-400' : isWorst ? 'text-rose-400' : 'text-slate-300'
                      }`}
                    >
                      {col.format(v)}
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
