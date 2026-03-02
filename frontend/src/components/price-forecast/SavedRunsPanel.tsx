import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { listSavedRuns, getSavedRun } from '@/lib/api/models'
import type { PriceForecastResult } from '@/types/models'

interface SavedRunsPanelProps {
  onLoad: (result: PriceForecastResult) => void
}

export function SavedRunsPanel({ onLoad }: SavedRunsPanelProps) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState<string | null>(null)

  const { data: runs } = useQuery({
    queryKey: ['saved-runs'],
    queryFn: listSavedRuns,
    enabled: open,
    staleTime: 10_000,
  })

  const handleLoad = async (runId: string) => {
    setLoading(runId)
    try {
      const result = await getSavedRun(runId)
      onLoad(result)
    } finally {
      setLoading(null)
    }
  }

  return (
    <details open={open} onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="text-xs text-slate-400 cursor-pointer select-none hover:text-slate-200 transition-colors">
        Saved runs {runs ? `(${runs.length})` : ''}
      </summary>
      <div className="mt-1 space-y-1 max-h-48 overflow-y-auto">
        {runs?.length === 0 && (
          <p className="text-[10px] text-slate-600">No saved runs yet.</p>
        )}
        {runs?.map((run) => (
          <div
            key={run.run_id}
            className="flex items-center justify-between gap-1 bg-slate-800/60 rounded px-2 py-1"
          >
            <div className="min-w-0">
              <p className="text-xs text-slate-200 truncate font-medium">{run.run_name}</p>
              <p className="text-[9px] text-slate-500 truncate">
                {run.assets.join('+')} · {run.forecast_horizon} ·{' '}
                {new Date(run.timestamp).toLocaleDateString()}
              </p>
            </div>
            <button
              onClick={() => handleLoad(run.run_id)}
              disabled={loading === run.run_id}
              className="shrink-0 text-[10px] px-2 py-0.5 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded transition-colors disabled:opacity-50"
            >
              {loading === run.run_id ? '…' : 'Load'}
            </button>
          </div>
        ))}
      </div>
    </details>
  )
}
