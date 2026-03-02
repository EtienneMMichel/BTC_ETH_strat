import type { JobStatus } from '@/types/backtest'

export function JobProgressBar({ status }: { status: JobStatus }) {
  if (status !== 'pending' && status !== 'running') return null

  return (
    <div className="relative h-1 w-full bg-slate-700 rounded overflow-hidden">
      <div
        className="absolute inset-y-0 left-0 w-1/2 bg-sky-500 rounded"
        style={{ animation: 'progress-indeterminate 1.5s ease-in-out infinite' }}
      />
    </div>
  )
}
