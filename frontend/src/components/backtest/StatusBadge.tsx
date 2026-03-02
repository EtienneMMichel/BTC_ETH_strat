import { Badge } from '@/components/ui/badge'
import type { JobStatus } from '@/types/backtest'

const config: Record<JobStatus, { label: string; variant: 'default' | 'warning' | 'success' | 'destructive' | 'secondary' }> = {
  pending: { label: 'Pending', variant: 'secondary' },
  running: { label: 'Running…', variant: 'warning' },
  done: { label: 'Done', variant: 'success' },
  failed: { label: 'Failed', variant: 'destructive' },
}

export function StatusBadge({ status }: { status: JobStatus }) {
  const { label, variant } = config[status]
  return <Badge variant={variant}>{label}</Badge>
}
