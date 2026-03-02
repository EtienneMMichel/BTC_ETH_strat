import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import type { BacktestMetrics } from '@/types/backtest'
import { formatNum, formatPct } from '@/lib/utils'

interface MetricCardProps {
  label: string
  value: string
  positive?: boolean
}

function MetricCard({ label, value, positive }: MetricCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{label}</CardTitle>
      </CardHeader>
      <CardContent className={positive === false ? 'text-rose-400' : positive ? 'text-emerald-400' : undefined}>
        {value}
      </CardContent>
    </Card>
  )
}

export function MetricsGrid({ metrics }: { metrics: BacktestMetrics }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
      <MetricCard label="Sharpe Ratio" value={formatNum(metrics.sharpe_ratio)} positive={metrics.sharpe_ratio >= 0} />
      <MetricCard label="Max Drawdown" value={formatPct(metrics.max_drawdown)} positive={false} />
      <MetricCard label="Calmar Ratio" value={formatNum(metrics.calmar_ratio)} positive={metrics.calmar_ratio >= 0} />
      <MetricCard label="VaR 95%" value={formatPct(metrics.historical_var_5pct)} positive={false} />
      <MetricCard label="ES 95%" value={formatPct(metrics.expected_shortfall_5pct)} positive={false} />
      <MetricCard label="Win Rate" value={formatPct(metrics.win_rate ?? 0)} positive={(metrics.win_rate ?? 0) >= 0.5} />
    </div>
  )
}
