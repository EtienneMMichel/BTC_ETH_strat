import type { FrontierPoint, StrategyResultEntry } from '@/types/compare'

interface EfficientFrontierChartProps {
  frontier: FrontierPoint[]
  special?: Record<string, FrontierPoint> | null
  strategies: StrategyResultEntry[]
  colors: string[]
  height?: number
}

const W = 400
const H = 280
const ML = 52
const MT = 16
const MR = 16
const MB = 36

const PW = W - ML - MR
const PH = H - MT - MB

export function EfficientFrontierChart({
  frontier,
  special,
  strategies,
  colors,
  height = 280,
}: EfficientFrontierChartProps) {
  // Collect all points to determine axis ranges
  const allVols: number[] = []
  const allRets: number[] = []

  for (const p of frontier) {
    allVols.push(p.ann_vol * 100)
    allRets.push(p.ann_return * 100)
  }
  if (special) {
    for (const p of Object.values(special)) {
      allVols.push(p.ann_vol * 100)
      allRets.push(p.ann_return * 100)
    }
  }
  for (const s of strategies) {
    allVols.push(s.ann_vol * 100)
    allRets.push(s.ann_return * 100)
  }

  if (allVols.length === 0) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={10} fill="#475569">
          No frontier data
        </text>
      </svg>
    )
  }

  const minVol = Math.max(0, Math.min(...allVols) - 5)
  const maxVol = Math.max(...allVols) + 5
  const minRet = Math.min(...allRets) - 5
  const maxRet = Math.max(...allRets) + 5

  const toX = (vol: number) => ML + ((vol - minVol) / (maxVol - minVol)) * PW
  const toY = (ret: number) => MT + (1 - (ret - minRet) / (maxRet - minRet)) * PH

  // Generate nice ticks
  const volTicks = niceTicksRange(minVol, maxVol, 5)
  const retTicks = niceTicksRange(minRet, maxRet, 5)

  // Frontier polyline points
  const sortedFrontier = [...frontier].sort((a, b) => a.ann_vol - b.ann_vol)
  const polyPoints = sortedFrontier
    .map((p) => `${toX(p.ann_vol * 100).toFixed(1)},${toY(p.ann_return * 100).toFixed(1)}`)
    .join(' ')

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
      {/* Grid */}
      {volTicks.map((v) => (
        <line key={`vg${v}`} x1={toX(v)} y1={MT} x2={toX(v)} y2={MT + PH} stroke="#1e293b" strokeWidth={1} />
      ))}
      {retTicks.map((r) => (
        <line key={`rg${r}`} x1={ML} y1={toY(r)} x2={ML + PW} y2={toY(r)} stroke="#1e293b" strokeWidth={1} />
      ))}

      {/* Axes */}
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#475569" strokeWidth={1} />
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#475569" strokeWidth={1} />

      {/* Frontier curve */}
      {sortedFrontier.length > 1 && (
        <polyline
          points={polyPoints}
          fill="none"
          stroke="#64748b"
          strokeWidth={1.5}
          strokeLinejoin="round"
        />
      )}

      {/* Frontier dots */}
      {sortedFrontier.map((p, i) => (
        <circle
          key={`f${i}`}
          cx={toX(p.ann_vol * 100)}
          cy={toY(p.ann_return * 100)}
          r={2.5}
          fill="#64748b"
          opacity={0.6}
        >
          <title>{`gamma=${p.gamma.toFixed(2)} vol=${(p.ann_vol * 100).toFixed(1)}% ret=${(p.ann_return * 100).toFixed(1)}%`}</title>
        </circle>
      ))}

      {/* Special points */}
      {special?.min_var && (
        <circle
          cx={toX(special.min_var.ann_vol * 100)}
          cy={toY(special.min_var.ann_return * 100)}
          r={5}
          fill="none"
          stroke="#facc15"
          strokeWidth={2}
        >
          <title>{`Min Variance: vol=${(special.min_var.ann_vol * 100).toFixed(1)}% ret=${(special.min_var.ann_return * 100).toFixed(1)}%`}</title>
        </circle>
      )}
      {special?.max_sharpe && (
        <circle
          cx={toX(special.max_sharpe.ann_vol * 100)}
          cy={toY(special.max_sharpe.ann_return * 100)}
          r={5}
          fill="none"
          stroke="#22d3ee"
          strokeWidth={2}
        >
          <title>{`Max Sharpe: vol=${(special.max_sharpe.ann_vol * 100).toFixed(1)}% ret=${(special.max_sharpe.ann_return * 100).toFixed(1)}%`}</title>
        </circle>
      )}

      {/* Strategy points */}
      {strategies.map((s, i) => (
        <circle
          key={`s${i}`}
          cx={toX(s.ann_vol * 100)}
          cy={toY(s.ann_return * 100)}
          r={5}
          fill={colors[i % colors.length]}
          opacity={0.9}
        >
          <title>{`${s.label}: vol=${(s.ann_vol * 100).toFixed(1)}% ret=${(s.ann_return * 100).toFixed(1)}%`}</title>
        </circle>
      ))}

      {/* Axis tick labels */}
      {volTicks.map((v) => (
        <text key={`vl${v}`} x={toX(v)} y={MT + PH + 14} textAnchor="middle" fontSize={8} fill="#64748b">
          {v.toFixed(0)}%
        </text>
      ))}
      {retTicks.map((r) => (
        <text key={`rl${r}`} x={ML - 4} y={toY(r) + 3} textAnchor="end" fontSize={8} fill="#64748b">
          {r.toFixed(0)}%
        </text>
      ))}

      {/* Axis labels */}
      <text x={ML + PW / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#94a3b8">
        Annualised Volatility
      </text>
      <text
        x={10}
        y={MT + PH / 2}
        textAnchor="middle"
        fontSize={9}
        fill="#94a3b8"
        transform={`rotate(-90, 10, ${MT + PH / 2})`}
      >
        Annualised Return
      </text>

      {/* Legend */}
      {special?.min_var && (
        <g transform={`translate(${ML + 8}, ${MT + 8})`}>
          <circle cx={0} cy={0} r={4} fill="none" stroke="#facc15" strokeWidth={1.5} />
          <text x={8} y={3} fontSize={8} fill="#facc15">Min Var</text>
        </g>
      )}
      {special?.max_sharpe && (
        <g transform={`translate(${ML + 8}, ${MT + 22})`}>
          <circle cx={0} cy={0} r={4} fill="none" stroke="#22d3ee" strokeWidth={1.5} />
          <text x={8} y={3} fontSize={8} fill="#22d3ee">Max Sharpe</text>
        </g>
      )}
    </svg>
  )
}

/** Generate roughly `count` nice ticks between min and max. */
function niceTicksRange(min: number, max: number, count: number): number[] {
  const range = max - min
  if (range <= 0) return [min]
  const rawStep = range / count
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const residual = rawStep / magnitude
  let step: number
  if (residual <= 1.5) step = magnitude
  else if (residual <= 3) step = 2 * magnitude
  else if (residual <= 7) step = 5 * magnitude
  else step = 10 * magnitude

  const ticks: number[] = []
  let t = Math.ceil(min / step) * step
  while (t <= max) {
    ticks.push(t)
    t += step
  }
  return ticks
}
