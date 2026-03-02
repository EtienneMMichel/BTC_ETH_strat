import type { HistogramBin } from '@/types/models'

interface HistogramChartProps {
  data: HistogramBin[]
  color: string
  height?: number
}

const W = 280
const H = 160
const ML = 36
const MT = 8
const MR = 8
const MB = 22

const PW = W - ML - MR
const PH = H - MT - MB

export function HistogramChart({ data, color, height = 160 }: HistogramChartProps) {
  if (data.length === 0) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={10} fill="#475569">
          No data
        </text>
      </svg>
    )
  }

  const maxCount = Math.max(...data.map((d) => d.count), 1)
  const minBin = data[0].bin_center
  const maxBin = data[data.length - 1].bin_center
  const range = maxBin - minBin || 1

  const barW = PW / data.length
  const barPad = Math.max(0.5, barW * 0.08)

  function toSvgX(v: number) {
    return ML + ((v - minBin) / range) * PW
  }

  // Zero line
  const zeroX = toSvgX(0)
  const showZero = minBin < 0 && maxBin > 0

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
      {/* Axes */}
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#475569" strokeWidth={1} />
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#475569" strokeWidth={1} />

      {/* Zero dashed line */}
      {showZero && (
        <line
          x1={zeroX} y1={MT}
          x2={zeroX} y2={MT + PH}
          stroke="#64748b" strokeWidth={1} strokeDasharray="3 2"
        />
      )}

      {/* Bars */}
      {data.map((d, i) => {
        const x = ML + i * barW + barPad
        const bw = Math.max(1, barW - 2 * barPad)
        const barH = (d.count / maxCount) * PH
        const y = MT + PH - barH
        return (
          <rect
            key={i}
            x={x} y={y} width={bw} height={barH}
            fill={color} opacity={0.7}
          />
        )
      })}

      {/* Y-axis: max count */}
      <text x={ML - 3} y={MT + 4} textAnchor="end" fontSize={8} fill="#64748b">
        {maxCount}
      </text>
      <text x={ML - 3} y={MT + PH} textAnchor="end" fontSize={8} fill="#64748b">
        0
      </text>

      {/* X-axis min/max */}
      <text x={ML} y={H - 4} textAnchor="start" fontSize={8} fill="#64748b">
        {minBin.toFixed(2)}
      </text>
      <text x={ML + PW} y={H - 4} textAnchor="end" fontSize={8} fill="#64748b">
        {maxBin.toFixed(2)}
      </text>
    </svg>
  )
}
