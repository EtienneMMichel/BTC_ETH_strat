import type { AcfPoint } from '@/types/models'

interface AcfChartProps {
  data: AcfPoint[]
  color: string
  n_obs: number
  height?: number
}

const W = 280
const H = 160
const ML = 36
const MT = 10
const MR = 8
const MB = 22

const PW = W - ML - MR
const PH = H - MT - MB

export function AcfChart({ data, color, n_obs, height = 160 }: AcfChartProps) {
  if (data.length === 0) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={10} fill="#475569">
          No data
        </text>
      </svg>
    )
  }

  const maxLag = data[data.length - 1].lag
  const bound = n_obs > 0 ? 1.96 / Math.sqrt(n_obs) : 0.2

  // SVG helpers
  function toSvgX(lag: number) {
    return ML + ((lag - 0.5) / maxLag) * PW
  }
  function toSvgY(acf: number) {
    // center at 0.5 of plot height
    const mid = MT + PH / 2
    return mid - (acf / 1) * (PH / 2)
  }

  const zeroY = toSvgY(0)
  const boundPosY = toSvgY(bound)
  const boundNegY = toSvgY(-bound)

  const barW = (PW / maxLag) * 0.6

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
      {/* Axes */}
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#475569" strokeWidth={1} />
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#475569" strokeWidth={1} />

      {/* Zero line */}
      <line x1={ML} y1={zeroY} x2={ML + PW} y2={zeroY} stroke="#334155" strokeWidth={1} />

      {/* ±1.96/√N significance bounds */}
      <line
        x1={ML} y1={boundPosY} x2={ML + PW} y2={boundPosY}
        stroke="#f59e0b" strokeWidth={1} strokeDasharray="4 3"
      />
      <line
        x1={ML} y1={boundNegY} x2={ML + PW} y2={boundNegY}
        stroke="#f59e0b" strokeWidth={1} strokeDasharray="4 3"
      />

      {/* ACF bars */}
      {data.map((d) => {
        const cx = toSvgX(d.lag)
        const y1 = zeroY
        const y2 = toSvgY(d.acf)
        const top = Math.min(y1, y2)
        const barH = Math.abs(y2 - y1)
        return (
          <rect
            key={d.lag}
            x={cx - barW / 2} y={top}
            width={barW} height={Math.max(1, barH)}
            fill={color} opacity={0.8}
          />
        )
      })}

      {/* Y-axis labels */}
      <text x={ML - 3} y={MT + 4} textAnchor="end" fontSize={8} fill="#64748b">+1</text>
      <text x={ML - 3} y={zeroY + 3} textAnchor="end" fontSize={8} fill="#64748b">0</text>
      <text x={ML - 3} y={MT + PH + 3} textAnchor="end" fontSize={8} fill="#64748b">−1</text>

      {/* X-axis label */}
      <text x={ML + PW / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#64748b">
        Lag
      </text>
    </svg>
  )
}
