import type { CalibrationBin } from '@/types/models'

interface CalibrationChartProps {
  data: CalibrationBin[]
  color: string
  height?: number
}

const W = 320
const H = 180
const ML = 42
const MT = 10
const MR = 10
const MB = 28

const PW = W - ML - MR
const PH = H - MT - MB

export function CalibrationChart({ data, color, height = 180 }: CalibrationChartProps) {
  if (data.length === 0) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={10} fill="#475569">
          No data
        </text>
      </svg>
    )
  }

  const maxRet = Math.max(...data.map((d) => d.mean_abs_return), 1e-9)
  const meanRet = data.reduce((s, d) => s + d.mean_abs_return, 0) / data.length

  const barW = PW / data.length
  const barPad = Math.max(1, barW * 0.15)

  function toSvgY(v: number) {
    return MT + PH - (v / maxRet) * PH
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1.0]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
      {/* Grid lines */}
      {ticks.map((t) => (
        <line
          key={t}
          x1={ML} y1={toSvgY(t * maxRet)}
          x2={ML + PW} y2={toSvgY(t * maxRet)}
          stroke="#1e293b" strokeWidth={1}
        />
      ))}

      {/* Axes */}
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#475569" strokeWidth={1} />
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#475569" strokeWidth={1} />

      {/* Y-axis ticks */}
      {ticks.map((t) => (
        <text key={t} x={ML - 4} y={toSvgY(t * maxRet) + 3} textAnchor="end" fontSize={8} fill="#64748b">
          {(t * maxRet * 100).toFixed(1)}%
        </text>
      ))}

      {/* Bars */}
      {data.map((d, i) => {
        const x = ML + i * barW + barPad
        const bw = barW - 2 * barPad
        const barH = (d.mean_abs_return / maxRet) * PH
        const y = MT + PH - barH
        return (
          <rect
            key={i}
            x={x} y={y} width={Math.max(1, bw)} height={barH}
            fill={color} opacity={0.75}
          />
        )
      })}

      {/* Mean dashed line */}
      {maxRet > 0 && (
        <line
          x1={ML} y1={toSvgY(meanRet)}
          x2={ML + PW} y2={toSvgY(meanRet)}
          stroke="#f59e0b" strokeWidth={1} strokeDasharray="4 3"
        />
      )}

      {/* X-axis label */}
      <text x={ML + PW / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#64748b">
        |signal| decile → avg |return|
      </text>
    </svg>
  )
}
