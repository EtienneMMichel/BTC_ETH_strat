import type { ReliabilityBin } from '@/types/models'

interface ReliabilityChartProps {
  data: ReliabilityBin[]
  color: string
  height?: number
}

const W = 300
const H = 220
const ML = 42
const MT = 12
const MR = 12
const MB = 32

const PW = W - ML - MR
const PH = H - MT - MB

function toSvgX(p: number) { return ML + p * PW }
function toSvgY(r: number) { return MT + (1 - r) * PH }

export function ReliabilityChart({ data, color, height = 220 }: ReliabilityChartProps) {
  if (data.length === 0) {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
        <text x={W / 2} y={H / 2} textAnchor="middle" fontSize={10} fill="#475569">
          No data
        </text>
      </svg>
    )
  }

  const ticks = [0, 0.25, 0.5, 0.75, 1.0]

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ height }} className="select-none">
      {/* Grid */}
      {ticks.map((t) => (
        <g key={t}>
          <line x1={toSvgX(t)} y1={MT} x2={toSvgX(t)} y2={MT + PH} stroke="#1e293b" strokeWidth={1} />
          <line x1={ML} y1={toSvgY(t)} x2={ML + PW} y2={toSvgY(t)} stroke="#1e293b" strokeWidth={1} />
        </g>
      ))}

      {/* Axes */}
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#475569" strokeWidth={1} />
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#475569" strokeWidth={1} />

      {/* Perfect calibration diagonal */}
      <line
        x1={toSvgX(0)} y1={toSvgY(0)}
        x2={toSvgX(1)} y2={toSvgY(1)}
        stroke="#334155" strokeWidth={1.5} strokeDasharray="4 3"
      />

      {/* Model calibration line */}
      {data.length > 1 && (
        <polyline
          points={data
            .map((d) => `${toSvgX(d.prob_bin).toFixed(1)},${toSvgY(d.actual_rate).toFixed(1)}`)
            .join(' ')}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinejoin="round"
        />
      )}

      {/* Dots */}
      {data.map((d, i) => (
        <circle
          key={i}
          cx={toSvgX(d.prob_bin)}
          cy={toSvgY(d.actual_rate)}
          r={3.5}
          fill={color}
          opacity={0.9}
        />
      ))}

      {/* Axis tick labels */}
      {ticks.map((t) => (
        <g key={t}>
          <text x={toSvgX(t)} y={MT + PH + 12} textAnchor="middle" fontSize={8} fill="#64748b">
            {t.toFixed(2)}
          </text>
          <text x={ML - 4} y={toSvgY(t) + 3} textAnchor="end" fontSize={8} fill="#64748b">
            {t.toFixed(2)}
          </text>
        </g>
      ))}

      {/* Axis labels */}
      <text x={ML + PW / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#94a3b8">
        Predicted P(up)
      </text>
      <text
        x={10} y={MT + PH / 2}
        textAnchor="middle" fontSize={9} fill="#94a3b8"
        transform={`rotate(-90, 10, ${MT + PH / 2})`}
      >
        Actual rate
      </text>
    </svg>
  )
}
