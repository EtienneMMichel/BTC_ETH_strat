interface RocCurveDef {
  name: string
  color: string
  points: [number, number][]
  auc: number
}

interface RocCurveChartProps {
  curves: RocCurveDef[]
  height?: number
}

// Fixed SVG viewport — scales naturally via CSS width: 100%
const W = 400
const H = 300
const ML = 48  // margin left (y-axis labels)
const MT = 12  // margin top
const MR = 16  // margin right
const MB = 36  // margin bottom

const PW = W - ML - MR  // plot width
const PH = H - MT - MB  // plot height

function toSvgX(fpr: number) { return ML + fpr * PW }
function toSvgY(tpr: number) { return MT + (1 - tpr) * PH }

function pointsToPath(pts: [number, number][]): string {
  if (pts.length === 0) return ''
  return pts
    .map(([fpr, tpr], i) => `${i === 0 ? 'M' : 'L'}${toSvgX(fpr).toFixed(1)},${toSvgY(tpr).toFixed(1)}`)
    .join(' ')
}

const TICK_COUNT = 5

export function RocCurveChart({ curves, height = 300 }: RocCurveChartProps) {
  const ticks = Array.from({ length: TICK_COUNT + 1 }, (_, i) => i / TICK_COUNT)

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      style={{ height }}
      className="select-none"
    >
      {/* Grid lines */}
      {ticks.map((t) => (
        <g key={t}>
          <line
            x1={toSvgX(t)} y1={MT}
            x2={toSvgX(t)} y2={MT + PH}
            stroke="#1e293b" strokeWidth={1}
          />
          <line
            x1={ML} y1={toSvgY(t)}
            x2={ML + PW} y2={toSvgY(t)}
            stroke="#1e293b" strokeWidth={1}
          />
        </g>
      ))}

      {/* Axes */}
      <line x1={ML} y1={MT} x2={ML} y2={MT + PH} stroke="#475569" strokeWidth={1} />
      <line x1={ML} y1={MT + PH} x2={ML + PW} y2={MT + PH} stroke="#475569" strokeWidth={1} />

      {/* Axis tick labels */}
      {ticks.map((t) => (
        <g key={t}>
          {/* x-axis */}
          <text
            x={toSvgX(t)} y={MT + PH + 14}
            textAnchor="middle" fontSize={9} fill="#64748b"
          >
            {t.toFixed(1)}
          </text>
          {/* y-axis */}
          <text
            x={ML - 6} y={toSvgY(t) + 3}
            textAnchor="end" fontSize={9} fill="#64748b"
          >
            {t.toFixed(1)}
          </text>
        </g>
      ))}

      {/* Axis labels */}
      <text x={ML + PW / 2} y={H - 2} textAnchor="middle" fontSize={10} fill="#94a3b8">
        FPR (False Positive Rate)
      </text>
      <text
        x={10} y={MT + PH / 2}
        textAnchor="middle" fontSize={10} fill="#94a3b8"
        transform={`rotate(-90, 10, ${MT + PH / 2})`}
      >
        TPR (True Positive Rate)
      </text>

      {/* Random classifier diagonal */}
      <line
        x1={toSvgX(0)} y1={toSvgY(0)}
        x2={toSvgX(1)} y2={toSvgY(1)}
        stroke="#334155" strokeWidth={1.5} strokeDasharray="4 3"
      />

      {/* ROC curves */}
      {curves.map(({ name, color, points }) => (
        <path
          key={name}
          d={pointsToPath(points)}
          fill="none"
          stroke={color}
          strokeWidth={1.8}
          strokeLinejoin="round"
        />
      ))}

      {/* Legend */}
      {curves.map(({ name, color, auc }, i) => (
        <g key={name} transform={`translate(${ML + 8}, ${MT + 10 + i * 16})`}>
          <line x1={0} y1={5} x2={16} y2={5} stroke={color} strokeWidth={2} />
          <text x={20} y={9} fontSize={9} fill="#cbd5e1">
            {name} (AUC {auc.toFixed(3)})
          </text>
        </g>
      ))}
    </svg>
  )
}
