import { useMemo } from 'react'

const REGIME_PALETTE = [
  '#38bdf8', // sky
  '#34d399', // emerald
  '#fbbf24', // amber
  '#f87171', // red
  '#a78bfa', // violet
  '#fb923c', // orange
  '#e879f9', // fuchsia
  '#4ade80', // green
]

function buildColorMap(regimes: string[]): Record<string, string> {
  const map: Record<string, string> = {}
  regimes.forEach((r, i) => {
    map[r] = REGIME_PALETTE[i % REGIME_PALETTE.length]
  })
  return map
}

interface RegimeRowProps {
  modelId: string
  modelType: string
  labels: [string, string][]
  probabilities: Record<string, [string, number][]>
  uniqueRegimes: string[]
  dateRange: [string, string]
  mode: 'max' | 'prob'
  showTimeAxis?: boolean
  paddingLeft?: number
  paddingRight?: number
}

function MaxStrip({
  labels,
  colorMap,
  paddingLeft = 0,
  paddingRight = 0,
}: {
  labels: [string, string][]
  uniqueRegimes: string[]
  colorMap: Record<string, string>
  paddingLeft?: number
  paddingRight?: number
}) {
  const total = labels.length
  if (total === 0) return null

  const tsFirst = Date.parse(labels[0][0])
  const tsLast = Date.parse(labels[total - 1][0])
  const totalMs = tsLast - tsFirst || 1

  // Run-length encode
  const segments: { label: string; start: number; end: number }[] = []
  let i = 0
  while (i < total) {
    const label = labels[i][1]
    let j = i
    while (j < total && labels[j][1] === label) j++
    segments.push({ label, start: i, end: j - 1 })
    i = j
  }

  return (
    <div style={{ paddingLeft, paddingRight }}>
      <svg width="100%" height={28} className="rounded overflow-hidden">
        {segments.map((seg, idx) => {
          const xFrac = (Date.parse(labels[seg.start][0]) - tsFirst) / totalMs
          const wFrac =
            (Date.parse(labels[seg.end][0]) - Date.parse(labels[seg.start][0])) / totalMs +
            1 / total
          return (
            <rect
              key={idx}
              x={`${xFrac * 100}%`}
              y={0}
              width={`${Math.max(wFrac * 100, 0.1)}%`}
              height={28}
              fill={colorMap[seg.label] ?? '#64748b'}
              opacity={0.75}
            />
          )
        })}
      </svg>
    </div>
  )
}

function ProbStrip({
  probabilities,
  uniqueRegimes,
  colorMap,
  paddingLeft = 0,
  paddingRight = 0,
}: {
  probabilities: Record<string, [string, number][]>
  uniqueRegimes: string[]
  colorMap: Record<string, string>
  paddingLeft?: number
  paddingRight?: number
}) {
  const height = 72

  // All regimes must have same length; use first available
  const firstKey = uniqueRegimes[0]
  if (!firstKey || !probabilities[firstKey]) return null
  const probs0 = probabilities[firstKey]
  const n = probs0.length
  if (n === 0) return null

  const tsFirst = Date.parse(probs0[0][0])
  const tsLast = Date.parse(probs0[n - 1][0])
  const totalMs = tsLast - tsFirst || 1

  // Build stacked polygon paths per regime
  const paths = useMemo(() => {
    return uniqueRegimes.map((regime, rIdx) => {
      const probs = probabilities[regime] ?? []

      // bottom of this band = sum of probs of regimes below
      const getBottom = (t: number): number => {
        let bottom = 0
        for (const prevRegime of uniqueRegimes.slice(0, rIdx)) {
          bottom += (probabilities[prevRegime]?.[t]?.[1] ?? 0)
        }
        return bottom
      }
      const getTop = (t: number): number => getBottom(t) + (probs[t]?.[1] ?? 0)

      const pts: string[] = []
      // Left to right along top edge — use timestamp fractions for x
      for (let t = 0; t < n; t++) {
        const xFrac = (Date.parse(probs0[t][0]) - tsFirst) / totalMs
        const x = xFrac * 100
        const y = (1 - getTop(t)) * height
        pts.push(`${x}%,${y}`)
      }
      // Right to left along bottom edge
      for (let t = n - 1; t >= 0; t--) {
        const xFrac = (Date.parse(probs0[t][0]) - tsFirst) / totalMs
        const x = xFrac * 100
        const y = (1 - getBottom(t)) * height
        pts.push(`${x}%,${y}`)
      }
      return { regime, points: pts.join(' '), color: colorMap[regime] ?? '#64748b' }
    })
  }, [probabilities, uniqueRegimes, colorMap, n, height, tsFirst, totalMs, probs0])

  return (
    <div style={{ paddingLeft, paddingRight }}>
      <div className="relative">
        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 h-full flex flex-col justify-between pointer-events-none text-[9px] text-slate-500 pr-1" style={{ width: 28 }}>
          <span>100%</span>
          <span>50%</span>
          <span>0%</span>
        </div>
        <div style={{ marginLeft: 30 }}>
          <svg width="100%" height={height} className="rounded overflow-hidden">
            {paths.map(({ regime, points, color }) => (
              <polygon
                key={regime}
                points={points}
                fill={color}
                opacity={0.75}
              />
            ))}
            {/* 50% guide line */}
            <line
              x1="0"
              y1={height / 2}
              x2="100%"
              y2={height / 2}
              stroke="rgba(255,255,255,0.1)"
              strokeWidth={1}
            />
          </svg>
        </div>
      </div>
    </div>
  )
}

export function RegimeRow({
  modelId,
  modelType,
  labels,
  probabilities,
  uniqueRegimes,
  dateRange,
  mode,
  showTimeAxis = false,
  paddingLeft = 0,
  paddingRight = 0,
}: RegimeRowProps) {
  const colorMap = useMemo(() => buildColorMap(uniqueRegimes), [uniqueRegimes])

  return (
    <div className="space-y-1">
      {/* Header row */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-medium text-slate-300 truncate max-w-[180px]">{modelId}</span>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-400 uppercase tracking-wide">
          {modelType}
        </span>
        {/* Color legend */}
        <div className="flex gap-2 flex-wrap">
          {uniqueRegimes.map((r) => (
            <span key={r} className="flex items-center gap-1 text-[10px] text-slate-400">
              <span
                className="inline-block w-2 h-2 rounded-sm"
                style={{ background: colorMap[r] }}
              />
              {r}
            </span>
          ))}
        </div>
      </div>

      {/* Chart strip */}
      {mode === 'max' ? (
        <MaxStrip
          labels={labels}
          uniqueRegimes={uniqueRegimes}
          colorMap={colorMap}
          paddingLeft={paddingLeft}
          paddingRight={paddingRight}
        />
      ) : (
        <ProbStrip
          probabilities={probabilities}
          uniqueRegimes={uniqueRegimes}
          colorMap={colorMap}
          paddingLeft={paddingLeft}
          paddingRight={paddingRight}
        />
      )}

      {/* Time axis (bottom-most row only) */}
      {showTimeAxis && (
        <div className="flex justify-between text-[9px] text-slate-500 px-0.5">
          <span>{dateRange[0]?.slice(0, 10)}</span>
          <span>{dateRange[1]?.slice(0, 10)}</span>
        </div>
      )}
    </div>
  )
}
