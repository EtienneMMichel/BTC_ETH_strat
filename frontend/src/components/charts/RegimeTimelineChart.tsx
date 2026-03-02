import { useMemo } from 'react'

// Palette for regime labels
const REGIME_COLORS = [
  '#38bdf8', // sky
  '#34d399', // emerald
  '#fbbf24', // amber
  '#f87171', // red
  '#a78bfa', // violet
  '#fb923c', // orange
]

interface RegimeTimelineChartProps {
  /** Array of [iso_timestamp, label] */
  data: [string, string][]
  height?: number
}

export function RegimeTimelineChart({ data, height = 32 }: RegimeTimelineChartProps) {
  const { segments, labelColors, labelCounts } = useMemo(() => {
    if (data.length === 0)
      return { segments: [], labelColors: {} as Record<string, string>, labelCounts: {} as Record<string, number> }

    const uniqueLabels = [...new Set(data.map(([, l]) => l))].sort()
    const labelColors: Record<string, string> = {}
    uniqueLabels.forEach((l, i) => {
      labelColors[l] = REGIME_COLORS[i % REGIME_COLORS.length]
    })

    const labelCounts: Record<string, number> = {}
    data.forEach(([, l]) => {
      labelCounts[l] = (labelCounts[l] ?? 0) + 1
    })

    // Build run-length segments
    const segments: { label: string; start: number; end: number }[] = []
    let i = 0
    while (i < data.length) {
      const label = data[i][1]
      let j = i
      while (j < data.length && data[j][1] === label) j++
      segments.push({ label, start: i, end: j - 1 })
      i = j
    }

    return { segments, labelColors, labelCounts }
  }, [data])

  if (data.length === 0) return null

  const total = data.length

  return (
    <div className="space-y-1">
      <svg width="100%" height={height} className="rounded overflow-hidden">
        {segments.map((seg, i) => {
          const x = (seg.start / total) * 100
          const w = ((seg.end - seg.start + 1) / total) * 100
          return (
            <rect
              key={i}
              x={`${x}%`}
              y={0}
              width={`${w}%`}
              height={height}
              fill={labelColors[seg.label]}
              opacity={0.7}
            />
          )
        })}
      </svg>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5">
        {Object.entries(labelColors).map(([label, color]) => (
          <span key={label} className="text-[9px] flex items-center gap-0.5">
            <span className="inline-block w-2 h-2 rounded-sm" style={{ background: color }} />
            <span className="text-slate-400">
              {label} <span className="text-slate-500">({labelCounts[label]})</span>
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}
