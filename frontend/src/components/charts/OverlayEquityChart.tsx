import { useEffect, useRef } from 'react'
import {
  createChart,
  LineSeries,
  ColorType,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type Time,
} from 'lightweight-charts'

export interface CurveEntry {
  label: string
  color: string
  data: [string, number][]
  visible: boolean
}

interface OverlayEquityChartProps {
  curves: CurveEntry[]
  benchmarks?: Record<string, [string, number][]>
}

const BENCHMARK_COLORS: Record<string, string> = {
  BTC: '#f59e0b',
  ETH: '#8b5cf6',
}

function toLineData(curve: [string, number][]): LineData[] {
  return [...curve]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([time, value]) => ({ time: time.slice(0, 10) as Time, value }))
    .filter((item, idx, arr) => idx === 0 || item.time !== arr[idx - 1].time)
}

export function OverlayEquityChart({ curves, benchmarks }: OverlayEquityChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesMapRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map())
  const benchSeriesRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map())

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return
    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#1e293b' },
        horzLines: { color: '#1e293b' },
      },
      width: containerRef.current.clientWidth,
      height: 350,
    })
    chartRef.current = chart

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) chart.applyOptions({ width: entry.contentRect.width })
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      seriesMapRef.current = new Map()
      benchSeriesRef.current = new Map()
    }
  }, [])

  // Sync strategy curves
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    const prev = seriesMapRef.current
    const curveKeys = new Set(curves.map((c) => c.label))

    // Remove series no longer present
    for (const [key, s] of prev) {
      if (!curveKeys.has(key)) {
        chart.removeSeries(s)
        prev.delete(key)
      }
    }

    // Add or update each curve
    for (const curve of curves) {
      let s = prev.get(curve.label)
      if (!s) {
        s = chart.addSeries(LineSeries, {
          color: curve.color,
          lineWidth: 2,
          title: curve.label,
          visible: curve.visible,
        })
        prev.set(curve.label, s)
      }
      s.applyOptions({ visible: curve.visible, color: curve.color })
      s.setData(toLineData(curve.data))
    }

    chart.timeScale().fitContent()
  }, [curves])

  // Sync benchmarks
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    const prev = benchSeriesRef.current
    const nextKeys = new Set(benchmarks ? Object.keys(benchmarks) : [])

    for (const [key, s] of prev) {
      if (!nextKeys.has(key)) {
        chart.removeSeries(s)
        prev.delete(key)
      }
    }

    if (!benchmarks) return

    for (const [asset, curve] of Object.entries(benchmarks)) {
      let s = prev.get(asset)
      if (!s) {
        const c = BENCHMARK_COLORS[asset] ?? '#64748b'
        s = chart.addSeries(LineSeries, {
          color: c,
          lineWidth: 1,
          title: asset,
          lineStyle: 2,
        })
        prev.set(asset, s)
      }
      s.setData(toLineData(curve))
    }
  }, [benchmarks])

  return <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
}
