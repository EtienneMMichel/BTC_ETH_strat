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

interface EquityChartProps {
  equityCurve: [string, number][]
  label?: string
  color?: string
  equityCurve2?: [string, number][]
  label2?: string
  color2?: string
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

export function EquityChart({
  equityCurve,
  label = 'NAV',
  color = '#38bdf8',
  equityCurve2,
  label2 = 'NAV 2',
  color2 = '#a78bfa',
  benchmarks,
}: EquityChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const series2Ref = useRef<ISeriesApi<'Line'> | null>(null)
  const benchSeriesRef = useRef<Map<string, ISeriesApi<'Line'>>>(new Map())

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
      height: 300,
    })
    chartRef.current = chart

    const series = chart.addSeries(LineSeries, { color, lineWidth: 2, title: label })
    seriesRef.current = series

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) chart.applyOptions({ width: entry.contentRect.width })
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
      series2Ref.current = null
      benchSeriesRef.current = new Map()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const series = seriesRef.current
    if (!series || !equityCurve) return
    series.setData(toLineData(equityCurve))
    chartRef.current?.timeScale().fitContent()
  }, [equityCurve])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return
    if (equityCurve2) {
      if (!series2Ref.current) {
        series2Ref.current = chart.addSeries(LineSeries, { color: color2, lineWidth: 2, title: label2 })
      }
      series2Ref.current.setData(toLineData(equityCurve2))
    } else if (series2Ref.current) {
      chart.removeSeries(series2Ref.current)
      series2Ref.current = null
    }
  }, [equityCurve2, color2, label2])

  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    const prev = benchSeriesRef.current
    const nextKeys = new Set(benchmarks ? Object.keys(benchmarks) : [])

    // Remove series that are no longer in benchmarks
    for (const [key, s] of prev) {
      if (!nextKeys.has(key)) {
        chart.removeSeries(s)
        prev.delete(key)
      }
    }

    if (!benchmarks) return

    // Add or update each benchmark
    for (const [asset, curve] of Object.entries(benchmarks)) {
      let s = prev.get(asset)
      if (!s) {
        const c = BENCHMARK_COLORS[asset] ?? '#64748b'
        s = chart.addSeries(LineSeries, { color: c, lineWidth: 1, title: asset, lineStyle: 2 })
        prev.set(asset, s)
      }
      s.setData(toLineData(curve))
    }
  }, [benchmarks])

  return <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
}
