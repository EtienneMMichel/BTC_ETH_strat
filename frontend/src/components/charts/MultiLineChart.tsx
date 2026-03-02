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
import type { TimeSeries } from '@/types/models'

export interface SeriesDef {
  data: TimeSeries
  color: string
  title: string
}

interface MultiLineChartProps {
  series: SeriesDef[]
  height?: number
}

export function MultiLineChart({ series, height = 300 }: MultiLineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const seriesRefs = useRef<ISeriesApi<'Line'>[]>([])

  // Mount: create chart once
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
      height,
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
      seriesRefs.current = []
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Data: update series when data changes
  useEffect(() => {
    const chart = chartRef.current
    if (!chart) return

    // Remove old series
    seriesRefs.current.forEach((s) => chart.removeSeries(s))
    seriesRefs.current = []

    // Add new series
    series.forEach(({ data, color, title }) => {
      const s = chart.addSeries(LineSeries, { color, lineWidth: 2, title })
      const lineData: LineData[] = [...data]
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .map(([time, value]) => ({ time: time.slice(0, 10) as Time, value }))
        .filter((item, idx, arr) => idx === 0 || item.time !== arr[idx - 1].time)
      s.setData(lineData)
      seriesRefs.current.push(s)
    })

    if (series.length > 0) chart.timeScale().fitContent()
  }, [series])

  return <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
}
