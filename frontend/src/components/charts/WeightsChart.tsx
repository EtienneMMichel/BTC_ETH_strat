import { useEffect, useRef } from 'react'
import {
  createChart,
  AreaSeries,
  ColorType,
  type IChartApi,
  type Time,
} from 'lightweight-charts'

interface WeightsChartProps {
  signals: [string, Record<string, number>][]
}

export function WeightsChart({ signals }: WeightsChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  useEffect(() => {
    if (!containerRef.current || !signals?.length) return
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
      height: 220,
    })
    chartRef.current = chart

    const btcSeries = chart.addSeries(AreaSeries, {
      lineColor: '#f59e0b',
      topColor: 'rgba(245,158,11,0.4)',
      bottomColor: 'rgba(245,158,11,0)',
      title: 'BTC',
    })
    const ethSeries = chart.addSeries(AreaSeries, {
      lineColor: '#38bdf8',
      topColor: 'rgba(56,189,248,0.4)',
      bottomColor: 'rgba(56,189,248,0)',
      title: 'ETH',
    })

    const sorted = [...signals].sort(([a], [b]) => (a < b ? -1 : 1))
    btcSeries.setData(sorted.map(([time, w]) => ({ time: time.slice(0, 10) as Time, value: w['BTC'] ?? 0 })))
    ethSeries.setData(sorted.map(([time, w]) => ({ time: time.slice(0, 10) as Time, value: w['ETH'] ?? 0 })))
    chart.timeScale().fitContent()

    const ro = new ResizeObserver((es) => {
      const e = es[0]
      if (e) chart.applyOptions({ width: e.contentRect.width })
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [signals])

  return <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
}
