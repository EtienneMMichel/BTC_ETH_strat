import { useEffect, useRef } from 'react'
import {
  createChart,
  LineSeries,
  HistogramSeries,
  ColorType,
  type IChartApi,
  type ISeriesApi,
  type SeriesType,
  type Time,
} from 'lightweight-charts'

type TimeSeries = [string, number][]

interface PriceVolumeChartProps {
  btcClose: TimeSeries
  ethClose: TimeSeries
  btcVolume: TimeSeries
  ethVolume: TimeSeries
  displayMode?: 'price' | 'return'
  onChartMargins?: (left: number, right: number) => void
}

function toLineData(ts: TimeSeries) {
  return [...ts]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([time, value]) => ({ time: time.slice(0, 10) as Time, value }))
    .filter((item, idx, arr) => idx === 0 || item.time !== arr[idx - 1].time)
}

function toCumRetData(ts: TimeSeries) {
  const sorted = [...ts]
    .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    .map(([time, value]) => ({ time: time.slice(0, 10) as Time, value }))
    .filter((item, idx, arr) => idx === 0 || item.time !== arr[idx - 1].time)
  if (sorted.length === 0) return []
  const v0 = sorted[0].value
  if (v0 === 0) return sorted
  return sorted.map((p) => ({ time: p.time, value: (p.value / v0 - 1) * 100 }))
}

export function PriceVolumeChart({
  btcClose,
  ethClose,
  btcVolume,
  ethVolume,
  displayMode = 'return',
  onChartMargins,
}: PriceVolumeChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const btcLineRef = useRef<ISeriesApi<SeriesType> | null>(null)
  const ethLineRef = useRef<ISeriesApi<SeriesType> | null>(null)
  const btcVolRef = useRef<ISeriesApi<SeriesType> | null>(null)
  const ethVolRef = useRef<ISeriesApi<SeriesType> | null>(null)

  // Mount + resize
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
      rightPriceScale: { visible: true },
      leftPriceScale: { visible: true },
    })
    chartRef.current = chart

    // BTC close — left price scale
    btcLineRef.current = chart.addSeries(LineSeries, {
      color: '#f97316',
      lineWidth: 2,
      title: 'BTC',
      priceScaleId: 'left',
    })

    // ETH close — right price scale
    ethLineRef.current = chart.addSeries(LineSeries, {
      color: '#38bdf8',
      lineWidth: 2,
      title: 'ETH',
      priceScaleId: 'right',
    })

    // BTC volume — overlay price scale at bottom 25%
    btcVolRef.current = chart.addSeries(HistogramSeries, {
      color: 'rgba(249,115,22,0.35)',
      title: 'BTC Vol',
      priceScaleId: 'vol',
      priceFormat: { type: 'volume' },
    })
    chart.priceScale('vol').applyOptions({
      scaleMargins: { top: 0.75, bottom: 0 },
    })

    // ETH volume — same overlay scale
    ethVolRef.current = chart.addSeries(HistogramSeries, {
      color: 'rgba(56,189,248,0.35)',
      title: 'ETH Vol',
      priceScaleId: 'vol',
      priceFormat: { type: 'volume' },
    })

    const ro = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) chart.applyOptions({ width: entry.contentRect.width })
    })
    ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      chart.remove()
      chartRef.current = null
      btcLineRef.current = null
      ethLineRef.current = null
      btcVolRef.current = null
      ethVolRef.current = null
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Data updates
  useEffect(() => {
    const converter = displayMode === 'return' ? toCumRetData : toLineData
    btcLineRef.current?.setData(converter(btcClose))
    ethLineRef.current?.setData(converter(ethClose))
    btcVolRef.current?.setData(toLineData(btcVolume))
    ethVolRef.current?.setData(toLineData(ethVolume))
    chartRef.current?.timeScale().fitContent()

    if (!onChartMargins || !containerRef.current) return

    const allData = converter(btcClose)
    if (allData.length === 0) return

    const timer = setTimeout(() => {
      const chart = chartRef.current
      const container = containerRef.current
      if (!chart || !container) return
      const timeScale = chart.timeScale()
      const containerWidth = container.clientWidth
      const firstTime = allData[0].time
      const lastTime = allData[allData.length - 1].time
      const x0 = timeScale.timeToCoordinate(firstTime)
      const xN = timeScale.timeToCoordinate(lastTime)
      if (x0 == null || xN == null) return
      const n = allData.length
      const step = n > 1 ? (xN - x0) / (n - 1) : 0
      const leftPad = Math.max(0, x0 - step / 2)
      const rightPad = Math.max(0, containerWidth - xN - step / 2)
      onChartMargins(leftPad, rightPad)
    }, 50)

    return () => clearTimeout(timer)
  }, [btcClose, ethClose, btcVolume, ethVolume, displayMode, onChartMargins])

  const isReturn = displayMode === 'return'

  return (
    <div className="space-y-1">
      <div className="flex gap-4 text-xs text-slate-400 px-1">
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-orange-500" />
          {isReturn ? 'BTC cumul. %' : 'BTC close'}
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-0.5 bg-sky-400" />
          {isReturn ? 'ETH cumul. %' : 'ETH close'}
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-2 bg-orange-500/35 rounded-sm" />
          BTC volume
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block w-3 h-2 bg-sky-400/35 rounded-sm" />
          ETH volume
        </span>
      </div>
      <div ref={containerRef} className="w-full rounded-lg overflow-hidden" />
    </div>
  )
}
