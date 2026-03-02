import { Label } from '@/components/ui/label'

export interface DateRange {
  min: string
  max: string
  total_bars?: number
}

interface DateSplitPanelProps {
  enabled: boolean
  onToggle: (v: boolean) => void
  trainStart: string
  trainEnd: string
  testStart: string
  testEnd: string
  dataRange?: DateRange
  onChange: (field: 'trainStart' | 'trainEnd' | 'testStart' | 'testEnd', value: string) => void
  onSetAll: (ts: string, te: string, ss: string, se: string) => void
}

// Add N months to a "YYYY-MM-DD" string
function addDays(date: string, days: number): string {
  const d = new Date(date)
  d.setUTCDate(d.getUTCDate() + days)
  return d.toISOString().slice(0, 10)
}

function subDays(date: string, days: number): string {
  return addDays(date, -days)
}

function daysBetween(a: string, b: string): number {
  return Math.round((new Date(b).getTime() - new Date(a).getTime()) / 86_400_000)
}

export function DateSplitPanel({
  enabled,
  onToggle,
  trainStart,
  trainEnd,
  testStart,
  testEnd,
  dataRange,
  onChange,
  onSetAll,
}: DateSplitPanelProps) {
  const minDate = dataRange?.min ?? ''
  const maxDate = dataRange?.max ?? ''

  // Preset: last N days as test, rest as train
  const applyLastN = (testDays: number) => {
    if (!minDate || !maxDate) return
    const tStart = subDays(maxDate, testDays - 1)
    const tEnd = maxDate
    const trEnd = subDays(tStart, 1)
    onSetAll(minDate, trEnd, tStart, tEnd)
  }

  // Preset: split the full range at a given fraction
  const applySplit = (trainFrac: number) => {
    if (!minDate || !maxDate) return
    const total = daysBetween(minDate, maxDate)
    const splitDay = Math.floor(total * trainFrac)
    const trEnd = addDays(minDate, splitDay - 1)
    const tStart = addDays(minDate, splitDay)
    onSetAll(minDate, trEnd, tStart, maxDate)
  }

  // SVG timeline — positions relative to the full data range
  const rangeMs = minDate && maxDate ? new Date(maxDate).getTime() - new Date(minDate).getTime() : 0

  const toFrac = (d: string) => {
    if (!minDate || !maxDate || !d || rangeMs === 0) return null
    return Math.max(0, Math.min(1, (new Date(d).getTime() - new Date(minDate).getTime()) / rangeMs))
  }

  const trS = toFrac(trainStart)
  const trE = toFrac(trainEnd)
  const teS = toFrac(testStart)
  const teE = toFrac(testEnd)

  const showBar = trS !== null && trE !== null && teS !== null && teE !== null && trE > trS && teE > teS

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <input
          type="checkbox"
          id="date-split-toggle"
          checked={enabled}
          onChange={(e) => onToggle(e.target.checked)}
          className="accent-sky-500"
        />
        <Label htmlFor="date-split-toggle" className="text-xs text-slate-300 cursor-pointer">
          Date-based split
        </Label>
      </div>

      {enabled && (
        <div className="space-y-2 pl-1">
          {/* Available range hint */}
          {dataRange && (
            <p className="text-[9px] text-slate-500">
              Available: <span className="text-slate-400">{dataRange.min}</span>
              {' → '}
              <span className="text-slate-400">{dataRange.max}</span>
              {dataRange.total_bars != null && (
                <span className="text-slate-600"> ({dataRange.total_bars} bars)</span>
              )}
            </p>
          )}

          {/* Quick presets */}
          {minDate && maxDate && (
            <div className="flex flex-wrap gap-1">
              <span className="text-[9px] text-slate-500 self-center">Quick:</span>
              {[
                { label: '2/3 ÷ 1/3', action: () => applySplit(2 / 3) },
                { label: 'Last 12m', action: () => applyLastN(365) },
                { label: 'Last 6m', action: () => applyLastN(182) },
                { label: 'Last 3m', action: () => applyLastN(91) },
              ].map(({ label, action }) => (
                <button
                  key={label}
                  onClick={action}
                  className="text-[9px] px-1.5 py-0.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded transition-colors"
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          {/* 4 date inputs */}
          <div className="grid grid-cols-2 gap-x-2 gap-y-1.5">
            <div>
              <Label className="text-[10px] text-blue-400">Train start</Label>
              <input
                type="date"
                value={trainStart}
                min={minDate}
                max={trainEnd || maxDate}
                onChange={(e) => onChange('trainStart', e.target.value)}
                className="w-full text-xs bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
              />
            </div>
            <div>
              <Label className="text-[10px] text-blue-400">Train end</Label>
              <input
                type="date"
                value={trainEnd}
                min={trainStart || minDate}
                max={testStart || maxDate}
                onChange={(e) => onChange('trainEnd', e.target.value)}
                className="w-full text-xs bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
              />
            </div>
            <div>
              <Label className="text-[10px] text-emerald-400">Test start</Label>
              <input
                type="date"
                value={testStart}
                min={trainEnd || minDate}
                max={testEnd || maxDate}
                onChange={(e) => onChange('testStart', e.target.value)}
                className="w-full text-xs bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
              />
            </div>
            <div>
              <Label className="text-[10px] text-emerald-400">Test end</Label>
              <input
                type="date"
                value={testEnd}
                min={testStart || minDate}
                max={maxDate}
                onChange={(e) => onChange('testEnd', e.target.value)}
                className="w-full text-xs bg-slate-800 border border-slate-700 rounded px-1 py-0.5 text-slate-300"
              />
            </div>
          </div>

          {/* SVG timeline — full data range as backdrop */}
          <div className="space-y-0.5">
            <svg width="100%" height="20">
              {/* Full range backdrop */}
              <rect x="0" y="6" width="100%" height="8" fill="#1e293b" rx="2" />
              {/* Train block */}
              {showBar && (
                <rect
                  x={`${trS! * 100}%`}
                  y="4"
                  width={`${(trE! - trS!) * 100}%`}
                  height="12"
                  fill="#38bdf8"
                  rx="2"
                />
              )}
              {/* Gap ticks */}
              {showBar && teS! > trE! && (
                <rect
                  x={`${trE! * 100}%`}
                  y="6"
                  width={`${(teS! - trE!) * 100}%`}
                  height="8"
                  fill="#334155"
                />
              )}
              {/* Test block */}
              {showBar && (
                <rect
                  x={`${teS! * 100}%`}
                  y="4"
                  width={`${(teE! - teS!) * 100}%`}
                  height="12"
                  fill="#34d399"
                  rx="2"
                />
              )}
              {/* Min / Max labels */}
              {minDate && (
                <text x="1" y="19" fontSize="7" fill="#475569">{minDate.slice(0, 7)}</text>
              )}
              {maxDate && (
                <text x="99%" y="19" fontSize="7" fill="#475569" textAnchor="end">{maxDate.slice(0, 7)}</text>
              )}
            </svg>
            <div className="flex gap-3 text-[9px]">
              {showBar && (
                <>
                  <span className="text-blue-400">
                    ■ Train ({daysBetween(trainStart, trainEnd) + 1}d)
                  </span>
                  <span className="text-emerald-400">
                    ■ Test ({daysBetween(testStart, testEnd) + 1}d)
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
