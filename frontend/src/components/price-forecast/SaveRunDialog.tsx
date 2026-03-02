import { useState } from 'react'
import { savePriceForecastRun } from '@/lib/api/models'
import { Button } from '@/components/ui/button'

interface SaveRunDialogProps {
  jobId: string
}

export function SaveRunDialog({ jobId }: SaveRunDialogProps) {
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)
  const [savedName, setSavedName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSave = async () => {
    if (!name.trim()) return
    setSaving(true)
    setError(null)
    try {
      const meta = await savePriceForecastRun(jobId, name.trim())
      setSavedName(meta.run_name)
      setName('')
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Save failed')
    } finally {
      setSaving(false)
    }
  }

  if (savedName) {
    return (
      <p className="text-xs text-emerald-400">
        Saved as "<span className="font-medium">{savedName}</span>"
      </p>
    )
  }

  return (
    <div className="space-y-1">
      <p className="text-[10px] text-slate-400">Save this run</p>
      <div className="flex gap-1">
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSave()}
          placeholder="Run name…"
          maxLength={128}
          className="flex-1 text-xs bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-300 placeholder:text-slate-600"
        />
        <Button
          size="sm"
          onClick={handleSave}
          disabled={saving || !name.trim()}
          className="text-xs px-2 py-1 h-auto"
        >
          {saving ? '…' : 'Save'}
        </Button>
      </div>
      {error && <p className="text-xs text-rose-400">{error}</p>}
    </div>
  )
}
