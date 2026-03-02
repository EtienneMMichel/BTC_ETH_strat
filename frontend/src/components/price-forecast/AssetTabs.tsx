interface AssetTabsProps {
  assets: string[]
  active: string
  onSelect: (asset: string) => void
}

export function AssetTabs({ assets, active, onSelect }: AssetTabsProps) {
  if (assets.length <= 1) return null
  return (
    <div className="flex gap-1">
      {assets.map((a) => (
        <button
          key={a}
          onClick={() => onSelect(a)}
          className={`px-3 py-1 rounded text-xs font-semibold transition-colors ${
            active === a
              ? 'bg-sky-600 text-white'
              : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
          }`}
        >
          {a}
        </button>
      ))}
    </div>
  )
}
