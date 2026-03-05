export function EyeButton({ visible, onToggle }: { visible: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      title={visible ? 'Hide' : 'Show'}
      className="text-slate-500 hover:text-slate-300 transition-colors text-sm leading-none"
    >
      {visible ? '\u{1F441}' : '\u{1F441}\u200D\u{1F5E8}'}
    </button>
  )
}
