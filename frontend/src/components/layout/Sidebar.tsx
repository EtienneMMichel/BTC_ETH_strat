import { NavLink } from 'react-router-dom'
import { cn } from '@/lib/utils'

const links = [
  { to: '/', label: 'Backtest' },
  { to: '/evaluation', label: 'Evaluation' },
  { to: '/vol-eval', label: 'Vol Eval' },
  { to: '/price-forecast', label: 'Price Forecast' },
  { to: '/co-mov', label: 'Co-Movement' },
]

export function Sidebar() {
  return (
    <aside className="w-48 shrink-0 border-r border-slate-800 bg-slate-950 flex flex-col py-6 px-3 gap-1">
      <div className="text-xs font-semibold text-slate-500 uppercase tracking-widest px-3 mb-4">
        BTC/ETH Strategy
      </div>
      {links.map(({ to, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            cn(
              'rounded-md px-3 py-2 text-sm font-medium transition-colors',
              isActive
                ? 'bg-slate-800 text-slate-100'
                : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
            )
          }
        >
          {label}
        </NavLink>
      ))}
    </aside>
  )
}
