import { cn } from '@/lib/utils'

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'destructive' | 'warning' | 'secondary'
}

const variantClasses = {
  default: 'bg-slate-700 text-slate-200',
  success: 'bg-emerald-900 text-emerald-300',
  destructive: 'bg-rose-900 text-rose-300',
  warning: 'bg-amber-900 text-amber-300',
  secondary: 'bg-sky-900 text-sky-300',
}

export function Badge({ className, variant = 'default', ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold',
        variantClasses[variant],
        className
      )}
      {...props}
    />
  )
}
