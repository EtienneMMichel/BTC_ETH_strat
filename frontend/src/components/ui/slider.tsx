import * as RadixSlider from '@radix-ui/react-slider'
import { cn } from '@/lib/utils'

interface SliderProps {
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step?: number
  className?: string
}

export function Slider({ value, onChange, min, max, step = 1, className }: SliderProps) {
  return (
    <RadixSlider.Root
      className={cn('relative flex items-center select-none touch-none w-full h-5', className)}
      value={[value]}
      min={min}
      max={max}
      step={step}
      onValueChange={([v]) => onChange(v!)}
    >
      <RadixSlider.Track className="bg-slate-700 relative grow rounded-full h-1.5">
        <RadixSlider.Range className="absolute bg-sky-500 rounded-full h-full" />
      </RadixSlider.Track>
      <RadixSlider.Thumb
        className="block w-4 h-4 bg-white rounded-full shadow focus:outline-none focus:ring-2 focus:ring-sky-500"
        aria-label="value"
      />
    </RadixSlider.Root>
  )
}
