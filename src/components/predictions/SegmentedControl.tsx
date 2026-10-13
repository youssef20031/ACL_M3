import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  /** Shorter label for narrow screens. */
  shortLabel?: ReactNode;
}

interface SegmentedControlProps<T extends string> {
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  /** Accessible name for the group. */
  label: string;
  className?: string;
}

/**
 * A row of toggle buttons where exactly one is pressed (a filter, not navigation).
 * Each button reports its state with aria-pressed; the selected one is filled mint.
 */
export function SegmentedControl<T extends string>({ options, value, onChange, label, className }: SegmentedControlProps<T>) {
  return (
    <div
      role="group"
      aria-label={label}
      className={cn('inline-flex max-w-full overflow-x-auto rounded-md border border-line-strong bg-surface-2 p-0.5', className)}
    >
      {options.map((opt) => {
        const selected = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            aria-pressed={selected}
            onClick={() => onChange(opt.value)}
            className={cn(
              'whitespace-nowrap rounded-sm px-3 py-1.5 text-data font-semibold transition-colors',
              selected ? 'bg-accent text-accent-on' : 'text-fg-muted hover:bg-brand-2/60 hover:text-fg',
            )}
          >
            {opt.shortLabel ? (
              <>
                <span className="sm:hidden">{opt.shortLabel}</span>
                <span className="hidden sm:inline">{opt.label}</span>
              </>
            ) : (
              opt.label
            )}
          </button>
        );
      })}
    </div>
  );
}
