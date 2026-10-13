import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';

export interface Option<T extends string> {
  key: T;
  label: string;
  description?: string;
}

interface OptionGroupProps<T extends string> {
  /** Radio group name (unique on the page). */
  name: string;
  legend: ReactNode;
  hint?: ReactNode;
  options: readonly Option<T>[];
  value: string;
  onChange: (value: T) => void;
  /** Grid columns from sm up. */
  columns?: 1 | 2 | 3;
  className?: string;
}

const COLUMNS = { 1: '', 2: 'sm:grid-cols-2', 3: 'sm:grid-cols-3' } as const;

/** Radio cards: a native radio group (arrow keys, labels) styled as selectable options. */
export function OptionGroup<T extends string>({
  name,
  legend,
  hint,
  options,
  value,
  onChange,
  columns = 1,
  className,
}: OptionGroupProps<T>) {
  return (
    <fieldset className={cn('min-w-0', className)}>
      <legend className="text-micro font-semibold uppercase text-fg-muted">{legend}</legend>
      {hint && <p className="mt-1 text-data text-fg-subtle">{hint}</p>}
      <div className={cn('mt-2 grid grid-cols-1 gap-2', COLUMNS[columns])}>
        {options.map((o) => {
          const selected = value === o.key;
          return (
            <label
              key={o.key}
              className={cn(
                'flex cursor-pointer items-start gap-3 rounded-md border px-3 py-2.5 transition-colors',
                'has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-accent-2',
                selected ? 'border-accent bg-accent/10' : 'border-line bg-surface-2/50 hover:border-line-strong',
              )}
            >
              <input
                type="radio"
                name={name}
                value={o.key}
                checked={selected}
                onChange={() => onChange(o.key)}
                className="mt-0.5 h-4 w-4 shrink-0 accent-accent focus-visible:outline-none"
              />
              <span className="min-w-0">
                <span className="flex flex-wrap items-center gap-1.5 text-sm font-semibold text-fg">
                  {o.label}
                </span>
                {o.description && <span className="mt-0.5 block text-data text-fg-subtle">{o.description}</span>}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
