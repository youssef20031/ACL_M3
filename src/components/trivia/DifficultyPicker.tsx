import { useId } from 'react';
import { cn } from '../../utils/cn';

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  /** Short line shown under the control for the selected option. */
  description?: string;
}

interface SegmentedControlProps<T extends string> {
  /** Visible group label (rendered as the fieldset legend). */
  label: string;
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  className?: string;
}

/**
 * Segmented control built on native radio inputs, so arrow keys, form semantics
 * and screen-reader announcements come for free. Selected segment is mint and
 * also marked by weight and aria-checked, not colour alone.
 */
export function SegmentedControl<T extends string>({
  label,
  options,
  value,
  onChange,
  disabled,
  className,
}: SegmentedControlProps<T>) {
  const name = useId();
  const descId = `${name}-desc`;
  const selected = options.find((o) => o.value === value);

  return (
    <fieldset className={cn('min-w-0', className)} disabled={disabled} aria-describedby={selected?.description ? descId : undefined}>
      <legend className="mb-1.5 text-micro font-semibold uppercase text-fg-muted">{label}</legend>
      <div className="inline-flex w-full rounded-md border border-line-strong bg-surface-2 p-1 sm:w-auto">
        {options.map((o) => {
          const checked = o.value === value;
          return (
            <label key={o.value} className="relative flex-1 sm:flex-none">
              <input
                type="radio"
                name={name}
                value={o.value}
                checked={checked}
                onChange={() => onChange(o.value)}
                className="peer sr-only"
              />
              <span
                className={cn(
                  'flex h-9 cursor-pointer select-none items-center justify-center rounded-sm px-4 text-sm transition-colors',
                  'peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent-2',
                  'peer-disabled:cursor-not-allowed peer-disabled:opacity-50',
                  checked ? 'bg-accent font-bold text-accent-on' : 'font-medium text-fg-muted hover:bg-brand-2/60 hover:text-fg',
                )}
              >
                {o.label}
              </span>
            </label>
          );
        })}
      </div>
      {selected?.description && (
        <p id={descId} className="mt-2 text-data text-fg-subtle">
          {selected.description}
        </p>
      )}
    </fieldset>
  );
}
