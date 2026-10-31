import { cn } from '../../utils/cn';

export interface RankedItem {
  key: string;
  label: string;
  /** A second, quieter line, e.g. a timezone's region. */
  detail?: string;
  value: number;
}

interface RankedListProps {
  items: RankedItem[];
  /** Singular noun for screen readers, e.g. "visitor". */
  unit: string;
  empty: string;
  className?: string;
}

/** A top-N list with each value as a number and a thin bar against the largest. */
export function RankedList({ items, unit, empty, className }: RankedListProps) {
  if (!items.length) return <p className={cn('text-data text-fg-subtle', className)}>{empty}</p>;
  const max = Math.max(...items.map((i) => i.value), 1);
  return (
    <ol className={cn('flex flex-col gap-2.5', className)}>
      {items.map((item) => (
        <li key={item.key} className="min-w-0">
          <div className="flex items-baseline justify-between gap-3">
            <span className="min-w-0 truncate text-data text-fg" title={item.label}>
              {item.label}
              {item.detail && <span className="text-fg-subtle"> · {item.detail}</span>}
            </span>
            <span className="shrink-0 text-data font-semibold tabular-nums text-fg">
              {item.value.toLocaleString()}
              <span className="sr-only"> {unit}{item.value === 1 ? '' : 's'}</span>
            </span>
          </div>
          <div aria-hidden="true" className="mt-1 h-1.5 rounded-full bg-surface-2">
            <div className="h-full rounded-full bg-accent" style={{ width: `${(item.value / max) * 100}%` }} />
          </div>
        </li>
      ))}
    </ol>
  );
}
