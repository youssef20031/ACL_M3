import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';
import { tickLabel } from './format';

/*
 * Horizontal bar rows drawn in HTML, so they scale with their container and every label stays
 * real text. The list is a CSS size container: narrower than 40rem, each row puts its label and
 * value on one line with the bar under them; wider, it's label | bar | value on one line. The
 * columns have fixed widths, so bars line up across rows and share one scale.
 */

// Literal class names (Tailwind can't see names built from strings).
const ROW =
  'grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 ' +
  '[@container(min-width:40rem)]:grid-cols-[minmax(0,14rem)_minmax(0,1fr)_4.5rem]';
const LABEL_CELL = 'col-start-1 row-start-1 min-w-0';
const VALUE_CELL = 'col-start-2 row-start-1 text-right tabular-nums [@container(min-width:40rem)]:col-start-3';
const BAR_CELL =
  'col-span-2 col-start-1 row-start-2 min-w-0 ' +
  '[@container(min-width:40rem)]:col-span-1 [@container(min-width:40rem)]:col-start-2 [@container(min-width:40rem)]:row-start-1';

interface ChartRowsProps {
  /** Accessible name of the list. */
  label: string;
  children: ReactNode;
  className?: string;
}

export function ChartRows({ label, children, className }: ChartRowsProps) {
  return (
    <ol aria-label={label} className={cn('space-y-2.5 [container-type:inline-size]', className)}>
      {children}
    </ol>
  );
}

interface ChartRowProps {
  label: ReactNode;
  value: ReactNode;
  /** Decorative: the label and value carry the information. */
  bar: ReactNode;
  className?: string;
}

export function ChartRow({ label, value, bar, className }: ChartRowProps) {
  return (
    <li className={cn(ROW, className)}>
      <div className={LABEL_CELL}>{label}</div>
      <div className={VALUE_CELL}>{value}</div>
      <div aria-hidden="true" className={BAR_CELL}>
        {bar}
      </div>
    </li>
  );
}

/** Tick labels under the bars, on the same scale. */
export function AxisRow({ ticks, toPercent, caption }: { ticks: number[]; toPercent: (v: number) => number; caption?: string }) {
  return (
    <li aria-hidden="true" className={cn(ROW, 'pt-1')}>
      <div className={BAR_CELL}>
        <div className="relative h-4 border-t border-line text-micro text-fg-subtle">
          {ticks.map((t) => {
            const pct = toPercent(t);
            return (
              <span
                key={t}
                className={cn(
                  'absolute top-0.5 whitespace-nowrap tabular-nums',
                  pct < 4 ? 'translate-x-0' : pct > 96 ? '-translate-x-full' : '-translate-x-1/2',
                )}
                style={{ left: `${pct}%` }}
              >
                {tickLabel(t, ticks)}
              </span>
            );
          })}
        </div>
        {caption && <p className="mt-0.5 text-center text-micro text-fg-subtle">{caption}</p>}
      </div>
    </li>
  );
}

/** Two-line row label: the name, and a quieter detail line under it. */
export function RowLabel({ title, detail, tooltip }: { title: ReactNode; detail?: ReactNode; tooltip?: string }) {
  return (
    <span className="block min-w-0" title={tooltip}>
      <span className="block truncate text-data font-semibold text-fg">{title}</span>
      {detail != null && detail !== '' && <span className="block truncate text-micro text-fg-subtle">{detail}</span>}
    </span>
  );
}

export type BarTone = 'positive' | 'negative' | 'positive-muted' | 'negative-muted' | 'total' | 'base' | 'accent' | 'accent-2' | 'baseline';

const TONES: Record<BarTone, string> = {
  positive: 'bg-accent',
  negative: 'bg-danger',
  'positive-muted': 'bg-accent/45',
  'negative-muted': 'bg-danger/45',
  total: 'bg-fg-muted',
  base: 'bg-fg-subtle/60',
  accent: 'bg-accent',
  'accent-2': 'bg-accent-2',
  baseline: 'bg-fg-subtle/60',
};

interface BarTrackProps {
  /** Position of the zero line, in percent of the track (omit for none). */
  zero?: number;
  children: ReactNode;
  /** Shows the track's background (magnitude bars). */
  filled?: boolean;
}

/** A bar's track: a 10px-high strip, optionally with a zero line. */
export function BarTrack({ zero, children, filled = false }: BarTrackProps) {
  return (
    <div className={cn('relative h-2.5 w-full', filled && 'rounded-full bg-surface-2')}>
      {zero != null && <span className="absolute -bottom-1 -top-1 w-px bg-line-strong" style={{ left: `${zero}%` }} />}
      {children}
    </div>
  );
}

interface BarSpanProps {
  /** Start and end in percent of the track (either order). */
  from: number;
  to: number;
  tone: BarTone;
  /** Which ends are rounded: the data end only (bars from zero) or both (floating bars). */
  rounded?: 'end' | 'both';
}

/** A bar from one point of the track to another, at least 2px wide so tiny values still show. */
export function BarSpan({ from, to, tone, rounded = 'both' }: BarSpanProps) {
  const left = Math.min(from, to);
  const width = Math.abs(to - from);
  const towardsRight = to >= from;
  return (
    <span
      className={cn(
        'absolute inset-y-0',
        TONES[tone],
        rounded === 'both' ? 'rounded-[3px]' : towardsRight ? 'rounded-r-[4px]' : 'rounded-l-[4px]',
      )}
      style={{ left: `${left}%`, width: `max(2px, ${width}%)` }}
    />
  );
}

/** A thin vertical mark at one value (a total on the waterfall). */
export function BarMark({ at }: { at: number }) {
  return <span className="absolute -bottom-1 -top-1 w-[3px] -translate-x-1/2 rounded-full bg-fg" style={{ left: `${at}%` }} />;
}
