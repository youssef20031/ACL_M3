import { useMemo, useState, type PointerEvent } from 'react';
import type { ExplainBeeswarm } from '../../services/api';
import { cn } from '../../utils/cn';
import { useElementWidth } from './useElementWidth';
import { domainOf, niceTicks, ordinal, points, signed, tickLabel, valueDirection } from './format';
import { DataDisclosure } from './DataDisclosure';

/*
 * SHAP summary plot ("beeswarm"): one row per feature, one dot per player-fixture at x = its SHAP
 * value, coloured by the feature's value (cyan low, grey middle, pink high; a hollow ring when
 * the value is missing). Dots that land on the same spot stack up and down from the row's centre
 * in input order, so the plot is the same on every render.
 */

const ROW_H = 34;
const R = 2.5;
const PAD = R + 1;

/** Feature-value colour, quantised to 11 steps along cyan → grey → pink. */
function colourFor(p: number): string {
  const q = Math.round(Math.min(1, Math.max(0, p)) * 10) / 10;
  if (q <= 0.5) {
    const t = Math.round((q / 0.5) * 100);
    return `color-mix(in oklab, rgb(var(--text-subtle)) ${t}%, rgb(var(--accent-2)))`;
  }
  const t = Math.round(((q - 0.5) / 0.5) * 100);
  return `color-mix(in oklab, rgb(var(--pos-fwd)) ${t}%, rgb(var(--text-subtle)))`;
}

interface Dot {
  cx: number;
  cy: number;
  p: number | null;
  shap: number;
}

function layoutRow(row: ExplainBeeswarm, x: (v: number) => number): Dot[] {
  const binW = R * 2 + 0.5;
  const counts = new Map<number, number>();
  const placed = row.points.map(([p, shap]) => {
    const cx = x(shap);
    const bin = Math.round(cx / binW);
    const k = counts.get(bin) ?? 0;
    counts.set(bin, k + 1);
    return { cx, k, p, shap };
  });
  const maxCount = Math.max(1, ...counts.values());
  const maxHalf = Math.max(1, Math.ceil((maxCount - 1) / 2));
  const spacing = Math.min(R * 1.7, (ROW_H / 2 - R - 1) / maxHalf);
  return placed.map(({ cx, k, p, shap }) => ({
    cx,
    cy: ROW_H / 2 + (k === 0 ? 0 : (k % 2 ? 1 : -1) * Math.ceil(k / 2) * spacing),
    p,
    shap,
  }));
}

function percentileText(p: number | null): string {
  if (p == null) return 'missing';
  const pct = Math.round(p * 100);
  return pct >= 100 ? 'the highest' : pct <= 0 ? 'the lowest' : `${ordinal(pct)} percentile`;
}

const DIRECTION_TEXT = {
  up: 'higher values raise the prediction',
  down: 'higher values lower the prediction',
  mixed: 'no simple direction',
} as const;

// Literal class names for the size-container layout (label above the strip when narrow).
const ROW = 'grid grid-cols-1 gap-x-3 [@container(min-width:34rem)]:grid-cols-[12rem_minmax(0,1fr)] [@container(min-width:34rem)]:items-center';
const STRIP = '[@container(min-width:34rem)]:col-start-2';

interface BeeswarmProps {
  rows: ExplainBeeswarm[];
  /** Feature descriptions, shown as hover text on the labels. */
  descriptions?: Record<string, string>;
  className?: string;
}

export function Beeswarm({ rows, descriptions = {}, className }: BeeswarmProps) {
  const [measureRef, width] = useElementWidth<HTMLDivElement>(560);
  const [active, setActive] = useState<{ row: number; dot: number } | null>(null);

  const [lo, hi] = useMemo(() => domainOf(rows.flatMap((r) => r.points.map((p) => p[1])), { pad: 0.02 }), [rows]);
  const ticks = useMemo(() => niceTicks(lo, hi, width < 400 ? 4 : 6), [lo, hi, width]);
  const x = useMemo(() => {
    const span = hi - lo || 1;
    return (v: number) => PAD + ((v - lo) / span) * Math.max(1, width - 2 * PAD);
  }, [lo, hi, width]);
  const layouts = useMemo(() => rows.map((r) => layoutRow(r, x)), [rows, x]);

  const stats = useMemo(
    () =>
      rows.map((r) => {
        const shaps = r.points.map((p) => p[1]);
        const mean = shaps.length ? shaps.reduce((s, v) => s + Math.abs(v), 0) / shaps.length : 0;
        return {
          mean,
          min: shaps.length ? Math.min(...shaps) : 0,
          max: shaps.length ? Math.max(...shaps) : 0,
          direction: valueDirection(r.points),
          missing: r.points.filter((p) => p[0] == null).length,
        };
      }),
    [rows],
  );

  const summary =
    `SHAP summary plot of the ${rows.length} most important features, one dot per player and fixture. ` +
    rows.map((r, i) => `${r.label}: ${DIRECTION_TEXT[stats[i].direction]}, from ${signed(stats[i].min)} to ${signed(stats[i].max)} points`).join('. ') +
    '.';

  const onMove = (rowIndex: number) => (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = -1;
    let bestD = 14 * 14;
    layouts[rowIndex].forEach((d, i) => {
      const dd = (d.cx - mx) ** 2 + (d.cy - my) ** 2;
      if (dd < bestD) {
        bestD = dd;
        best = i;
      }
    });
    setActive(best >= 0 ? { row: rowIndex, dot: best } : null);
  };

  const zeroX = x(0);

  return (
    <figure className={cn('min-w-0', className)}>
      <div className="mb-3 flex flex-wrap items-center justify-end gap-x-4 gap-y-1 text-micro text-fg-subtle" aria-hidden="true">
        <span className="flex items-center gap-2">
          Feature value
          <span>Low</span>
          <span
            className="h-2 w-20 rounded-full"
            style={{ background: 'linear-gradient(to right, rgb(var(--accent-2)), rgb(var(--text-subtle)), rgb(var(--pos-fwd)))' }}
          />
          <span>High</span>
        </span>
        <span className="flex items-center gap-1.5">
          <svg width="8" height="8" viewBox="0 0 8 8" className="overflow-visible">
            <circle cx="4" cy="4" r={R + 0.5} fill="none" stroke="rgb(var(--text-subtle))" strokeWidth={1} />
          </svg>
          Missing
        </span>
      </div>

      <div role="img" aria-label={summary} className="[container-type:inline-size]">
        <div className="space-y-1">
          {rows.map((row, ri) => {
            const hit = active && active.row === ri ? layouts[ri][active.dot] : null;
            return (
              <div key={row.feature} className={ROW}>
                <div className="min-w-0 truncate pt-1 text-data font-semibold text-fg [@container(min-width:34rem)]:pt-0" title={descriptions[row.feature] ?? row.label}>
                  {row.label}
                </div>
                <div className={cn('relative min-w-0', STRIP)}>
                  <svg
                    width={width}
                    height={ROW_H}
                    className="block max-w-full overflow-visible"
                    onPointerMove={onMove(ri)}
                    onPointerDown={onMove(ri)}
                    onPointerLeave={() => setActive(null)}
                  >
                    <line x1={zeroX} x2={zeroX} y1={0} y2={ROW_H} stroke="rgb(var(--border-strong))" strokeWidth={1} />
                    <line x1={0} x2={width} y1={ROW_H / 2} y2={ROW_H / 2} stroke="rgb(var(--border))" strokeWidth={1} />
                    {layouts[ri].map((d, i) =>
                      d.p == null ? (
                        <circle key={i} cx={d.cx} cy={d.cy} r={R} fill="none" stroke="rgb(var(--text-subtle))" strokeWidth={1} />
                      ) : (
                        <circle key={i} cx={d.cx} cy={d.cy} r={R} style={{ fill: colourFor(d.p) }} fillOpacity={0.9} />
                      ),
                    )}
                    {hit && <circle cx={hit.cx} cy={hit.cy} r={R + 2.5} fill="none" stroke="rgb(var(--text))" strokeWidth={1.5} />}
                  </svg>
                  {hit && (
                    <div
                      className="pointer-events-none absolute z-10 w-max max-w-[14rem] -translate-x-1/2 -translate-y-full rounded-md border border-line-strong bg-surface-2 px-2 py-1 text-micro text-fg shadow-pop"
                      style={{ left: Math.min(Math.max(hit.cx, 70), width - 70), top: hit.cy - 8 }}
                    >
                      <span className="block font-semibold">{signed(hit.shap)} pts</span>
                      <span className="block text-fg-muted">Value: {percentileText(hit.p)}</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Axis: measured strip, the same width as every row's. */}
        <div className={cn(ROW, 'mt-1')} aria-hidden="true">
          <div className={cn('min-w-0', STRIP)} ref={measureRef}>
            <div className="relative h-4 border-t border-line text-micro text-fg-subtle">
              {ticks.map((t) => {
                const px = x(t);
                return (
                  <span
                    key={t}
                    className={cn(
                      'absolute top-0.5 whitespace-nowrap tabular-nums',
                      px < 16 ? 'translate-x-0' : px > width - 16 ? '-translate-x-full' : '-translate-x-1/2',
                      t === 0 && 'font-semibold text-fg-muted',
                    )}
                    style={{ left: px }}
                  >
                    {tickLabel(t, ticks)}
                  </span>
                );
              })}
            </div>
            <p className="mt-1 text-center text-micro text-fg-subtle">SHAP value: points added to (right) or taken from (left) a prediction</p>
          </div>
        </div>
      </div>

      <DataDisclosure summary="Show the numbers">
        <table className="w-full text-left text-data">
          <caption className="sr-only">Impact of the {rows.length} most important features</caption>
          <thead className="text-micro uppercase text-fg-subtle">
            <tr>
              <th scope="col" className="py-1.5 pr-3 font-semibold">Feature</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">Mean impact</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">Range</th>
              <th scope="col" className="py-1.5 font-semibold">Higher values</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.feature} className="border-t border-line">
                <th scope="row" className="py-1.5 pr-3 font-semibold text-fg">{r.label}</th>
                <td className="py-1.5 pr-3 text-right tabular-nums text-fg-muted">{points(stats[i].mean)}</td>
                <td className="whitespace-nowrap py-1.5 pr-3 text-right tabular-nums text-fg-muted">
                  {signed(stats[i].min)} to {signed(stats[i].max)}
                </td>
                <td className="py-1.5 text-fg-muted">
                  {stats[i].direction === 'up' ? 'Raise it' : stats[i].direction === 'down' ? 'Lower it' : 'Mixed'}
                  {stats[i].missing > 0 && <span className="text-fg-subtle"> · {stats[i].missing} missing</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </DataDisclosure>
    </figure>
  );
}
