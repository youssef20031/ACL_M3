import { useId, useMemo, useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { SiteStatsDay } from '../../services/api';
import { cn } from '../../utils/cn';
import { DataDisclosure } from '../explain/DataDisclosure';
import { useElementWidth } from '../explain/useElementWidth';
import { niceTicks, tickLabel } from '../explain/format';

/*
 * Other people's page views per UTC day (bars). Drawn at its real width, so text stays legible.
 * Hover or tap a day for its views, visitors and the owner's own views; with the chart focused,
 * the arrow keys step through the days.
 */

const MARGIN = { top: 22, right: 8, bottom: 26, left: 34 };
const BAR = 'rgb(var(--accent))';

/** A bar from the zero line `y0` up to `y1`, with only the data end rounded. */
function barPath(x: number, w: number, y0: number, y1: number, radius: number): string {
  const r = Math.min(radius, w / 2, y0 - y1);
  return `M${x},${y0} V${y1 + r} Q${x},${y1} ${x + r},${y1} H${x + w - r} Q${x + w},${y1} ${x + w},${y1 + r} V${y0} Z`;
}

const asDate = (day: string) => new Date(`${day}T00:00:00Z`);
const shortDay = (day: string) =>
  asDate(day).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: 'UTC' });
const longDay = (day: string) =>
  asDate(day).toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });

const plural = (n: number, word: string) => `${n.toLocaleString()} ${word}${n === 1 ? '' : 's'}`;

function describe(d: SiteStatsDay): string {
  return (
    `${longDay(d.date)}: ${plural(d.views, 'view')}, ${plural(d.visitors, 'visitor')}` +
    (d.owner_views > 0 ? `, and ${plural(d.owner_views, 'view')} of your own` : '')
  );
}

export function DailyViewsChart({ days, className }: { days: SiteStatsDay[]; className?: string }) {
  const [measureRef, width] = useElementWidth<HTMLDivElement>(640);
  const [active, setActive] = useState<number | null>(null);
  const [announce, setAnnounce] = useState('');
  const hintId = useId();
  const height = width < 480 ? 200 : 240;
  const plotW = Math.max(40, width - MARGIN.left - MARGIN.right);
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const n = days.length;

  const scale = useMemo(() => {
    const hi = Math.max(4, ...days.map((d) => d.views));
    const rough = niceTicks(0, hi, 4);
    const step = rough.length > 1 ? rough[1] - rough[0] : 1;
    const top = Math.ceil(hi / step) * step;
    const ticks: number[] = [];
    for (let v = 0; v <= top + step * 1e-9; v += step) ticks.push(Number(v.toFixed(6)));
    return { ticks, y: (v: number) => plotH - (v / (top || 1)) * plotH };
  }, [days, plotH]);

  if (n === 0) return null;

  const band = plotW / n;
  const barW = Math.max(1, Math.min(28, band - 2)); // a 2px gap between neighbouring days
  const cx = (i: number) => band * (i + 0.5);
  const zeroY = scale.y(0);
  // Every k-th day labelled so the dates never collide; the latest is always labelled.
  const labelEvery = Math.max(1, Math.ceil(n / Math.max(1, Math.floor(plotW / 56))));
  const busiest = days.reduce((b, d, i) => (d.views > days[b].views ? i : b), 0);
  const total = days.reduce((s, d) => s + d.views, 0);

  const current = active != null ? days[active] : undefined;
  const summary =
    `Views by other people per day from ${longDay(days[0].date)} to ${longDay(days[n - 1].date)}: ${plural(total, 'view')} in all` +
    (total > 0 ? `, the busiest day ${longDay(days[busiest].date)} with ${plural(days[busiest].views, 'view')}.` : '.');

  const onPointer = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const i = Math.floor((e.clientX - rect.left - MARGIN.left) / band);
    setActive(i >= 0 && i < n ? i : null);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    let next: number | null = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next = active == null ? n - 1 : Math.min(n - 1, active + 1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next = active == null ? n - 1 : Math.max(0, active - 1);
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = n - 1;
    else if (e.key === 'Escape') {
      setActive(null);
      return;
    } else return;
    e.preventDefault();
    setActive(next);
    setAnnounce(describe(days[next]));
  };

  return (
    <figure className={cn('min-w-0', className)}>
      <div ref={measureRef} className="relative min-w-0">
        <div
          tabIndex={0}
          aria-describedby={hintId}
          onKeyDown={onKeyDown}
          onBlur={() => setActive(null)}
          className="rounded-md focus-visible:outline-offset-2"
        >
          <svg
            role="img"
            aria-label={summary}
            width={width}
            height={height}
            className="block max-w-full touch-pan-y"
            onPointerMove={onPointer}
            onPointerDown={onPointer}
            onPointerLeave={() => setActive(null)}
          >
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {active != null && <rect x={band * active} y={0} width={band} height={plotH} fill="rgb(var(--surface-2))" rx={4} />}
              {scale.ticks.map((t) => (
                <g key={`y${t}`}>
                  <line x1={0} x2={plotW} y1={scale.y(t)} y2={scale.y(t)} stroke="rgb(var(--border))" strokeWidth={1} />
                  <text x={-8} y={scale.y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="rgb(var(--text-subtle))" className="tabular-nums">
                    {tickLabel(t, scale.ticks)}
                  </text>
                </g>
              ))}
              <line x1={0} x2={plotW} y1={zeroY} y2={zeroY} stroke="rgb(var(--border-strong))" strokeWidth={1} />

              {days.map((d, i) =>
                d.views > 0 ? (
                  <path
                    key={`bar${d.date}`}
                    d={barPath(cx(i) - barW / 2, barW, zeroY, scale.y(d.views), 4)}
                    fill={BAR}
                    fillOpacity={active == null || active === i ? 1 : 0.55}
                  />
                ) : null,
              )}
              {total > 0 && barW >= 14 && (
                <text
                  x={cx(busiest)}
                  y={scale.y(days[busiest].views) - 6}
                  textAnchor="middle"
                  fontSize={11}
                  fontWeight={700}
                  fill="rgb(var(--text))"
                  className="tabular-nums"
                >
                  {days[busiest].views}
                </text>
              )}

              {days.map((d, i) => {
                if (i % labelEvery !== (n - 1) % labelEvery) return null;
                // The latest date ends at the plot's right edge when bars are too narrow to centre it
                const last = i === n - 1 && band < 48;
                return (
                  <text
                    key={`x${d.date}`}
                    x={last ? plotW + MARGIN.right - 1 : cx(i)}
                    y={plotH + 17}
                    textAnchor={last ? 'end' : 'middle'}
                    fontSize={11}
                    fill={active === i ? 'rgb(var(--text))' : 'rgb(var(--text-subtle))'}
                    className="tabular-nums"
                  >
                    {shortDay(d.date)}
                  </text>
                );
              })}
            </g>
          </svg>
        </div>

        {current && active != null && (
          <div
            className="pointer-events-none absolute z-10 w-max max-w-[14rem] -translate-x-1/2 -translate-y-full rounded-md border border-line-strong bg-surface-2 px-2.5 py-1.5 text-micro text-fg shadow-pop"
            style={{
              left: Math.min(Math.max(cx(active) + MARGIN.left, 72), width - 72),
              top: Math.max(scale.y(current.views) + MARGIN.top - 8, 40),
            }}
          >
            <span className="block text-data font-semibold">{longDay(current.date)}</span>
            <span className="block text-fg-muted">
              Views: <span className="font-semibold text-fg">{current.views}</span>
            </span>
            <span className="block text-fg-muted">
              Visitors: <span className="font-semibold text-fg">{current.visitors}</span>
            </span>
            {current.owner_views > 0 && (
              <span className="block text-fg-muted">
                Your own views: <span className="font-semibold text-fg">{current.owner_views}</span>
              </span>
            )}
          </div>
        )}
      </div>

      <p id={hintId} className="mt-2 text-micro text-fg-subtle">
        <span className="hidden sm:inline">Hover a day for its numbers, or focus the chart and use the arrow keys. Days are UTC.</span>
        <span className="sm:hidden">Tap a day for its numbers. Days are UTC.</span>
      </p>
      <p aria-live="polite" className="sr-only">
        {announce}
      </p>

      <DataDisclosure summary="Show the numbers">
        <table className="w-full text-left text-data">
          <caption className="sr-only">Views by other people, visitors who accepted the consent banner, and your own views per day</caption>
          <thead className="text-micro uppercase text-fg-subtle">
            <tr>
              <th scope="col" className="py-1.5 pr-3 font-semibold">Day (UTC)</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">Views</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">Visitors</th>
              <th scope="col" className="py-1.5 text-right font-semibold">Your views</th>
            </tr>
          </thead>
          <tbody>
            {[...days].reverse().map((d) => (
              <tr key={d.date} className="border-t border-line">
                <th scope="row" className="py-1 pr-3 font-normal text-fg">{longDay(d.date)}</th>
                <td className="py-1 pr-3 text-right tabular-nums text-fg">{d.views}</td>
                <td className="py-1 pr-3 text-right tabular-nums text-fg-muted">{d.visitors}</td>
                <td className="py-1 text-right tabular-nums text-fg-muted">{d.owner_views}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </DataDisclosure>
    </figure>
  );
}
