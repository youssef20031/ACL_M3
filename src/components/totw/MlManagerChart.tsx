import { useId, useMemo, useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { MlManagerHistoryRow } from '../../services/api';
import { cn } from '../../utils/cn';
import { DataDisclosure } from '../explain/DataDisclosure';
import { useElementWidth } from '../explain/useElementWidth';
import { niceTicks, signed, spokenSigned, tickLabel } from '../explain/format';
import { CHIP_NAMES } from './mlManagerFormat';

/*
 * The ML manager's points per gameweek (bars, after hits) against the average manager's score
 * that gameweek (a line with dots). Drawn at its real width, so text stays legible. Hover or tap a
 * gameweek for its numbers; with the chart focused, the arrow keys step through the gameweeks.
 */

const MARGIN = { top: 22, right: 8, bottom: 26, left: 34 };
const BAR = 'rgb(var(--accent))';
const AVERAGE = 'rgb(var(--text-muted))';

/** A bar from the zero line `y0` to `y1`, with only the data end rounded. */
function barPath(x: number, w: number, y0: number, y1: number, radius: number): string {
  const r = Math.min(radius, w / 2, Math.abs(y1 - y0));
  const d = y1 <= y0 ? r : -r; // up for positive values, down for negative ones
  return `M${x},${y0} V${y1 + d} Q${x},${y1} ${x + r},${y1} H${x + w - r} Q${x + w},${y1} ${x + w},${y1 + d} V${y0} Z`;
}

const diffOf = (r: MlManagerHistoryRow) => (r.gw_average != null ? r.points - r.gw_average : null);

function describe(r: MlManagerHistoryRow): string {
  const diff = diffOf(r);
  return (
    `Gameweek ${r.gameweek}: ${r.points} points` +
    (r.hit_cost > 0 ? ` after a ${r.hit_cost}-point hit` : '') +
    (r.gw_average != null && diff != null ? `, average ${r.gw_average}, ${spokenSigned(diff, 0)} against it` : '')
  );
}

export function PointsChart({ history, className }: { history: MlManagerHistoryRow[]; className?: string }) {
  const [measureRef, width] = useElementWidth<HTMLDivElement>(640);
  const [active, setActive] = useState<number | null>(null);
  const [announce, setAnnounce] = useState('');
  const hintId = useId();
  const height = width < 480 ? 220 : 260;
  const plotW = Math.max(40, width - MARGIN.left - MARGIN.right);
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const n = history.length;

  const scale = useMemo(() => {
    const values = history.flatMap((r) => (r.gw_average != null ? [r.points, r.gw_average] : [r.points]));
    const lo = Math.min(0, ...values);
    const hi = Math.max(10, ...values);
    const rough = niceTicks(lo, hi, 4);
    const step = rough.length > 1 ? rough[1] - rough[0] : 10;
    const yLo = Math.floor(lo / step) * step;
    const yHi = Math.ceil(hi / step) * step;
    const ticks: number[] = [];
    for (let v = yLo; v <= yHi + step * 1e-9; v += step) ticks.push(Number(v.toFixed(6)));
    return { ticks, y: (v: number) => plotH - ((v - yLo) / (yHi - yLo || 1)) * plotH };
  }, [history, plotH]);

  if (n === 0) return null;

  const band = plotW / n;
  const barW = Math.max(4, Math.min(40, band * 0.56));
  const cx = (i: number) => band * (i + 0.5);
  const zeroY = scale.y(0);
  // Every k-th gameweek label, so they never collide; the latest is always labelled.
  const labelEvery = Math.max(1, Math.ceil(n / Math.max(1, Math.floor(plotW / 40))));
  const best = history.reduce((b, r, i) => (r.points > history[b].points ? i : b), 0);
  const labelled = new Set([best, n - 1]);

  const averagePath = history
    .map((r, i) => (r.gw_average == null ? null : `${cx(i).toFixed(1)},${scale.y(r.gw_average).toFixed(1)}`))
    .reduce<string>((path, point, i, all) => (point == null ? path : `${path}${i > 0 && all[i - 1] != null ? 'L' : 'M'}${point}`), '');

  const current = active != null ? history[active] : undefined;
  const withAverage = history.filter((r) => r.gw_average != null);
  const above = withAverage.filter((r) => r.points > (r.gw_average ?? 0)).length;
  const summary =
    `Points per gameweek from GW${history[0].gameweek} to GW${history[n - 1].gameweek} as bars, with the average manager’s score as a line. ` +
    (withAverage.length ? `Above the average in ${above} of ${withAverage.length} gameweeks. ` : '') +
    `Best: GW${history[best].gameweek} with ${history[best].points} points.`;

  const onPointer = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left - MARGIN.left;
    const i = Math.floor(mx / band);
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
    setAnnounce(describe(history[next]));
  };

  const tipTop = current ? Math.min(scale.y(current.points), current.gw_average != null ? scale.y(current.gw_average) : Infinity) : 0;
  const currentDiff = current ? diffOf(current) : null;

  return (
    <figure className={cn('min-w-0', className)}>
      <ul className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-micro font-semibold text-fg-muted">
        <li className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="h-2.5 w-2.5 rounded-[2px] bg-accent" />
          ML manager
        </li>
        <li className="inline-flex items-center gap-1.5">
          <svg aria-hidden="true" width="18" height="10" viewBox="0 0 18 10">
            <line x1="0" x2="18" y1="5" y2="5" stroke={AVERAGE} strokeWidth={2} />
            <circle cx="9" cy="5" r="3.5" fill={AVERAGE} stroke="rgb(var(--surface))" strokeWidth={1.5} />
          </svg>
          Gameweek average
        </li>
      </ul>

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

              {history.map((r, i) => (
                <path
                  key={`bar${r.gameweek}`}
                  d={barPath(cx(i) - barW / 2, barW, zeroY, scale.y(r.points), 4)}
                  fill={BAR}
                  fillOpacity={active == null || active === i ? 1 : 0.55}
                />
              ))}
              {history.map((r, i) =>
                labelled.has(i) && barW >= 14 ? (
                  <text
                    key={`label${r.gameweek}`}
                    x={cx(i)}
                    y={r.points >= 0 ? scale.y(r.points) - 6 : scale.y(r.points) + 14}
                    textAnchor="middle"
                    fontSize={11}
                    fontWeight={700}
                    fill="rgb(var(--text))"
                    className="tabular-nums"
                  >
                    {r.points}
                  </text>
                ) : null,
              )}

              {averagePath && <path d={averagePath} fill="none" stroke={AVERAGE} strokeWidth={2} strokeLinejoin="round" />}
              {history.map((r, i) =>
                r.gw_average != null ? (
                  <circle key={`avg${r.gameweek}`} cx={cx(i)} cy={scale.y(r.gw_average)} r={4} fill={AVERAGE} stroke="rgb(var(--surface))" strokeWidth={2} />
                ) : null,
              )}

              {history.map((r, i) =>
                i % labelEvery === (n - 1) % labelEvery ? (
                  <text
                    key={`x${r.gameweek}`}
                    x={cx(i)}
                    y={plotH + 17}
                    textAnchor="middle"
                    fontSize={11}
                    fill={active === i ? 'rgb(var(--text))' : 'rgb(var(--text-subtle))'}
                    className="tabular-nums"
                  >
                    GW{r.gameweek}
                  </text>
                ) : null,
              )}
            </g>
          </svg>
        </div>

        {current && active != null && (
          <div
            className="pointer-events-none absolute z-10 w-max max-w-[14rem] -translate-x-1/2 -translate-y-full rounded-md border border-line-strong bg-surface-2 px-2.5 py-1.5 text-micro text-fg shadow-pop"
            style={{
              left: Math.min(Math.max(cx(active) + MARGIN.left, 72), width - 72),
              top: Math.max(tipTop + MARGIN.top - 8, 40),
            }}
          >
            <span className="block text-data font-semibold">
              GW{current.gameweek}
              {current.chip && <span className="font-normal text-fg-muted"> · {CHIP_NAMES[current.chip]}</span>}
            </span>
            <span className="block text-fg-muted">
              Points: <span className="font-semibold text-fg">{current.points}</span>
              {current.hit_cost > 0 && <> (after −{current.hit_cost})</>}
            </span>
            {current.gw_average != null && (
              <span className="block text-fg-muted">
                Average: <span className="font-semibold text-fg">{current.gw_average}</span>
                {currentDiff != null && <> ({signed(currentDiff, 0)})</>}
              </span>
            )}
          </div>
        )}
      </div>

      <p id={hintId} className="mt-2 text-micro text-fg-subtle">
        <span className="hidden sm:inline">Hover a gameweek for its numbers, or focus the chart and use the arrow keys.</span>
        <span className="sm:hidden">Tap a gameweek for its numbers.</span>
      </p>
      <p aria-live="polite" className="sr-only">
        {announce}
      </p>

      <DataDisclosure summary="Show the numbers">
        <table className="w-full text-left text-data">
          <caption className="sr-only">The ML manager’s points per gameweek against the gameweek’s average and highest score</caption>
          <thead className="text-micro uppercase text-fg-subtle">
            <tr>
              <th scope="col" className="py-1.5 pr-3 font-semibold">Gameweek</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">Points</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">Average</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">
                <span aria-hidden="true">+/−</span>
                <span className="sr-only">Difference from the average</span>
              </th>
              <th scope="col" className="py-1.5 text-right font-semibold">Highest</th>
            </tr>
          </thead>
          <tbody>
            {history.map((r) => {
              const diff = diffOf(r);
              return (
                <tr key={r.gameweek} className="border-t border-line">
                  <th scope="row" className="py-1 pr-3 font-normal text-fg">GW{r.gameweek}</th>
                  <td className="py-1 pr-3 text-right tabular-nums text-fg">{r.points}</td>
                  <td className="py-1 pr-3 text-right tabular-nums text-fg-muted">{r.gw_average ?? '–'}</td>
                  <td className="py-1 pr-3 text-right tabular-nums text-fg-muted">{diff != null ? signed(diff, 0) : '–'}</td>
                  <td className="py-1 text-right tabular-nums text-fg-muted">{r.gw_highest ?? '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </DataDisclosure>
    </figure>
  );
}
