import { useId, useMemo, useState, type KeyboardEvent, type PointerEvent } from 'react';
import type { ExplainDependence, ExplainDependencePoint } from '../../services/api';
import { cn } from '../../utils/cn';
import { normalizePosition, PositionBadge, type Position } from '../ui';
import { DataDisclosure } from './DataDisclosure';
import { useElementWidth } from './useElementWidth';
import { domainOf, featureValue, linearScale, niceTicks, signed, tickLabel, valueDirection } from './format';

/*
 * SHAP dependence plot: every player-fixture at x = the feature's value, y = its SHAP value,
 * coloured by position. Hover (or tap) a dot for the player; with the plot focused, the arrow
 * keys step through the players from the lowest value to the highest.
 */

type PosKey = Position | 'OTHER';

const POS_ORDER: PosKey[] = ['GK', 'DEF', 'MID', 'FWD', 'OTHER'];
const POS_COLOUR: Record<PosKey, string> = {
  GK: 'rgb(var(--pos-gk))',
  DEF: 'rgb(var(--pos-def))',
  MID: 'rgb(var(--pos-mid))',
  FWD: 'rgb(var(--pos-fwd))',
  OTHER: 'rgb(var(--text-subtle))',
};
const POS_LABEL: Record<PosKey, string> = { GK: 'Goalkeepers', DEF: 'Defenders', MID: 'Midfielders', FWD: 'Forwards', OTHER: 'Other' };

const MARGIN = { top: 12, right: 14, bottom: 46, left: 50 };
const DOT_R = 3.5;

interface Placed extends ExplainDependencePoint {
  value: number;
  pos: PosKey;
  cx: number;
  cy: number;
}

/** Deterministic spread in [-0.5, 0.5) for dots on a discrete value (0/1 flags, fixture counts). */
function jitter(i: number): number {
  return (((Math.imul(i + 1, 2654435761) >>> 0) % 1000) / 1000) - 0.5;
}

export function DependencePlot({ data, className }: { data: ExplainDependence; className?: string }) {
  const [measureRef, width] = useElementWidth<HTMLDivElement>(640);
  const [hidden, setHidden] = useState<Set<PosKey>>(() => new Set());
  const [active, setActive] = useState<number | null>(null);
  const [announce, setAnnounce] = useState('');
  const hintId = useId();
  const height = width < 480 ? 280 : 340;
  const plotW = Math.max(40, width - MARGIN.left - MARGIN.right);
  const plotH = height - MARGIN.top - MARGIN.bottom;

  const known = useMemo(
    () =>
      data.points
        .filter((p): p is ExplainDependencePoint & { value: number } => p.value != null && Number.isFinite(p.value))
        .map((p) => ({ ...p, pos: (normalizePosition(p.position) ?? 'OTHER') as PosKey })),
    [data.points],
  );
  const skipped = data.points.length - known.length;

  const scales = useMemo(() => {
    const values = known.map((p) => p.value);
    const unique = [...new Set(values)].sort((a, b) => a - b);
    // A handful of distinct values (0/1 flags, fixtures in the gameweek): one column per value,
    // half a gap of margin either side, and the dots spread sideways so they don't pile up.
    const discrete = unique.length > 0 && unique.length <= 8;
    const gap = unique.length > 1 ? Math.min(...unique.slice(1).map((v, i) => v - unique[i])) : 1;
    const [lo, hi] = discrete
      ? [unique[0] - gap * 0.5, unique[unique.length - 1] + gap * 0.5]
      : domainOf(values, { includeZero: false, pad: 0.04 });
    const [yLo, yHi] = domainOf(known.map((p) => p.shap), { pad: 0.06 });
    const gapPx = (gap / (hi - lo || 1)) * plotW;
    return {
      x: linearScale(lo, hi, plotW),
      y: (v: number) => plotH - linearScale(yLo, yHi, plotH)(v),
      xTicks: discrete ? unique : niceTicks(lo, hi, width < 480 ? 4 : 6),
      yTicks: niceTicks(yLo, yHi, 5),
      discrete,
      spread: discrete ? Math.min(gapPx * 0.5, 60) : 0,
    };
  }, [known, plotW, plotH, width]);

  const placed: Placed[] = useMemo(
    () =>
      known.map((p, i) => ({
        ...p,
        cx: scales.x(p.value) + (scales.discrete ? jitter(i) * scales.spread : 0),
        cy: scales.y(p.shap),
      })),
    [known, scales],
  );
  const visible = useMemo(() => placed.filter((p) => !hidden.has(p.pos)).sort((a, b) => a.value - b.value || a.shap - b.shap), [placed, hidden]);
  const counts = useMemo(() => {
    const c = new Map<PosKey, number>();
    for (const p of known) c.set(p.pos, (c.get(p.pos) ?? 0) + 1);
    return c;
  }, [known]);
  const positions = POS_ORDER.filter((k) => counts.has(k));
  const current = active != null ? visible[active] : undefined;

  const describe = (p: Placed) =>
    `${p.name}, ${POS_LABEL[p.pos].replace(/s$/, '').toLowerCase()}: ${data.label} ${featureValue(data.feature, p.value)}, SHAP ${signed(p.shap)} points`;

  const toggle = (k: PosKey) => {
    setActive(null);
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(k)) next.delete(k);
      else next.add(k);
      return next;
    });
  };

  const onPointer = (e: PointerEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mx = e.clientX - rect.left - MARGIN.left;
    const my = e.clientY - rect.top - MARGIN.top;
    let best = -1;
    let bestD = 18 * 18;
    visible.forEach((p, i) => {
      const d = (p.cx - mx) ** 2 + (p.cy - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    });
    setActive(best >= 0 ? best : null);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (!visible.length) return;
    let next: number | null = null;
    if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next = active == null ? 0 : Math.min(visible.length - 1, active + 1);
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next = active == null ? 0 : Math.max(0, active - 1);
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = visible.length - 1;
    else if (e.key === 'Escape') {
      setActive(null);
      return;
    } else return;
    e.preventDefault();
    setActive(next);
    setAnnounce(describe(visible[next]));
  };

  const direction = valueDirection(known.map((p) => [p.value, p.shap]));
  const shaps = known.map((p) => p.shap);
  const summary =
    `Dependence plot for ${data.label}: ${known.length} player-fixtures, coloured by position. ` +
    (direction === 'up' ? 'Higher values raise the prediction' : direction === 'down' ? 'Higher values lower the prediction' : 'No simple direction') +
    (shaps.length ? `; SHAP from ${signed(Math.min(...shaps))} to ${signed(Math.max(...shaps))} points.` : '.');

  const xLabel = (t: number) =>
    scales.discrete ? featureValue(data.feature, t) : data.feature === 'price' ? `£${tickLabel(t, scales.xTicks)}m` : tickLabel(t, scales.xTicks);
  const zeroY = scales.y(0);

  return (
    <figure className={cn('min-w-0', className)}>
      {positions.length > 1 && (
        <div className="mb-3 flex flex-wrap items-center gap-2" role="group" aria-label="Show positions">
          {positions.map((k) => {
            const on = !hidden.has(k);
            return (
              <button
                key={k}
                type="button"
                aria-pressed={on}
                onClick={() => toggle(k)}
                className={cn(
                  'inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 text-micro font-semibold transition-colors',
                  on ? 'border-line-strong bg-surface-2 text-fg' : 'border-line bg-transparent text-fg-subtle line-through',
                )}
              >
                <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full" style={{ background: on ? POS_COLOUR[k] : 'transparent', boxShadow: `inset 0 0 0 1.5px ${POS_COLOUR[k]}` }} />
                {POS_LABEL[k]}
                <span className="font-normal text-fg-subtle">{counts.get(k)}</span>
              </button>
            );
          })}
        </div>
      )}

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
            className="block max-w-full"
            onPointerMove={onPointer}
            onPointerDown={onPointer}
            onPointerLeave={() => setActive(null)}
          >
            <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
              {scales.yTicks.map((t) => (
                <g key={`y${t}`}>
                  <line x1={0} x2={plotW} y1={scales.y(t)} y2={scales.y(t)} stroke="rgb(var(--border))" strokeWidth={1} />
                  <text x={-8} y={scales.y(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="rgb(var(--text-subtle))" className="tabular-nums">
                    {tickLabel(t, scales.yTicks)}
                  </text>
                </g>
              ))}
              <line x1={0} x2={plotW} y1={zeroY} y2={zeroY} stroke="rgb(var(--border-strong))" strokeWidth={1.5} strokeDasharray="4 3" />
              <line x1={0} x2={plotW} y1={plotH} y2={plotH} stroke="rgb(var(--border))" strokeWidth={1} />
              {scales.xTicks.map((t) => (
                <text key={`x${t}`} x={scales.x(t)} y={plotH + 16} textAnchor="middle" fontSize={11} fill="rgb(var(--text-subtle))" className="tabular-nums">
                  {xLabel(t)}
                </text>
              ))}
              <text x={plotW / 2} y={plotH + 36} textAnchor="middle" fontSize={12} fontWeight={600} fill="rgb(var(--text-muted))">
                {data.label}
              </text>
              <text
                transform={`translate(${-38},${plotH / 2}) rotate(-90)`}
                textAnchor="middle"
                fontSize={12}
                fontWeight={600}
                fill="rgb(var(--text-muted))"
              >
                SHAP value (points)
              </text>
              {visible.map((p, i) => (
                <circle
                  key={`${p.element}-${i}`}
                  cx={p.cx}
                  cy={p.cy}
                  r={DOT_R}
                  fill={POS_COLOUR[p.pos]}
                  fillOpacity={0.85}
                  stroke="rgb(var(--surface))"
                  strokeWidth={1}
                />
              ))}
              {current && <circle cx={current.cx} cy={current.cy} r={DOT_R + 3} fill="none" stroke="rgb(var(--text))" strokeWidth={2} />}
            </g>
          </svg>
        </div>

        {current && (
          <div
            className="pointer-events-none absolute z-10 w-max max-w-[15rem] -translate-x-1/2 -translate-y-full rounded-md border border-line-strong bg-surface-2 px-2.5 py-1.5 text-micro text-fg shadow-pop"
            style={{
              left: Math.min(Math.max(current.cx + MARGIN.left, 80), width - 80),
              top: current.cy + MARGIN.top - 10,
            }}
          >
            <span className="flex items-center gap-1.5">
              <span className="truncate text-data font-semibold">{current.name}</span>
              <PositionBadge position={current.pos === 'OTHER' ? null : current.pos} className="w-auto px-1" />
            </span>
            <span className="block text-fg-muted">
              {data.label}: <span className="font-semibold text-fg">{featureValue(data.feature, current.value)}</span>
            </span>
            <span className="block text-fg-muted">
              SHAP: <span className="font-semibold text-fg">{signed(current.shap)} pts</span>
            </span>
          </div>
        )}
      </div>

      <p id={hintId} className="mt-2 text-micro text-fg-subtle">
        {skipped > 0 && (
          <>
            {skipped} {skipped === 1 ? 'player has' : 'players have'} no value for this feature and {skipped === 1 ? 'isn’t' : 'aren’t'} shown.{' '}
          </>
        )}
        <span className="hidden sm:inline">Hover a dot for the player, or focus the plot and use the arrow keys.</span>
        <span className="sm:hidden">Tap a dot for the player.</span>
      </p>
      <p aria-live="polite" className="sr-only">
        {announce}
      </p>

      <DataDisclosure summary="Show the numbers">
        <table className="w-full text-left text-data">
          <caption className="sr-only">
            {data.label} and its SHAP value for every player shown, lowest value first
          </caption>
          <thead className="text-micro uppercase text-fg-subtle">
            <tr>
              <th scope="col" className="py-1.5 pr-3 font-semibold">Player</th>
              <th scope="col" className="py-1.5 pr-3 font-semibold">Pos</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">Value</th>
              <th scope="col" className="py-1.5 text-right font-semibold">SHAP</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((p, i) => (
              <tr key={`${p.element}-${i}`} className="border-t border-line">
                <th scope="row" className="py-1 pr-3 font-normal text-fg">{p.name}</th>
                <td className="py-1 pr-3 text-fg-muted">{p.pos === 'OTHER' ? p.position : p.pos}</td>
                <td className="whitespace-nowrap py-1 pr-3 text-right tabular-nums text-fg-muted">{featureValue(data.feature, p.value)}</td>
                <td className="py-1 text-right tabular-nums text-fg-muted">{signed(p.shap)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </DataDisclosure>
    </figure>
  );
}
