import type { ReactNode } from 'react';
import { ArrowRight, Check, RotateCcw } from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Button, Card, Chip, ClubCrest, PlayerAvatar, PositionBadge, SectionHeading } from '../ui';
import { cn } from '../../utils/cn';

export type PlayerStats = Record<string, any>;

// Chart colours follow the design tokens (player 1 mint, player 2 cyan).
const P1_COLOR = 'rgb(var(--accent))';
const P2_COLOR = 'rgb(var(--accent-2))';
const GRID_COLOR = 'rgb(var(--border))';
const AXIS_TICK = { fontSize: 11, fill: 'rgb(var(--text-subtle))' };
const TOOLTIP_PROPS = {
  contentStyle: {
    fontSize: 12,
    borderRadius: 8,
    background: 'rgb(var(--surface-2))',
    border: '1px solid rgb(var(--border-strong))',
    color: 'rgb(var(--text))',
    boxShadow: 'var(--shadow-pop)',
  },
  labelStyle: { color: 'rgb(var(--text))', fontWeight: 600, marginBottom: 2 },
};
const LEGEND_STYLE = { fontSize: 12, color: 'rgb(var(--text-muted))' };

type Side = 1 | 2;

interface StatRow {
  label: string;
  p1: number | null;
  p2: number | null;
  format: (v: number) => string;
  /** Whether a higher value counts as better (price is shown but not scored). */
  scored: boolean;
}

const num = (v: unknown): number | null => {
  if (v == null || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

const fixed = (d: number) => (v: number) => v.toFixed(d);
const whole = (v: number) => Math.round(v).toLocaleString();

function pointsPerGame(p: PlayerStats): number | null {
  const pts = num(p.total_points);
  const games = num(p.games);
  return pts != null && games ? pts / games : null;
}

function buildRows(p1: PlayerStats, p2: PlayerStats): StatRow[] {
  const rows: StatRow[] = [
    { label: 'Total points', p1: num(p1.total_points), p2: num(p2.total_points), format: whole, scored: true },
    { label: 'Points per game', p1: pointsPerGame(p1), p2: pointsPerGame(p2), format: fixed(1), scored: true },
    { label: 'Goals', p1: num(p1.goals), p2: num(p2.goals), format: whole, scored: true },
    { label: 'Assists', p1: num(p1.assists), p2: num(p2.assists), format: whole, scored: true },
    { label: 'Clean sheets', p1: num(p1.clean_sheets), p2: num(p2.clean_sheets), format: whole, scored: true },
    { label: 'Bonus points', p1: num(p1.bonus), p2: num(p2.bonus), format: whole, scored: true },
    { label: 'Minutes', p1: num(p1.minutes), p2: num(p2.minutes), format: whole, scored: true },
    { label: 'Games', p1: num(p1.games), p2: num(p2.games), format: whole, scored: true },
    { label: 'Average form', p1: num(p1.avg_form), p2: num(p2.avg_form), format: fixed(2), scored: true },
    { label: 'Average ICT index', p1: num(p1.avg_ict_index), p2: num(p2.avg_ict_index), format: fixed(2), scored: true },
    {
      label: 'Average price',
      p1: num(p1.avg_value_millions),
      p2: num(p2.avg_value_millions),
      format: (v) => `£${v.toFixed(1)}m`,
      scored: false,
    },
  ];
  return rows.filter((r) => r.p1 != null || r.p2 != null);
}

function winner(row: StatRow): Side | null {
  if (!row.scored || row.p1 == null || row.p2 == null || row.p1 === row.p2) return null;
  return row.p1 > row.p2 ? 1 : 2;
}

function StatValue({ value, format, side, better }: { value: number | null; format: StatRow['format']; side: Side; better: boolean }) {
  if (value == null) return <span className="text-fg-subtle">–</span>;
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-sm px-1.5 py-0.5 tabular-nums',
        side === 2 && 'flex-row-reverse',
        better
          ? side === 1
            ? 'bg-accent/15 font-bold text-accent'
            : 'bg-accent-2/15 font-bold text-accent-2'
          : 'font-medium text-fg',
      )}
    >
      {better && <Check aria-hidden="true" className="h-3.5 w-3.5" />}
      {format(value)}
      {better && <span className="sr-only"> (higher)</span>}
    </span>
  );
}

/** Two-sided share bar under each scored stat (decorative; the numbers carry the meaning). */
function ShareBar({ p1, p2 }: { p1: number; p2: number }) {
  const total = Math.abs(p1) + Math.abs(p2);
  if (total <= 0) return null;
  const left = Math.round((Math.abs(p1) / total) * 100);
  return (
    <span aria-hidden="true" className="mx-auto mt-1 flex h-1 w-full max-w-[9rem] gap-0.5 overflow-hidden rounded-full">
      <span className="rounded-l-full bg-accent" style={{ width: `${left}%` }} />
      <span className="flex-1 rounded-r-full bg-accent-2" />
    </span>
  );
}

function PlayerCard({ player, side, onOpenDetails }: { player: PlayerStats; side: Side; onOpenDetails: () => void }) {
  const name: string = player.player_name ?? `Player ${side}`;
  const team: string | undefined = player.team_name ?? player.team;
  const stats: [string, ReactNode][] = [
    ['Points', player.total_points ?? 0],
    ['Games', player.games ?? 0],
  ];

  return (
    <Card padded={false} className="flex min-w-0 flex-col overflow-hidden">
      <div aria-hidden="true" className={cn('h-1', side === 1 ? 'bg-accent' : 'bg-accent-2')} />
      <div className="flex flex-1 flex-col gap-3 p-3 sm:flex-row sm:items-center sm:p-4">
        <PlayerAvatar name={name} src={player.avatar} size={72} loading="eager" />
        <div className="min-w-0">
          <p className={cn('text-micro font-semibold uppercase', side === 1 ? 'text-accent' : 'text-accent-2')}>Player {side}</p>
          <h2 className="truncate font-display text-h3 font-bold text-fg sm:text-h2" title={name}>
            {name}
          </h2>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-data text-fg-muted">
            {team && <ClubCrest team={team} size={18} label="full" className="max-w-full" />}
            <PositionBadge position={player.position} />
            {player.season && <Chip>{player.season}</Chip>}
          </div>
        </div>
      </div>
      <dl className="grid grid-cols-2 divide-x divide-line border-t border-line">
        {stats.map(([label, value]) => (
          <div key={label} className="px-3 py-2 sm:px-4">
            <dt className="text-micro font-semibold uppercase text-fg-subtle">{label}</dt>
            <dd className="font-display text-h3 font-bold tabular-nums text-fg sm:text-h2">{value}</dd>
          </div>
        ))}
      </dl>
      <div className="border-t border-line px-1.5 py-1.5 sm:px-2.5">
        <Button variant="ghost" size="sm" onClick={onOpenDetails} aria-label={`View details for ${name}`}>
          View details
          <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
        </Button>
      </div>
    </Card>
  );
}

interface ComparisonResultsProps {
  p1: PlayerStats;
  p2: PlayerStats;
  onOpenDetails: (player: PlayerStats) => void;
  onChangePlayers: () => void;
}

export function ComparisonResults({ p1, p2, onOpenDetails, onChangePlayers }: ComparisonResultsProps) {
  const p1Name: string = p1.player_name ?? 'Player 1';
  const p2Name: string = p2.player_name ?? 'Player 2';
  const rows = buildRows(p1, p2);
  const wins = rows.reduce(
    (acc, r) => {
      const w = winner(r);
      if (w) acc[w - 1] += 1;
      return acc;
    },
    [0, 0],
  );

  const barStats = [
    { label: 'Points', p1: p1.total_points ?? 0, p2: p2.total_points ?? 0 },
    { label: 'Goals', p1: p1.goals ?? 0, p2: p2.goals ?? 0 },
    { label: 'Assists', p1: p1.assists ?? 0, p2: p2.assists ?? 0 },
    { label: 'Bonus', p1: p1.bonus ?? 0, p2: p2.bonus ?? 0 },
    { label: 'CS', p1: p1.clean_sheets ?? 0, p2: p2.clean_sheets ?? 0 },
  ];

  const radarMetrics = [
    { key: 'total_points', label: 'Points', max: 300 },
    { key: 'goals', label: 'Goals', max: 30 },
    { key: 'assists', label: 'Assists', max: 20 },
    { key: 'bonus', label: 'Bonus', max: 50 },
    { key: 'avg_ict_index', label: 'ICT', max: 10 },
    { key: 'games', label: 'Games', max: 38 },
  ];
  const radarData = radarMetrics.map(({ key, label, max }) => ({
    metric: label,
    [p1Name]: Math.round(((p1[key] ?? 0) / max) * 100),
    [p2Name]: Math.round(((p2[key] ?? 0) / max) * 100),
  }));

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:gap-4">
        <PlayerCard player={p1} side={1} onOpenDetails={() => onOpenDetails(p1)} />
        <PlayerCard player={p2} side={2} onOpenDetails={() => onOpenDetails(p2)} />
      </div>

      <Card padded={false}>
        <div className="px-4 pt-4 sm:px-5">
          <SectionHeading
            as="h2"
            size="md"
            title="Head to head"
            subtitle={
              <>
                Higher value marked with a tick. Price is shown but not scored.{' '}
                <span className="text-fg-muted">
                  {p1Name} leads {wins[0]}, {p2Name} leads {wins[1]}.
                </span>
              </>
            }
          />
        </div>
        <table className="w-full table-fixed border-collapse text-data">
          <caption className="sr-only">
            Stat-by-stat comparison of {p1Name} and {p2Name}
          </caption>
          <thead>
            <tr className="border-y border-line bg-surface-2 text-micro uppercase">
              <th scope="col" className="w-[35%] px-3 py-2 text-left font-semibold text-accent sm:px-5">
                <span className="block truncate">{p1Name}</span>
              </th>
              <th scope="col" className="px-2 py-2 text-center font-semibold text-fg-subtle">Stat</th>
              <th scope="col" className="w-[35%] px-3 py-2 text-right font-semibold text-accent-2 sm:px-5">
                <span className="block truncate">{p2Name}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const w = winner(row);
              return (
                <tr key={row.label} className="border-b border-line last:border-0 hover:bg-surface-2/50">
                  <td className="px-3 py-2 text-left sm:px-5">
                    <StatValue value={row.p1} format={row.format} side={1} better={w === 1} />
                  </td>
                  <th scope="row" className="px-2 py-2 text-center font-normal text-fg-muted">
                    {row.label}
                    {row.scored && row.p1 != null && row.p2 != null && <ShareBar p1={row.p1} p2={row.p2} />}
                  </th>
                  <td className="px-3 py-2 text-right sm:px-5">
                    <StatValue value={row.p2} format={row.format} side={2} better={w === 2} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <SectionHeading as="h2" size="md" title="Totals" subtitle="Season totals side by side" />
          <figure>
            <figcaption className="sr-only">Bar chart of the totals listed in the head-to-head table.</figcaption>
            <div aria-hidden="true">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={barStats} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={GRID_COLOR} vertical={false} />
                  <XAxis dataKey="label" tick={AXIS_TICK} axisLine={{ stroke: GRID_COLOR }} tickLine={false} />
                  <YAxis tick={AXIS_TICK} axisLine={false} tickLine={false} />
                  <Tooltip {...TOOLTIP_PROPS} cursor={{ fill: 'rgb(var(--text) / 0.06)' }} />
                  <Legend wrapperStyle={LEGEND_STYLE} />
                  <Bar dataKey="p1" name={p1Name} fill={P1_COLOR} radius={[3, 3, 0, 0]} />
                  <Bar dataKey="p2" name={p2Name} fill={P2_COLOR} radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </figure>
        </Card>

        <Card>
          <SectionHeading
            as="h2"
            size="md"
            title="Profile"
            subtitle="Normalised: 100 = 300 pts, 30 goals, 20 assists, 50 bonus, 10 ICT, 38 games"
          />
          <figure>
            <figcaption className="sr-only">Radar chart of the same stats scaled to a common range.</figcaption>
            <div aria-hidden="true">
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={radarData} outerRadius="72%">
                  <PolarGrid stroke={GRID_COLOR} />
                  <PolarAngleAxis dataKey="metric" tick={AXIS_TICK} />
                  <Radar name={p1Name} dataKey={p1Name} stroke={P1_COLOR} fill={P1_COLOR} fillOpacity={0.22} />
                  <Radar name={p2Name} dataKey={p2Name} stroke={P2_COLOR} fill={P2_COLOR} fillOpacity={0.18} />
                  <Tooltip {...TOOLTIP_PROPS} />
                  <Legend wrapperStyle={LEGEND_STYLE} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </figure>
        </Card>
      </div>

      <div className="flex justify-center">
        <Button variant="secondary" onClick={onChangePlayers} icon={<RotateCcw className="h-4 w-4" aria-hidden="true" />}>
          Change players
        </Button>
      </div>
    </div>
  );
}
