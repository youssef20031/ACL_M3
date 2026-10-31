import { useState } from 'react';
import type { PlayerProfile } from '../../services/api';
import { getClubAbbreviation } from '../../utils/clubLogos';
import { cn } from '../../utils/cn';

const CHART_HEIGHT = 140;

interface Bar {
  key: string;
  /** Axis label, e.g. "5" or "24-25". */
  label: string;
  points: number;
  /** Tooltip lines under the title. */
  lines: string[];
  title: string;
  /** Didn't play (gameweek) — drawn as a faint stub. */
  absent?: boolean;
}

function gameweekBars(gws: PlayerProfile['gameweeks']): Bar[] {
  return gws.map((g) => {
    const opponents = g.opponents
      .map((o) => `${getClubAbbreviation(o.team) || o.team} (${o.home ? 'H' : 'A'})${o.score ? ` ${o.score}` : ''}`)
      .join(', ');
    const lines = [g.minutes > 0 ? `${g.minutes} min` : "Didn't play"];
    if (g.goals) lines.push(`${g.goals} goal${g.goals > 1 ? 's' : ''}`);
    if (g.assists) lines.push(`${g.assists} assist${g.assists > 1 ? 's' : ''}`);
    return {
      key: String(g.gameweek),
      label: String(g.gameweek),
      points: g.points,
      title: `GW${g.gameweek}${opponents ? ` · ${opponents}` : ''}`,
      lines,
      absent: g.minutes === 0,
    };
  });
}

function seasonBars(seasons: PlayerProfile['seasons']): Bar[] {
  return seasons.map((s) => ({
    key: s.season,
    label: s.season.slice(2),
    points: s.points,
    title: `${s.season}${s.team ? ` · ${s.team}` : ''}`,
    lines: [`${s.appearances} apps`, `${s.goals} G · ${s.assists} A`],
  }));
}

/** Points per gameweek (one season) or per season (career) as columns, with a hover readout. */
export function PointsTimeline({ profile }: { profile: PlayerProfile }) {
  const career = profile.gameweeks.length === 0;
  const bars = career ? seasonBars(profile.seasons) : gameweekBars(profile.gameweeks);
  const [active, setActive] = useState<string | null>(null);
  if (!bars.length) return null;

  const max = Math.max(...bars.map((b) => b.points), 1);
  const min = Math.min(...bars.map((b) => b.points), 0);
  const span = max - min || 1;
  const zero = (max / span) * CHART_HEIGHT; // y of the zero line from the top
  const shown = bars.find((b) => b.key === active) ?? null;
  const best = bars.reduce((a, b) => (b.points > a.points ? b : a), bars[0]);
  const labelEvery = bars.length > 20 ? 5 : bars.length > 10 ? 2 : 1;

  return (
    <figure>
      <div className="mb-2 flex min-h-[2.5rem] items-end justify-between gap-3 text-data" aria-hidden="true">
        {shown ? (
          <span className="min-w-0">
            <span className="block truncate font-semibold text-fg">{shown.title}</span>
            <span className="block truncate text-fg-muted">{shown.lines.join(' · ')}</span>
          </span>
        ) : (
          <span className="text-fg-subtle">
            {career ? 'Hover a season' : 'Hover a gameweek'} for details. Best: {best.title.split(' · ')[0]}, {best.points} pts.
          </span>
        )}
        {shown && (
          <span className="shrink-0 font-display text-h2 font-bold tabular-nums text-accent">{shown.points} pts</span>
        )}
      </div>
      <div
        className="relative flex items-stretch gap-[2px]"
        style={{ height: CHART_HEIGHT }}
        onMouseLeave={() => setActive(null)}
      >
        <div aria-hidden="true" className="pointer-events-none absolute inset-x-0 border-t border-line" style={{ top: zero }} />
        {bars.map((b) => {
          const h = Math.max((Math.abs(b.points) / span) * CHART_HEIGHT, b.absent ? 2 : 3);
          const negative = b.points < 0;
          return (
            <button
              key={b.key}
              type="button"
              aria-label={`${b.title}: ${b.points} points, ${b.lines.join(', ')}`}
              onMouseEnter={() => setActive(b.key)}
              onFocus={() => setActive(b.key)}
              onBlur={() => setActive(null)}
              className="relative min-w-0 flex-1 rounded-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent"
            >
              <span
                className={cn(
                  'absolute inset-x-0 mx-auto max-w-[2.25rem] transition-opacity',
                  negative ? 'rounded-b-[3px] bg-danger' : 'rounded-t-[3px]',
                  !negative && (b.absent ? 'bg-line-strong' : 'bg-accent'),
                  active && active !== b.key && 'opacity-50',
                )}
                style={negative ? { top: zero, height: h } : { top: zero - h, height: h }}
              />
            </button>
          );
        })}
      </div>
      <div aria-hidden="true" className="mt-1 flex gap-[2px] text-micro text-fg-subtle">
        {bars.map((b, i) => (
          <span key={b.key} className="min-w-0 flex-1 text-center tabular-nums">
            {i % labelEvery === 0 || i === bars.length - 1 ? b.label : ''}
          </span>
        ))}
      </div>
      <figcaption className="sr-only">
        {career ? 'Points by season' : 'Points by gameweek'}:{' '}
        {bars.map((b) => `${b.title.split(' · ')[0]} ${b.points}`).join(', ')}
      </figcaption>
    </figure>
  );
}
