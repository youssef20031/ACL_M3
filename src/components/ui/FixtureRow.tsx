import { cn } from '../../utils/cn';
import { ClubCrest } from './Identity';

export type FixtureStatus = 'scheduled' | 'live' | 'finished' | 'postponed';

interface FixtureRowProps {
  home: string;
  away: string;
  homeScore?: number | null;
  awayScore?: number | null;
  kickoff?: string | Date | null;
  status?: FixtureStatus;
  className?: string;
}

function formatKickoff(kickoff: string | Date | null | undefined): string {
  if (!kickoff) return 'TBC';
  const d = new Date(kickoff);
  if (Number.isNaN(d.getTime())) return 'TBC';
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

/** Home crest/name · score or kickoff · away crest/name, centred on the score like FPL's fixtures list. */
export function FixtureRow({ home, away, homeScore, awayScore, kickoff, status = 'scheduled', className }: FixtureRowProps) {
  const hasScore = homeScore != null && awayScore != null && (status === 'finished' || status === 'live');
  const statusLabel = status === 'finished' ? 'FT' : status === 'live' ? 'Live' : status === 'postponed' ? 'Postponed' : null;

  return (
    <div
      className={cn(
        'grid grid-cols-[1fr_auto_1fr] items-center gap-2 border-b border-line px-2 py-3 last:border-0 sm:gap-4 sm:px-4',
        className,
      )}
    >
      <div className="flex min-w-0 items-center justify-end gap-2 text-right">
        <span className="truncate text-sm font-semibold text-fg">{home}</span>
        <ClubCrest team={home} size={24} />
      </div>

      <div className="flex flex-col items-center">
        {hasScore ? (
          <span
            className={cn(
              'min-w-[4.5rem] rounded-sm px-2 py-1 text-center font-display text-h3 font-bold tabular-nums',
              status === 'live' ? 'bg-accent text-accent-on' : 'bg-brand text-fg',
            )}
          >
            {homeScore}
            <span aria-hidden="true"> – </span>
            <span className="sr-only"> to </span>
            {awayScore}
          </span>
        ) : (
          <span className="min-w-[4.5rem] rounded-sm border border-line px-2 py-1 text-center text-sm font-semibold tabular-nums text-fg-muted">
            {status === 'postponed' ? 'P–P' : formatKickoff(kickoff)}
          </span>
        )}
        {statusLabel && (
          <span className={cn('mt-0.5 text-micro font-semibold uppercase', status === 'live' ? 'text-accent' : 'text-fg-subtle')}>
            {statusLabel}
          </span>
        )}
      </div>

      <div className="flex min-w-0 items-center gap-2">
        <ClubCrest team={away} size={24} />
        <span className="truncate text-sm font-semibold text-fg">{away}</span>
      </div>
    </div>
  );
}
