import { Clock } from 'lucide-react';
import { Countdown } from '../ui';
import { RepoLink } from './RepoLink';

interface DeadlineBarProps {
  gameweek: number;
  /** ISO timestamp of the FPL deadline. */
  deadline: string | Date;
}

/** Slim strip under the header: next gameweek, its deadline in local time, and a countdown. */
export function DeadlineBar({ gameweek, deadline }: DeadlineBarProps) {
  const date = new Date(deadline);
  const when = date.toLocaleString(undefined, {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  });
  return (
    // One line at every width: on phones the date and the "GitHub" label drop out (the bar wrapped
    // to two lines there, ~65px of chrome on every page).
    <div className="border-b border-line bg-surface">
      <div className="mx-auto flex w-full max-w-page items-center justify-between gap-3 px-4 py-2 sm:gap-4 sm:px-6">
        <p className="flex min-w-0 items-center gap-2 text-data">
          <Clock className="h-4 w-4 shrink-0 text-accent" aria-hidden="true" />
          <span className="whitespace-nowrap font-semibold text-fg">
            <span className="sm:hidden">GW{gameweek} deadline</span>
            <span className="hidden sm:inline">Gameweek {gameweek} deadline</span>
          </span>
          <span className="hidden truncate text-fg-subtle sm:inline">{when}</span>
        </p>
        <div className="flex shrink-0 items-center gap-3 sm:gap-4">
          <Countdown target={date} className="text-data" />
          <span aria-hidden="true" className="h-4 w-px bg-line" />
          <RepoLink label="GitHub" compact />
        </div>
      </div>
    </div>
  );
}
