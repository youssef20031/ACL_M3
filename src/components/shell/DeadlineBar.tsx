import { Clock } from 'lucide-react';
import { Countdown } from '../ui';

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
    <div className="border-b border-line bg-surface">
      <div className="mx-auto flex w-full max-w-page flex-wrap items-center justify-between gap-x-4 gap-y-1 px-4 py-2 sm:px-6">
        <p className="flex items-center gap-2 text-data">
          <Clock className="h-4 w-4 text-accent" aria-hidden="true" />
          <span className="font-semibold text-fg">Gameweek {gameweek} deadline</span>
          <span className="text-fg-subtle">{when}</span>
        </p>
        <Countdown target={date} className="text-data" />
      </div>
    </div>
  );
}
