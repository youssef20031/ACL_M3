import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';
import { ClubCrest, PlayerAvatar } from './Identity';
import { getClubAbbreviation } from '../../utils/clubLogos';

/** Green striped pitch with markings drawn in CSS/SVG. Children are PitchRow lines, GK first. */
export function Pitch({ children, label, className }: { children: ReactNode; label: string; className?: string }) {
  return (
    <section
      aria-label={label}
      className={cn('relative overflow-hidden rounded-lg border border-line', className)}
      style={{
        background:
          'repeating-linear-gradient(180deg, rgb(var(--pitch-a)) 0 48px, rgb(var(--pitch-b)) 48px 96px)',
      }}
    >
      <svg
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 h-full w-full"
        viewBox="0 0 100 140"
        preserveAspectRatio="none"
      >
        <g fill="none" stroke="rgb(255 255 255 / 0.28)" strokeWidth="0.5" vectorEffect="non-scaling-stroke">
          <rect x="3" y="3" width="94" height="134" />
          <line x1="3" y1="70" x2="97" y2="70" />
          <circle cx="50" cy="70" r="11" />
          <rect x="24" y="3" width="52" height="20" />
          <rect x="38" y="3" width="24" height="7" />
          <rect x="24" y="117" width="52" height="20" />
          <rect x="38" y="130" width="24" height="7" />
        </g>
      </svg>
      <div className="relative flex min-h-[30rem] flex-col justify-around gap-3 px-2 py-5 sm:min-h-[38rem] sm:py-8">
        {children}
      </div>
    </section>
  );
}

export function PitchRow({ children, label }: { children: ReactNode; label?: string }) {
  return (
    <ul aria-label={label} className="flex items-start justify-center gap-1 sm:gap-4">
      {children}
    </ul>
  );
}

export interface PitchOpponent {
  team: string;
  home: boolean;
}

interface PitchPlayerProps {
  name: string;
  /** The player's club: crest on the photo. */
  team?: string | null;
  /** Who they played (or play). Several entries for a double gameweek. */
  opponents?: PitchOpponent[];
  /** Bottom strip, e.g. "12 pts". */
  value?: ReactNode;
  photo?: string | null;
  captain?: boolean;
  viceCaptain?: boolean;
  /** FPL availability code; doubtful/injured/suspended tint the name plate like FPL does. */
  status?: string | null;
  /** Tooltip/sr text for the status, e.g. the FPL news line. */
  statusNote?: string | null;
  onClick?: () => void;
}

const PLATE: Record<string, string> = {
  d: 'bg-warning text-accent-on',
  i: 'bg-danger text-white',
  s: 'bg-danger text-white',
  u: 'bg-fg-subtle text-accent-on',
};
const STATUS_TEXT: Record<string, string> = { d: 'doubtful', i: 'injured', s: 'suspended', u: 'unavailable' };

function Armband({ letter, label }: { letter: 'C' | 'V'; label: string }) {
  return (
    <span className="absolute -left-1 -top-1 z-[1] flex h-5 w-5 items-center justify-center rounded-full bg-black text-micro font-extrabold leading-none text-white ring-1 ring-white/70 sm:h-6 sm:w-6">
      <span aria-hidden="true">{letter}</span>
      <span className="sr-only">{label}</span>
    </span>
  );
}

/** One player on the pitch: photo with crest and armband, name plate, opponent and value strips. */
export function PitchPlayer({
  name,
  team,
  opponents,
  value,
  photo,
  captain,
  viceCaptain,
  status,
  statusNote,
  onClick,
}: PitchPlayerProps) {
  const shortName = name.split(' ').slice(-1)[0] || name;
  const code = (status ?? '').toLowerCase();
  const plate = PLATE[code] ?? 'bg-fg text-brand';
  const Inner = onClick ? 'button' : 'div';

  return (
    <li className="w-[4.6rem] sm:w-[6.25rem]">
      <Inner
        {...(onClick ? { type: 'button' as const, onClick } : {})}
        className="relative flex w-full flex-col items-center text-center"
        title={statusNote ? `${name}: ${statusNote}` : name}
      >
        <span className="relative">
          <PlayerAvatar name={name} src={photo} size={56} className="rounded-b-none bg-transparent sm:h-[4.5rem] sm:w-[4.5rem]" />
          {captain && <Armband letter="C" label="Captain" />}
          {!captain && viceCaptain && <Armband letter="V" label="Vice-captain" />}
          {team && (
            <span className="absolute -right-1.5 -top-1 rounded-full bg-white/90 p-0.5 shadow-card">
              <ClubCrest team={team} size={16} />
            </span>
          )}
        </span>

        <span className={cn('w-full truncate rounded-t-sm px-1 py-0.5 text-micro font-bold', plate)}>
          <span aria-hidden="true">{shortName}</span>
          <span className="sr-only">
            {name}
            {STATUS_TEXT[code] ? ` (${STATUS_TEXT[code]}${statusNote ? `: ${statusNote}` : ''})` : ''}
          </span>
        </span>

        {opponents && opponents.length > 0 && (
          <span className="flex w-full flex-wrap items-center justify-center gap-x-1 bg-brand-2 px-1 py-0.5 text-micro font-semibold text-fg">
            {opponents.map((o, i) => (
              <span key={`${o.team}-${i}`} className="inline-flex items-center gap-0.5">
                <span className="sr-only">{i === 0 ? 'versus ' : 'and '}</span>
                <ClubCrest team={o.team} size={12} className="hidden sm:inline-flex" />
                {getClubAbbreviation(o.team) || o.team.slice(0, 3).toUpperCase()}
                <span aria-hidden="true">({o.home ? 'H' : 'A'})</span>
                <span className="sr-only">{o.home ? ' at home' : ' away'}</span>
              </span>
            ))}
          </span>
        )}

        {value != null && (
          <span className="w-full truncate rounded-b-sm bg-brand px-1 py-0.5 text-micro font-bold text-fg">{value}</span>
        )}
      </Inner>
    </li>
  );
}
