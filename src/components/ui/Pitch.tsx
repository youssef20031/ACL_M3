import type { ReactNode } from 'react';
import { cn } from '../../utils/cn';
import { PlayerAvatar } from './Identity';

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
      <div className="relative flex min-h-[28rem] flex-col justify-around gap-3 px-2 py-5 sm:min-h-[34rem] sm:py-8">
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

interface PitchPlayerProps {
  name: string;
  /** Short club/fixture line, e.g. "ARS (H)". */
  subtitle?: ReactNode;
  /** Value under the plate, e.g. predicted points. */
  value?: ReactNode;
  photo?: string | null;
  captain?: boolean;
  viceCaptain?: boolean;
  /** Small corner flag (e.g. a StatusBadge) for injury/doubt. */
  flag?: ReactNode;
  onClick?: () => void;
}

/** One player on the pitch: photo, name plate and value strip. */
export function PitchPlayer({ name, subtitle, value, photo, captain, viceCaptain, flag, onClick }: PitchPlayerProps) {
  const shortName = name.split(' ').slice(-1)[0] || name;
  const Inner = onClick ? 'button' : 'div';
  return (
    <li className="w-[4.6rem] sm:w-24">
      <Inner
        {...(onClick ? { type: 'button' as const, onClick } : {})}
        className="relative flex w-full flex-col items-center text-center"
        title={name}
      >
        <span className="relative">
          <PlayerAvatar name={name} src={photo} size={52} className="rounded-b-none bg-transparent sm:h-16 sm:w-16" />
          {(captain || viceCaptain) && (
            <span
              className={cn(
                'absolute -left-1.5 -top-1 flex h-5 w-5 items-center justify-center rounded-full text-micro font-extrabold',
                captain ? 'bg-fg text-brand' : 'bg-surface-2 text-fg',
              )}
            >
              {captain ? 'C' : 'V'}
              <span className="sr-only">{captain ? ' captain' : ' vice-captain'}</span>
            </span>
          )}
          {flag && <span className="absolute -right-2 -top-1">{flag}</span>}
        </span>
        <span className="w-full truncate rounded-t-sm bg-fg px-1 py-0.5 text-micro font-bold text-brand">
          <span aria-hidden="true">{shortName}</span>
          <span className="sr-only">{name}</span>
        </span>
        {(subtitle || value != null) && (
          <span className="w-full truncate rounded-b-sm bg-brand px-1 py-0.5 text-micro font-semibold text-fg">
            {value ?? subtitle}
          </span>
        )}
      </Inner>
    </li>
  );
}
