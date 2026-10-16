import { useState } from 'react';
import { AvatarImage } from '../AvatarImage';
import { getClubAbbreviation, getClubLogoPath } from '../../utils/clubLogos';
import { cn } from '../../utils/cn';

interface ClubCrestProps {
  team: string | null | undefined;
  /** Pixel size of the square crest. */
  size?: number;
  /** Show the club name (or abbreviation) next to the crest. */
  label?: 'none' | 'short' | 'full';
  className?: string;
}

export function ClubCrest({ team, size = 20, label = 'none', className }: ClubCrestProps) {
  const [failed, setFailed] = useState(false);
  const logo = getClubLogoPath(team);
  const abbr = getClubAbbreviation(team) || (team ?? '').slice(0, 3).toUpperCase();

  const crest =
    logo && !failed ? (
      <img
        src={logo}
        alt={label === 'none' ? (team ?? '') : ''}
        width={size}
        height={size}
        loading="lazy"
        decoding="async"
        onError={() => setFailed(true)}
        className="shrink-0 object-contain"
        style={{ width: size, height: size }}
      />
    ) : (
      <span
        aria-hidden={label !== 'none'}
        title={team ?? undefined}
        className="inline-flex shrink-0 items-center justify-center rounded-full bg-surface-2 font-bold text-fg-muted"
        style={{ width: size, height: size, fontSize: Math.max(8, size * 0.32) }}
      >
        {abbr}
      </span>
    );

  if (label === 'none') return <span className={cn('inline-flex', className)}>{crest}</span>;
  return (
    <span className={cn('inline-flex min-w-0 items-center gap-2', className)}>
      {crest}
      <span className="truncate">{label === 'short' ? abbr : team}</span>
    </span>
  );
}

/** Blank head-and-shoulders silhouette for players without a photo. */
function BlankAvatar() {
  return (
    <svg viewBox="0 0 40 40" aria-hidden="true" fill="currentColor" className="h-[88%] w-[88%] text-fg-muted/40">
      <circle cx="20" cy="14.5" r="7.5" />
      <path d="M4.5 40c0-9.2 6.9-15.5 15.5-15.5S35.5 30.8 35.5 40z" />
    </svg>
  );
}

interface PlayerAvatarProps {
  name: string;
  src?: string | null;
  size?: number;
  className?: string;
  loading?: 'lazy' | 'eager';
}

/** Player photo in a rounded tile; falls back to a blank silhouette while missing or broken. */
export function PlayerAvatar({ name, src, size = 40, className, loading }: PlayerAvatarProps) {
  return (
    <span
      className={cn('inline-flex shrink-0 items-end justify-center overflow-hidden rounded-md bg-surface-2', className)}
      style={{ width: size, height: size }}
    >
      <AvatarImage
        src={src}
        alt={name}
        size={size}
        loading={loading}
        className="h-full w-full object-cover object-top"
        fallback={<BlankAvatar />}
      />
    </span>
  );
}
