import { getClubLogoPath, getClubAbbreviation } from '../utils/clubLogos';
import { cn } from '../utils/cn';

interface PlayerWithClubProps {
  playerName: string;
  teamName?: string | null;
  showAbbreviation?: boolean;
  logoSize?: 'sm' | 'md' | 'lg' | 'xl' | '2xl';
  className?: string;
  nameClassName?: string;
  clubClassName?: string;
}

/**
 * Component to display a player name with their club logo and abbreviation
 */
export function PlayerWithClub({
  playerName,
  teamName,
  showAbbreviation = true,
  logoSize = 'md',
  className,
  nameClassName,
  clubClassName,
}: PlayerWithClubProps) {
  const logoPath = getClubLogoPath(teamName);
  const abbreviation = getClubAbbreviation(teamName);

  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6',
    xl: 'h-10 w-10',
    '2xl': 'h-14 w-14',
  };

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {logoPath && (
        <img
          src={logoPath}
          alt={`${teamName} logo`}
          className={cn('object-contain shrink-0', sizeClasses[logoSize])}
          loading="lazy"
          decoding="async"
          onError={(e) => {
            // Hide image if it fails to load
            (e.target as HTMLImageElement).style.display = 'none';
          }}
        />
      )}
      <span className={nameClassName}>{playerName}</span>
      {showAbbreviation && teamName && (
        <span className={cn('text-xs opacity-60 shrink-0', clubClassName)}>
          ({abbreviation})
        </span>
      )}
    </div>
  );
}
