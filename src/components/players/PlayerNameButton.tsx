import { usePlayerDialog } from '../../hooks/usePlayerDialog';
import { cn } from '../../utils/cn';
import type { PlayerHint } from './PlayerProfileView';

/** A player's name that opens his stats overlay (plain text outside the app shell). */
export function PlayerNameButton({ player, className, season }: { player: PlayerHint; className?: string; season?: string }) {
  const dialog = usePlayerDialog();
  if (!dialog) return <span className={className}>{player.name}</span>;
  return (
    <button
      type="button"
      onClick={() => dialog.openPlayer(player, season)}
      title={`${player.name}: show stats`}
      className={cn('max-w-full truncate text-left hover:text-accent hover:underline focus-visible:text-accent', className)}
    >
      {player.name}
    </button>
  );
}
