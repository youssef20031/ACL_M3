import { createContext, useContext } from 'react';
import type { PlayerHint } from '../components/players/PlayerProfileView';

export interface PlayerDialogApi {
  /** Opens the player's stats in an overlay over the current page. */
  openPlayer: (player: PlayerHint, season?: string) => void;
}

export const PlayerDialogContext = createContext<PlayerDialogApi | null>(null);

/** The player overlay from `PlayerDialogProvider` (in AppShell); null outside it, e.g. on /design. */
export function usePlayerDialog(): PlayerDialogApi | null {
  return useContext(PlayerDialogContext);
}
