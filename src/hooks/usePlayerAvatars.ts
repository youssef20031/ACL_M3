import { useQueries } from '@tanstack/react-query';
import { apiService } from '../services/api';

const AVATAR_STALE_MS = 24 * 60 * 60 * 1000;

/**
 * Looks up avatars for player names that don't already have one.
 *
 * One cached query per name (24h, shared across pages via the ['player-avatar', name]
 * key), so each name is fetched at most once per session; lookups for names that
 * leave the list are aborted. Returns a name -> URL map (undefined while loading,
 * null when there is no photo).
 */
export function usePlayerAvatars(names: string[]): Record<string, string | null | undefined> {
  const unique = Array.from(new Set(names.filter(Boolean)));
  const queries = useQueries({
    queries: unique.map((name) => ({
      queryKey: ['player-avatar', name],
      queryFn: async ({ signal }: { signal: AbortSignal }) => {
        try {
          const result = await apiService.searchImage(name, signal);
          return result.image_url ?? null;
        } catch (err) {
          if (signal.aborted) throw err;
          return null; // no avatar on lookup failure; don't retry
        }
      },
      staleTime: AVATAR_STALE_MS,
      gcTime: AVATAR_STALE_MS,
      retry: false,
    })),
  });

  const byName: Record<string, string | null | undefined> = {};
  unique.forEach((name, i) => {
    byName[name] = queries[i]?.data;
  });
  return byName;
}
