import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../services/api';

/** The table filters beyond position: one club, a highest price and (optionally) a name. */
export interface PlayerFilterValues {
  team: string | null;
  maxPrice: number | null;
  q: string;
}

export const EMPTY_PLAYER_FILTERS: PlayerFilterValues = { team: null, maxPrice: null, q: '' };

export const hasPlayerFilters = (f: PlayerFilterValues) => f.team != null || f.maxPrice != null || f.q.trim() !== '';

/** The clubs of a season (A-Z), shared by every table's club picker. */
export function useClubs(season?: string) {
  return useQuery({
    queryKey: ['clubs', season ?? 'current'],
    queryFn: () => apiService.getClubs(season),
    staleTime: 60 * 60 * 1000,
    select: (d) => d.clubs,
  });
}

/** Debounces the name box so a request isn't sent per keystroke. */
export function useDebounced<T>(value: T, ms = 300): T {
  const [out, setOut] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setOut(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return out;
}
