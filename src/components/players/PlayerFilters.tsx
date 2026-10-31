import { X } from 'lucide-react';
import { cn } from '../../utils/cn';
import { EMPTY_PLAYER_FILTERS, hasPlayerFilters, useClubs, type PlayerFilterValues } from '../../hooks/usePlayerFilters';
import { Button, SearchInput, Select } from '../ui';

const PRICES = [4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12];

interface PlayerFiltersProps {
  value: PlayerFilterValues;
  onChange: (value: PlayerFilterValues) => void;
  /** Season whose clubs the picker lists (default: the current one). */
  season?: string;
  /** Show the name box. */
  search?: boolean;
  /** "Max price" or "Max avg price" (past seasons list each player's average). */
  priceLabel?: string;
  className?: string;
}

/** Club, price and name filters in one row (two columns on phones). */
export function PlayerFilters({ value, onChange, season, search = false, priceLabel = 'Max price', className }: PlayerFiltersProps) {
  const clubs = useClubs(season);
  const set = (patch: Partial<PlayerFilterValues>) => onChange({ ...value, ...patch });
  const clubList = clubs.data ?? [];
  // A club picked for another season stays selectable, so the picker never shows a blank.
  const options = value.team && !clubList.includes(value.team) ? [value.team, ...clubList] : clubList;

  return (
    <div className={cn('grid grid-cols-2 gap-2 sm:flex sm:flex-wrap sm:items-center sm:gap-3', className)}>
      {search && (
        <SearchInput
          value={value.q}
          onChange={(q) => set({ q })}
          label="Filter by player name"
          placeholder="Player name…"
          maxLength={60}
          className="col-span-2 sm:w-56"
        />
      )}
      <label className="flex min-w-0 items-center gap-2">
        <span className="text-micro font-semibold uppercase text-fg-muted">Club</span>
        <Select
          value={value.team ?? ''}
          onChange={(e) => set({ team: e.target.value || null })}
          className="min-w-0 flex-1 sm:w-44 sm:flex-none"
        >
          <option value="">All clubs</option>
          {options.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </Select>
      </label>
      <label className="flex min-w-0 items-center gap-2">
        <span className="whitespace-nowrap text-micro font-semibold uppercase text-fg-muted">{priceLabel}</span>
        <Select
          value={value.maxPrice ?? ''}
          onChange={(e) => set({ maxPrice: e.target.value ? Number(e.target.value) : null })}
          className="min-w-0 flex-1 sm:w-28 sm:flex-none"
        >
          <option value="">Any</option>
          {PRICES.map((p) => (
            <option key={p} value={p}>
              £{p.toFixed(1)}m
            </option>
          ))}
        </Select>
      </label>
      {hasPlayerFilters(value) && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onChange(EMPTY_PLAYER_FILTERS)}
          icon={<X className="h-4 w-4" aria-hidden="true" />}
          className="justify-self-start"
        >
          Clear filters
        </Button>
      )}
    </div>
  );
}
