import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Loader2, Search, X } from 'lucide-react';
import { apiService, type ExplainPlayerHit } from '../../services/api';
import { cn } from '../../utils/cn';
import { ClubCrest, PositionBadge } from '../ui';

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return debounced;
}

/** Top players as quick picks: crest, name and predicted points. */
export function PlayerChips({
  players,
  selected,
  onSelect,
  className,
}: {
  players: ExplainPlayerHit[];
  selected?: number;
  onSelect: (player: ExplainPlayerHit) => void;
  className?: string;
}) {
  if (!players.length) return null;
  return (
    <ul className={cn('flex flex-wrap gap-2', className)} aria-label="Top predicted players">
      {players.map((p) => {
        const active = p.element === selected;
        return (
          <li key={p.element}>
            <button
              type="button"
              aria-pressed={active}
              onClick={() => onSelect(p)}
              className={cn(
                'inline-flex h-8 items-center gap-1.5 rounded-full border px-2.5 text-data font-semibold transition-colors',
                active ? 'border-accent bg-accent/15 text-fg' : 'border-line-strong bg-surface-2 text-fg-muted hover:border-fg-subtle hover:text-fg',
              )}
            >
              <ClubCrest team={p.team} size={16} />
              <span className="max-w-[10rem] truncate">{p.name}</span>
              <span className="tabular-nums text-accent">
                {p.predicted_points.toFixed(1)}
                <span className="sr-only"> predicted points</span>
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

interface PlayerSearchBoxProps {
  onSelect: (player: ExplainPlayerHit) => void;
  className?: string;
}

/**
 * Search-as-you-type over this gameweek's predicted players (min 2 letters, debounced).
 * A combobox: arrow keys move through results, Enter picks, Escape closes.
 */
export function PlayerSearchBox({ onSelect, className }: PlayerSearchBoxProps) {
  const [query, setQuery] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();
  const inputId = useId();
  const wrapRef = useRef<HTMLDivElement>(null);
  const term = useDebounced(query.trim(), 200);
  const searching = term.length >= 2;

  const { data, isFetching, isError } = useQuery({
    queryKey: ['ml-explain-players', term],
    queryFn: ({ signal }) => apiService.searchExplainPlayers(term, 20, signal),
    enabled: searching,
    staleTime: 5 * 60_000,
    retry: false,
  });
  const results = searching ? (data?.players ?? []) : [];
  const expanded = open && searching && results.length > 0;

  useEffect(() => setActive(0), [term]);

  // Close on a click outside.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  const choose = (p: ExplainPlayerHit) => {
    onSelect(p);
    setQuery('');
    setOpen(false);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown' && results.length) {
      e.preventDefault();
      setOpen(true);
      setActive((i) => (i + 1) % results.length);
    } else if (e.key === 'ArrowUp' && results.length) {
      e.preventDefault();
      setOpen(true);
      setActive((i) => (i - 1 + results.length) % results.length);
    } else if (e.key === 'Enter' && expanded && results[active]) {
      e.preventDefault();
      choose(results[active]);
    } else if (e.key === 'Escape') {
      if (open) {
        e.preventDefault();
        setOpen(false);
      } else if (query) {
        setQuery('');
      }
    }
  };

  let status: string | null = null;
  if (query.trim().length > 0 && query.trim().length < 2) status = 'Type at least two letters.';
  else if (searching && isError) status = 'Search is unavailable right now.';
  else if (searching && !isFetching && data && results.length === 0) status = `No player in this gameweek’s predictions matches “${term}”.`;

  return (
    <div ref={wrapRef} className={cn('relative', className)}>
      <label htmlFor={inputId} className="mb-1.5 block text-micro font-semibold uppercase text-fg-muted">
        Find a player
      </label>
      <div className="relative">
        {isFetching ? (
          <Loader2 aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-fg-subtle" />
        ) : (
          <Search aria-hidden="true" className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-subtle" />
        )}
        <input
          id={inputId}
          type="text"
          role="combobox"
          aria-expanded={expanded}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={expanded && results[active] ? `${listId}-${active}` : undefined}
          autoComplete="off"
          maxLength={100}
          value={query}
          placeholder="e.g. Haaland"
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
          className="h-10 w-full rounded-md border border-line-strong bg-surface-2 pl-9 pr-9 text-sm text-fg transition-colors placeholder:text-fg-subtle hover:border-fg-subtle"
        />
        {query && (
          <button
            type="button"
            aria-label="Clear search"
            onClick={() => {
              setQuery('');
              setOpen(false);
            }}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded-sm p-1 text-fg-subtle hover:text-fg"
          >
            <X className="h-4 w-4" aria-hidden="true" />
          </button>
        )}
      </div>

      <ul
        id={listId}
        role="listbox"
        aria-label="Players"
        hidden={!expanded}
        className="absolute inset-x-0 top-full z-20 mt-1 max-h-80 overflow-y-auto rounded-md border border-line bg-surface py-1 shadow-pop"
      >
        {results.map((p, i) => (
          <li
            key={p.element}
            id={`${listId}-${i}`}
            role="option"
            aria-selected={i === active}
            onMouseEnter={() => setActive(i)}
            onMouseDown={(e) => e.preventDefault()} // keep focus in the input
            onClick={() => choose(p)}
            className={cn('flex cursor-pointer items-center gap-3 px-3 py-2', i === active && 'bg-surface-2')}
          >
            <ClubCrest team={p.team} size={20} />
            <span className="min-w-0 flex-1 truncate text-sm font-semibold text-fg">{p.name}</span>
            <PositionBadge position={p.position} />
            <span className="w-10 text-right text-data font-semibold tabular-nums text-accent">{p.predicted_points.toFixed(1)}</span>
          </li>
        ))}
      </ul>

      <p aria-live="polite" className={cn('text-micro text-fg-subtle', status ? 'mt-1.5' : 'sr-only')}>
        {status ?? (expanded ? `${results.length} ${results.length === 1 ? 'player' : 'players'} found` : '')}
      </p>
    </div>
  );
}
