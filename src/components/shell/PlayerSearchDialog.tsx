import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { Loader2, Search, X } from 'lucide-react';
import { apiService } from '../../services/api';
import { ClubCrest, PlayerAvatar, PositionBadge } from '../ui';
import { cn } from '../../utils/cn';

interface SearchResult {
  player_name: string;
  position?: string | null;
  team_name?: string | null;
  avatar?: string | null;
}

function useDebounced<T>(value: T, ms: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return debounced;
}

interface PlayerSearchDialogProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Header player search: a modal combobox over the typo-tolerant search API.
 * Arrow keys move through results, Enter opens the player's profile.
 */
export function PlayerSearchDialog({ open, onClose }: PlayerSearchDialogProps) {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const listId = useId();
  const term = useDebounced(query.trim(), 150);

  const { data, isFetching, isError } = useQuery({
    queryKey: ['player-search', term],
    queryFn: ({ signal }) => apiService.searchPlayers(term, { limit: 8, signal }),
    enabled: open && term.length >= 2,
    staleTime: 60_000,
  });
  const results = (term.length >= 2 ? (data?.players ?? []) : []) as SearchResult[];

  useEffect(() => {
    if (!open) return;
    setQuery('');
    setActive(0);
    const previous = document.activeElement as HTMLElement | null;
    const id = window.setTimeout(() => inputRef.current?.focus(), 0);
    document.body.style.overflow = 'hidden';
    return () => {
      window.clearTimeout(id);
      document.body.style.overflow = '';
      previous?.focus?.();
    };
  }, [open]);

  useEffect(() => setActive(0), [term]);

  if (!open) return null;

  const choose = (row: SearchResult) => {
    onClose();
    navigate('/players', { state: { player: row, returnTo: 'history' } });
  };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    } else if (e.key === 'ArrowDown' && results.length) {
      e.preventDefault();
      setActive((i) => (i + 1) % results.length);
    } else if (e.key === 'ArrowUp' && results.length) {
      e.preventDefault();
      setActive((i) => (i - 1 + results.length) % results.length);
    } else if (e.key === 'Enter' && results[active]) {
      e.preventDefault();
      choose(results[active]);
    } else if (e.key === 'Tab') {
      // Only the input and the close button are focusable; keep focus inside.
      e.preventDefault();
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/60 px-4 pt-[10vh] backdrop-blur-[2px]" onMouseDown={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Find a player"
        onMouseDown={(e) => e.stopPropagation()}
        onKeyDown={onKeyDown}
        className="w-full max-w-xl overflow-hidden rounded-lg border border-line bg-surface shadow-pop"
      >
        <div className="flex items-center gap-3 border-b border-line px-4">
          {isFetching ? (
            <Loader2 className="h-5 w-5 shrink-0 animate-spin text-fg-subtle" aria-hidden="true" />
          ) : (
            <Search className="h-5 w-5 shrink-0 text-fg-subtle" aria-hidden="true" />
          )}
          <input
            ref={inputRef}
            role="combobox"
            aria-expanded={results.length > 0}
            aria-controls={listId}
            aria-activedescendant={results[active] ? `${listId}-${active}` : undefined}
            aria-autocomplete="list"
            aria-label="Player name"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search players, e.g. Haaland"
            className="h-14 flex-1 bg-transparent text-body text-fg placeholder:text-fg-subtle focus:outline-none"
          />
          <button type="button" onClick={onClose} className="rounded-sm p-1 text-fg-subtle hover:text-fg" aria-label="Close search">
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <ul id={listId} role="listbox" aria-label="Players" className="max-h-[60vh] overflow-y-auto py-1">
          {results.map((row, i) => (
            <li
              key={`${row.player_name}-${i}`}
              id={`${listId}-${i}`}
              role="option"
              aria-selected={i === active}
              onMouseEnter={() => setActive(i)}
              onClick={() => choose(row)}
              className={cn('flex cursor-pointer items-center gap-3 px-4 py-2', i === active && 'bg-surface-2')}
            >
              <PlayerAvatar name={row.player_name} src={row.avatar} size={36} />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-semibold text-fg">{row.player_name}</span>
                {row.team_name && <ClubCrest team={row.team_name} size={14} label="full" className="text-data text-fg-subtle" />}
              </span>
              <PositionBadge position={row.position} />
            </li>
          ))}
        </ul>

        <div aria-live="polite" className="text-data text-fg-subtle">
          {term.length < 2 ? (
            <p className="px-4 py-3">Type at least two letters.</p>
          ) : isError ? (
            <p className="px-4 py-3 text-danger">Search is unavailable right now.</p>
          ) : !isFetching && results.length === 0 ? (
            <p className="px-4 py-3">No players match “{term}”.</p>
          ) : (
            <p className="sr-only">{results.length} {results.length === 1 ? 'player' : 'players'} found</p>
          )}
        </div>
      </div>
    </div>
  );
}
