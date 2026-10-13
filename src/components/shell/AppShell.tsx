import { useCallback, useEffect, useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { apiService } from '../../services/api';
import { useAppStore } from '../../store/appStore';
import { useHealthSync } from '../../hooks/useHealthSync';
import { cn } from '../../utils/cn';
import { SiteHeader } from './SiteHeader';
import { DeadlineBar } from './DeadlineBar';
import { SiteFooter } from './SiteFooter';
import { PlayerSearchDialog } from './PlayerSearchDialog';

// Pages that manage their own full-height scrolling (chat) get a fixed-height main and no footer.
const FULL_HEIGHT_ROUTES = ['/assistant'];

function isTypingTarget(el: EventTarget | null) {
  const node = el as HTMLElement | null;
  return !!node && (node.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(node.tagName));
}

export function AppShell() {
  const location = useLocation();
  const [searchOpen, setSearchOpen] = useState(false);
  const theme = useAppStore((s) => s.theme);
  const setTheme = useAppStore((s) => s.setTheme);
  useHealthSync();

  // Dark is the only theme until the light token set exists. Legacy pages still read
  // `theme` from the store and the html.dark overrides in index.css.
  useEffect(() => {
    if (theme !== 'dark') setTheme('dark');
    const root = document.documentElement;
    root.classList.add('dark');
    root.dataset.theme = 'dark';
    root.style.colorScheme = 'dark';
  }, [theme, setTheme]);

  const { data: season } = useQuery({
    queryKey: ['season-current'],
    queryFn: () => apiService.getCurrentSeason(),
    staleTime: 5 * 60 * 1000,
    refetchInterval: 10 * 60 * 1000,
    retry: false,
  });

  const openSearch = useCallback(() => setSearchOpen(true), []);

  // "/" opens player search from anywhere except text fields.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey && !isTypingTarget(e.target)) {
        e.preventDefault();
        setSearchOpen(true);
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  const fullHeight = FULL_HEIGHT_ROUTES.includes(location.pathname);

  return (
    <div className={cn('pm-root flex flex-col', fullHeight ? 'h-[100dvh]' : 'min-h-[100dvh]')}>
      <a
        href="#main"
        className="sr-only z-50 rounded-md bg-accent px-3 py-2 font-semibold text-accent-on focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to content
      </a>
      <SiteHeader onOpenSearch={openSearch} />
      {season?.next_gameweek && season.next_deadline && (
        <DeadlineBar gameweek={season.next_gameweek} deadline={season.next_deadline} />
      )}
      <main id="main" tabIndex={-1} className={cn('flex-1 focus:outline-none', fullHeight && 'flex min-h-0 flex-col overflow-hidden')}>
        <Outlet />
      </main>
      {!fullHeight && <SiteFooter />}
      <PlayerSearchDialog open={searchOpen} onClose={() => setSearchOpen(false)} />
    </div>
  );
}
