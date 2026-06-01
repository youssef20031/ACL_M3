import { useEffect, useState } from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  MessageCircle,
  Target,
  Search,
  ArrowLeftRight,
  Settings as SettingsIcon,
  Database,
  Sparkles,
  Moon,
  Sun,
  Menu,
  X,
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService } from '../services/api';
import { cn } from '../utils/cn';

export function Layout() {
  const location = useLocation();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const {
    neo4jConnected,
    neo4jStats,
    embeddingsBuilt,
    embeddingCount,
    theme,
    setNeo4jConnected,
    setNeo4jStats,
    setEmbeddingsBuilt,
    setEmbeddingCount,
    toggleTheme,
  } = useAppStore();

  const { data: health } = useQuery({
    queryKey: ['health', 'layout'],
    queryFn: () => apiService.getHealth(),
    refetchInterval: 30000,
  });

  useEffect(() => {
    if (!health) {
      return;
    }

    setNeo4jConnected(health.neo4j === 'connected');
    setNeo4jStats(health.neo4j_stats ?? null);
    setEmbeddingsBuilt(health.embeddings_built);
    setEmbeddingCount(health.embedding_count);
  }, [health, setEmbeddingCount, setEmbeddingsBuilt, setNeo4jConnected, setNeo4jStats]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    document.documentElement.style.colorScheme = theme;
  }, [theme]);

  useEffect(() => {
    setMobileNavOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (!mobileNavOpen) {
      document.body.style.overflow = '';
      return;
    }

    document.body.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = '';
    };
  }, [mobileNavOpen]);

  const navItems = [
    { to: '/qa', icon: MessageCircle, label: 'Q&A Assistant' },
    { to: '/trivia', icon: Target, label: 'FantasyTrivia' },
    { to: '/search', icon: Search, label: 'Player Search' },
    { to: '/compare', icon: ArrowLeftRight, label: 'Compare Players' },
    { to: '/settings', icon: SettingsIcon, label: 'Settings' },
  ];

  return (
    <div className={cn('flex flex-col transition-colors lg:flex-row', 
      location.pathname === '/qa' ? 'h-[120dvh] lg:h-[100dvh]' : 'min-h-[100dvh]',
      theme === 'dark' ? 'bg-slate-950 text-slate-100' : 'bg-gray-50 text-slate-900')}>
      <header className={cn('sticky top-0 z-40 flex items-center justify-between border-b px-4 py-3 lg:hidden', theme === 'dark' ? 'border-slate-800 bg-slate-950/90 text-slate-100 backdrop-blur-xl' : 'border-white bg-white/90 text-slate-900 backdrop-blur-xl')}>
        <button
          type="button"
          onClick={() => setMobileNavOpen(true)}
          className={cn('inline-flex h-11 w-11 items-center justify-center rounded-full transition-colors', theme === 'dark' ? 'bg-white/5 text-slate-100 hover:bg-white/10' : 'bg-gray-100 text-gray-700 hover:bg-gray-200')}
          aria-label="Open navigation menu"
          aria-expanded={mobileNavOpen}
          aria-controls="mobile-sidebar"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="min-w-0 text-center">
          <p className="truncate text-sm font-semibold uppercase tracking-[0.2em] text-violet-300">FPL AI Assistant</p>
        </div>

        <button
          type="button"
          onClick={toggleTheme}
          className={cn('inline-flex h-11 w-11 items-center justify-center rounded-full transition-colors', theme === 'dark' ? 'bg-white/5 text-slate-100 hover:bg-white/10' : 'bg-gray-100 text-gray-700 hover:bg-gray-200')}
          aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </header>

      {mobileNavOpen && (
        <button
          type="button"
          className="fixed inset-0 z-40 cursor-default bg-slate-950/65 backdrop-blur-[2px] lg:hidden"
          onClick={() => setMobileNavOpen(false)}
          aria-label="Close navigation menu"
        />
      )}

      {/* Sidebar */}
      <aside
        id="mobile-sidebar"
        className={cn(
          'fixed inset-y-0 left-0 z-50 flex w-[18rem] max-w-[85vw] transform flex-col text-white transition-transform duration-300 ease-out lg:static lg:w-64 lg:min-h-[100dvh] lg:max-w-none lg:translate-x-0',
          mobileNavOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0',
          theme === 'dark' ? 'bg-gradient-to-br from-slate-950 via-slate-900 to-violet-950' : 'bg-gradient-to-br from-purple-600 to-indigo-700'
        )}
      >
        <div className="px-4 py-4 sm:px-6 lg:p-6">
          <div className="flex items-start justify-between gap-3">
            <div>
              <h1 className="flex items-center gap-2 text-xl font-bold sm:text-2xl">
                ⚽ FPL FantasyTrivia
              </h1>
              <p className="mt-1 text-sm text-purple-200">Graph-RAG Q&A System</p>
            </div>
            <div className="flex items-center gap-2 lg:gap-0">
              <button
                type="button"
                onClick={toggleTheme}
                className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/15 bg-white/10 text-white transition-colors hover:bg-white/20"
                aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              >
                {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>
              <button
                type="button"
                onClick={() => setMobileNavOpen(false)}
                className="ml-1 inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/15 bg-white/10 text-white transition-colors hover:bg-white/20 lg:hidden"
                aria-label="Close navigation menu"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="border-y border-purple-500/30 bg-purple-700/30 px-4 py-3 sm:px-6">
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Database className="w-4 h-4" />
                Neo4j
              </span>
              <span
                className={cn(
                  'px-2 py-0.5 rounded-full text-xs font-medium',
                  neo4jConnected
                    ? 'bg-green-500 text-white'
                    : 'bg-red-500 text-white'
                )}
              >
                {neo4jConnected ? 'Connected' : 'Offline'}
              </span>
            </div>
            {neo4jStats && (
              <div className="text-xs text-purple-200 pl-6">
                {neo4jStats.total_nodes.toLocaleString()} nodes
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-2">
                <Sparkles className="w-4 h-4" />
                Embeddings
              </span>
              <span
                className={cn(
                  'px-2 py-0.5 rounded-full text-xs font-medium',
                  embeddingsBuilt
                    ? 'bg-green-500 text-white'
                    : 'bg-gray-500 text-white'
                )}
              >
                {embeddingsBuilt ? 'Ready' : 'Not Built'}
              </span>
            </div>
            {embeddingsBuilt && (
              <div className="text-xs text-purple-200 pl-6">
                {embeddingCount.toLocaleString()} vectors
              </div>
            )}
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto px-2 py-3 space-y-1 sm:px-3">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              onClick={() => setMobileNavOpen(false)}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
                  'hover:bg-purple-500/30',
                  isActive
                    ? theme === 'dark' ? 'bg-slate-900 text-white font-medium shadow-lg' : 'bg-purple-700/50 text-white font-medium shadow-sm'
                    : 'text-purple-100/80 hover:text-white'
                )
              }
            >
              <item.icon className="w-5 h-5" />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-purple-500/30 p-4 text-xs text-purple-200">
          <p>© 2026 FPL FantasyTrivia</p>
          <p className="mt-1">Powered by Neo4j & FastAPI</p>
        </div>
      </aside>

      {/* Main Content */}
      <main className={cn('min-w-0 flex-1 transition-colors',
        location.pathname === '/qa' ? 'flex flex-col overflow-hidden' : 'overflow-auto',
        theme === 'dark' ? 'bg-slate-950' : 'bg-gray-50')}>
        <Outlet />
      </main>
    </div>
  );
}