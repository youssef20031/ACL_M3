import { useEffect } from 'react';
import { Outlet, NavLink } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  MessageCircle,
  Target,
  Search,
  ArrowLeftRight,
  Settings as SettingsIcon,
  Database,
  Sparkles,
} from 'lucide-react';
import { useAppStore } from '../store/appStore';
import { apiService } from '../services/api';
import { cn } from '../utils/cn';

export function Layout() {
  const {
    neo4jConnected,
    neo4jStats,
    embeddingsBuilt,
    embeddingCount,
    setNeo4jConnected,
    setNeo4jStats,
    setEmbeddingsBuilt,
    setEmbeddingCount,
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

  const navItems = [
    { to: '/qa', icon: MessageCircle, label: 'Q&A Assistant' },
    { to: '/trivia', icon: Target, label: 'FantasyTrivia' },
    { to: '/search', icon: Search, label: 'Player Search' },
    { to: '/compare', icon: ArrowLeftRight, label: 'Compare Players' },
    { to: '/settings', icon: SettingsIcon, label: 'Settings' },
  ];

  return (
    <div className="flex min-h-[100dvh] flex-col bg-gray-50 lg:flex-row">
      {/* Sidebar */}
      <aside className="flex w-full flex-col bg-gradient-to-br from-purple-600 to-indigo-700 text-white lg:w-64 lg:min-h-[100dvh]">
        <div className="px-4 py-4 sm:px-6 lg:p-6">
          <h1 className="flex items-center gap-2 text-xl font-bold sm:text-2xl">
            ⚽ FPL FantasyTrivia
          </h1>
          <p className="mt-1 text-sm text-purple-200">Graph-RAG Q&A System</p>
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
        <nav className="flex-1 overflow-x-auto px-2 py-3 sm:px-3 lg:space-y-1 lg:overflow-visible lg:px-3">
          <div className="flex gap-2 overflow-x-auto pb-1 lg:block lg:space-y-1 lg:overflow-visible lg:pb-0">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  'flex min-w-max items-center gap-3 rounded-lg px-3 py-2.5 transition-colors lg:min-w-0',
                  'hover:bg-purple-500/30',
                  isActive
                    ? 'bg-white text-purple-700 font-medium shadow-lg'
                    : 'text-purple-100'
                )
              }
            >
              <item.icon className="w-5 h-5" />
              <span>{item.label}</span>
            </NavLink>
          ))}
          </div>
        </nav>

        {/* Footer */}
        <div className="border-t border-purple-500/30 p-4 text-xs text-purple-200">
          <p>© 2024 FPL FantasyTrivia</p>
          <p className="mt-1">Powered by Neo4j & FastAPI</p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="min-w-0 flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}