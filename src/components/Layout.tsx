import { Outlet, NavLink } from 'react-router-dom';
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
import { cn } from '../utils/cn';

export function Layout() {
  const { neo4jConnected, neo4jStats, embeddingsBuilt, embeddingCount } = useAppStore();

  const navItems = [
    { to: '/qa', icon: MessageCircle, label: 'Q&A Assistant' },
    { to: '/trivia', icon: Target, label: 'FantasyTrivia' },
    { to: '/search', icon: Search, label: 'Player Search' },
    { to: '/compare', icon: ArrowLeftRight, label: 'Compare Players' },
    { to: '/settings', icon: SettingsIcon, label: 'Settings' },
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <aside className="w-64 bg-gradient-to-br from-purple-600 to-indigo-700 text-white flex flex-col">
        <div className="p-6">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            ⚽ FPL FantasyTrivia
          </h1>
          <p className="text-purple-200 text-sm mt-1">Graph-RAG Q&A System</p>
        </div>

        {/* Status */}
        <div className="px-6 py-3 bg-purple-700/30 border-y border-purple-500/30">
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
        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                cn(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors',
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
        </nav>

        {/* Footer */}
        <div className="p-4 text-xs text-purple-200 border-t border-purple-500/30">
          <p>© 2024 FPL FantasyTrivia</p>
          <p className="mt-1">Powered by Neo4j & FastAPI</p>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}