import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Search, Loader2, AlertCircle, User, TrendingUp } from 'lucide-react';
import { useAppStore } from './appStore';
import { apiService, handleApiError } from './api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const POSITION_COLORS: Record<string, string> = {
  GK: 'bg-yellow-100 text-yellow-800',
  DEF: 'bg-blue-100 text-blue-800',
  MID: 'bg-green-100 text-green-800',
  FWD: 'bg-red-100 text-red-800',
};

const EXAMPLE_SEARCHES = [
  'Salah', 'Haaland', 'De Bruyne', 'Trent', 'Saka',
];

type PlayerResult = Record<string, any>;

export function PlayerSearch() {
  const { neo4jConnected } = useAppStore();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<PlayerResult[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerResult | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const searchMutation = useMutation({
    mutationFn: (q: string) => apiService.searchPlayers(q),
    onSuccess: (data) => {
      setResults(data.players);
      setSelectedPlayer(null);
      setHasSearched(true);
    },
  });

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !neo4jConnected) return;
    searchMutation.mutate(query.trim());
  };

  const handleExample = (name: string) => {
    setQuery(name);
    searchMutation.mutate(name);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="bg-white border-b px-6 py-4">
        <h2 className="text-2xl font-bold text-gray-900">🔍 Player Search</h2>
        <p className="text-sm text-gray-600 mt-1">Search for any FPL player and view their stats</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Search bar */}
        <form onSubmit={handleSearch} className="flex gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={neo4jConnected ? 'Search by player name...' : 'Connect to Neo4j first...'}
              disabled={!neo4jConnected}
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
            />
          </div>
          <button
            type="submit"
            disabled={!neo4jConnected || searchMutation.isPending || !query.trim()}
            className="px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {searchMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            Search
          </button>
        </form>

        {/* Not connected */}
        {!neo4jConnected && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 shrink-0" />
            <p className="text-sm text-yellow-800">Connect to Neo4j in Settings to search players.</p>
          </div>
        )}

        {/* Example searches */}
        {!hasSearched && neo4jConnected && (
          <div>
            <p className="text-sm text-gray-500 mb-2">Try searching for:</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_SEARCHES.map((name) => (
                <button
                  key={name}
                  onClick={() => handleExample(name)}
                  className="px-3 py-1.5 text-sm bg-gray-100 hover:bg-purple-100 hover:text-purple-700 rounded-lg transition-colors"
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {searchMutation.isError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-sm text-red-700">
            {handleApiError(searchMutation.error)}
          </div>
        )}

        {/* No results */}
        {hasSearched && results.length === 0 && !searchMutation.isPending && (
          <div className="text-center py-12 text-gray-500">
            <User className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p>No players found for "{query}"</p>
            <p className="text-sm mt-1">Try a partial name or check the spelling</p>
          </div>
        )}

        {/* Results list */}
        {results.length > 0 && !selectedPlayer && (
          <div className="space-y-2">
            <p className="text-sm text-gray-500">{results.length} player{results.length !== 1 ? 's' : ''} found</p>
            {results.map((player, i) => (
              <button
                key={i}
                onClick={() => setSelectedPlayer(player)}
                className="w-full text-left bg-white border border-gray-200 hover:border-purple-400 hover:shadow-sm rounded-xl p-4 transition-all flex items-center justify-between group"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center text-purple-600 font-bold text-sm">
                    {(player.player_name ?? player.name ?? '?')[0].toUpperCase()}
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{player.player_name ?? player.name}</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      {player.position && (
                        <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${POSITION_COLORS[player.position] ?? 'bg-gray-100 text-gray-600'}`}>
                          {player.position}
                        </span>
                      )}
                      {player.season && (
                        <span className="text-xs text-gray-400">{player.season}</span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  {player.total_points != null && (
                    <p className="font-bold text-purple-600">{player.total_points} pts</p>
                  )}
                  {player.goals != null && (
                    <p className="text-xs text-gray-500">{player.goals}G {player.assists ?? 0}A</p>
                  )}
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Player detail */}
        {selectedPlayer && (
          <PlayerDetail
            player={selectedPlayer}
            onBack={() => setSelectedPlayer(null)}
          />
        )}
      </div>
    </div>
  );
}

function PlayerDetail({ player, onBack }: { player: PlayerResult; onBack: () => void }) {
  const name = player.player_name ?? player.name ?? 'Unknown';
  const pos = player.position ?? '—';

  // Build chart data from available stats
  const statFields = [
    { key: 'total_points', label: 'Points' },
    { key: 'goals', label: 'Goals' },
    { key: 'assists', label: 'Assists' },
    { key: 'clean_sheets', label: 'CS' },
    { key: 'bonus', label: 'Bonus' },
  ];
  const chartData = statFields
    .filter((s) => player[s.key] != null && player[s.key] > 0)
    .map((s) => ({ name: s.label, value: player[s.key] }));

  const detailStats = [
    { label: 'Total Points', value: player.total_points },
    { label: 'Goals', value: player.goals },
    { label: 'Assists', value: player.assists },
    { label: 'Clean Sheets', value: player.clean_sheets },
    { label: 'Bonus', value: player.bonus },
    { label: 'Minutes', value: player.minutes?.toLocaleString() },
    { label: 'Games', value: player.games },
    { label: 'Avg ICT', value: player.avg_ict_index ?? player.avg_ict },
    { label: 'Value (£m)', value: player.avg_value_millions ?? player.max_value },
    { label: 'Max Selected', value: player.max_selected?.toLocaleString() },
  ].filter((s) => s.value != null);

  return (
    <div className="space-y-4">
      <button
        onClick={onBack}
        className="text-sm text-purple-600 hover:text-purple-800 flex items-center gap-1"
      >
        ← Back to results
      </button>

      {/* Player header */}
      <div className="bg-gradient-to-br from-purple-600 to-indigo-700 text-white rounded-xl p-6 flex items-center gap-4">
        <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center text-2xl font-bold">
          {name[0].toUpperCase()}
        </div>
        <div>
          <h3 className="text-2xl font-bold">{name}</h3>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-purple-200 text-sm">{pos}</span>
            {player.season && <span className="text-purple-200 text-sm">· {player.season}</span>}
          </div>
        </div>
        {player.total_points != null && (
          <div className="ml-auto text-right">
            <p className="text-3xl font-bold">{player.total_points}</p>
            <p className="text-purple-200 text-sm">FPL Points</p>
          </div>
        )}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {detailStats.map((s) => (
          <div key={s.label} className="bg-white border rounded-xl p-3 text-center">
            <p className="text-xl font-bold text-gray-900">{s.value}</p>
            <p className="text-xs text-gray-500 mt-0.5">{s.label}</p>
          </div>
        ))}
      </div>

      {/* Chart */}
      {chartData.length > 0 && (
        <div className="bg-white border rounded-xl p-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" /> Performance Overview
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{ fontSize: 12, borderRadius: 8 }}
                cursor={{ fill: '#f3f0ff' }}
              />
              <Bar dataKey="value" fill="#7c3aed" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
