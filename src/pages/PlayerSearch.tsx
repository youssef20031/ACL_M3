import { useEffect, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Search, Loader2, AlertCircle, User, TrendingUp, Target, ArrowLeftRight, Gauge, Shield, Flame, Aperture } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { cn } from '../utils/cn';

const POSITION_COLORS: Record<string, string> = {
  GK: 'bg-yellow-100 text-yellow-800',
  DEF: 'bg-blue-100 text-blue-800',
  MID: 'bg-green-100 text-green-800',
  FWD: 'bg-red-100 text-red-800',
};

const EXAMPLE_SEARCHES = ['Salah', 'Haaland', 'De Bruyne', 'Trent', 'Saka'];

type PlayerResult = Record<string, any>;
type PlayerSearchRouteState = {
  player?: PlayerResult;
  season?: string;
  returnTo?: 'history' | 'results';
  compare?: {
    player1?: string;
    player2?: string;
    season?: string;
  };
};

export function PlayerSearch() {
  const { neo4jConnected, theme } = useAppStore();
  const location = useLocation();
  const navigate = useNavigate();
  const isDark = theme === 'dark';
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<PlayerResult[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerResult | null>(null);
  const [detailOrigin, setDetailOrigin] = useState<'history' | 'results' | null>(null);
  const [detailSeason, setDetailSeason] = useState<string>('All seasons');
  const [detailCompareState, setDetailCompareState] = useState<PlayerSearchRouteState['compare'] | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const headerClass = isDark ? 'border-slate-800 bg-slate-950/80' : 'border-white bg-white';
  const headerTitleClass = isDark ? 'text-slate-100' : 'text-gray-900';
  const headerTextClass = isDark ? 'text-slate-400' : 'text-gray-600';
  const chipClass = isDark ? 'bg-white/5 text-slate-200 hover:bg-white/10 hover:text-white' : 'bg-gray-100 text-gray-700 hover:bg-purple-100 hover:text-purple-700';
  const inputClass = isDark
    ? 'border-slate-700 bg-slate-950/70 text-slate-100 placeholder:text-slate-500 focus:ring-violet-500 disabled:bg-slate-900'
    : 'border-gray-300 bg-white text-gray-900 placeholder:text-gray-400 focus:ring-purple-500 disabled:bg-gray-100';

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

  useEffect(() => {
    const state = location.state as PlayerSearchRouteState | null;
    if (!state?.player) {
      return;
    }

    setQuery(state.player.player_name ?? state.player.name ?? '');
    setSelectedPlayer(state.player);
    setDetailOrigin(state.returnTo ?? 'history');
    setDetailSeason(state.season || 'All seasons');
    setDetailCompareState(state.compare ?? null);
  }, [location.state]);

  return (
    <div className="flex flex-col h-full">
      <div className={cn('border-b px-4 py-4 sm:px-6', headerClass)}>
        <h2 className={cn('text-xl font-bold sm:text-2xl', headerTitleClass)}>🔍 Player Search</h2>
        <p className={cn('mt-1 text-sm', headerTextClass)}>Search for any FPL player and view their stats</p>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6 sm:px-6">
        <form onSubmit={handleSearch} className="flex flex-col gap-3 sm:flex-row">
          <div className="relative flex-1">
            <Search className={cn('absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4', isDark ? 'text-slate-500' : 'text-gray-400')} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={neo4jConnected ? 'Search by player name...' : 'Connect to Neo4j first...'}
              disabled={!neo4jConnected}
              className={cn('w-full rounded-xl border pl-10 pr-4 py-3 text-base focus:outline-none focus:ring-2 disabled:cursor-not-allowed', inputClass)}
            />
          </div>
          <button
            type="submit"
            disabled={!neo4jConnected || searchMutation.isPending || !query.trim()}
            className="w-full px-6 py-3 bg-purple-600 hover:bg-purple-700 text-white rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 sm:w-auto"
          >
            {searchMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            Search
          </button>
        </form>

        {!neo4jConnected && (
          <div className={cn('flex items-start gap-3 rounded-lg border p-4', isDark ? 'border-amber-400/20 bg-amber-500/10' : 'border-yellow-200 bg-yellow-50')}>
            <AlertCircle className="w-5 h-5 text-yellow-600 mt-0.5 shrink-0" />
            <p className={cn('text-sm', isDark ? 'text-amber-200' : 'text-yellow-800')}>Connecting to Neo4j... Please wait.</p>
          </div>
        )}

        {!hasSearched && neo4jConnected && (
          <div>
            <p className={cn('mb-2 text-sm', headerTextClass)}>Try searching for:</p>
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_SEARCHES.map((name) => (
                <button
                  key={name}
                  onClick={() => handleExample(name)}
                  className={cn('rounded-lg px-3 py-1.5 text-sm transition-colors', chipClass)}
                >
                  {name}
                </button>
              ))}
            </div>
          </div>
        )}

        {searchMutation.isError && (
          <div className={cn('rounded-lg border p-4 text-sm', isDark ? 'border-red-400/20 bg-red-500/10 text-red-200' : 'border-red-200 bg-red-50 text-red-700')}>
            {handleApiError(searchMutation.error)}
          </div>
        )}

        {hasSearched && results.length === 0 && !searchMutation.isPending && (
          <div className={cn('py-12 text-center', headerTextClass)}>
            <User className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p>No players found for "{query}"</p>
            <p className="text-sm mt-1">Try a partial name or check the spelling</p>
          </div>
        )}

        {results.length > 0 && !selectedPlayer && (
          <div className="space-y-2">
            <p className={cn('text-sm', headerTextClass)}>{results.length} player{results.length !== 1 ? 's' : ''} found</p>
            {results.map((player, i) => (
              <button
                key={i}
                onClick={() => setSelectedPlayer(player)}
                className={cn('w-full text-left rounded-xl border p-4 transition-all flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between', isDark ? 'border-slate-800 bg-slate-900/80 text-slate-100 hover:border-violet-400/60 hover:shadow-[0_18px_50px_rgba(2,6,23,0.35)]' : 'border-gray-200 bg-white text-gray-900 hover:border-purple-400 hover:shadow-sm')}
              >
                <div className="flex items-center gap-3">
                  {player.avatar ? (
                    <img src={player.avatar} alt={player.player_name ?? player.name} className="w-10 h-10 rounded-full object-cover" />
                  ) : (
                    <div className={cn('flex h-10 w-10 items-center justify-center rounded-full text-sm font-bold', isDark ? 'bg-violet-500/15 text-violet-200' : 'bg-purple-100 text-purple-600')}>
                      {(player.player_name ?? player.name ?? '?')[0].toUpperCase()}
                    </div>
                  )}
                  <div>
                    <p className={cn('font-medium', headerTitleClass)}>{player.player_name ?? player.name}</p>
                    <div className="flex items-center gap-2 mt-0.5">
                      {player.position && (
                        <span className={cn('rounded px-1.5 py-0.5 text-xs font-medium', isDark ? 'bg-white/8 text-slate-200' : POSITION_COLORS[player.position] ?? 'bg-gray-100 text-gray-600')}>
                          {player.position}
                        </span>
                      )}
                      {player.season && (
                        <span className={cn('text-xs', headerTextClass)}>{player.season}</span>
                      )}
                    </div>
                  </div>
                </div>
                <div className="text-left sm:text-right">
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

        {selectedPlayer && (
          <PlayerDetail
            player={selectedPlayer}
            initialSeason={detailSeason}
            onBack={
              detailOrigin === 'history'
                ? () => {
                    if (detailCompareState?.player1 && detailCompareState?.player2) {
                      navigate('/compare', {
                        state: detailCompareState,
                      });
                      return;
                    }

                    navigate(-1);
                  }
                : () => {
                    setSelectedPlayer(null);
                    setDetailOrigin(null);
                    setDetailSeason('All seasons');
                    setDetailCompareState(null);
                  }
            }
            backLabel={detailOrigin === 'history' ? 'Back to compare players' : 'Back to results'}
          />
        )}
      </div>
    </div>
  );
}

function PlayerDetail({
  player,
  onBack,
  backLabel,
  initialSeason,
}: {
  player: PlayerResult;
  onBack: () => void;
  backLabel: string;
  initialSeason: string;
}) {
  const resolvedInitialSeason = initialSeason === 'All seasons' ? '2025-26' : initialSeason;
  const [selectedSeason, setSelectedSeason] = useState<string>(resolvedInitialSeason);
  const [stats, setStats] = useState<PlayerResult>(player);
  const { theme } = useAppStore();
  const isDark = theme === 'dark';
  const pageText = isDark ? 'text-slate-100' : 'text-gray-900';
  const mutedText = isDark ? 'text-slate-400' : 'text-gray-500';
  const panelClass = isDark ? 'border-slate-800 bg-slate-900/80 text-slate-100' : 'border-gray-200 bg-white text-gray-900';
  const controlClass = isDark ? 'border-slate-700 bg-slate-950/70 text-slate-100 focus:ring-violet-500 focus:border-violet-500' : 'border-gray-400 bg-gray-100 text-gray-900 focus:ring-purple-500 focus:border-purple-500 focus:bg-white';

  const statsMutation = useMutation({
    mutationFn: ({ name, season }: { name: string; season?: string }) =>
      apiService.getPlayerStats(name, season === 'All seasons' ? undefined : season),
    onSuccess: (data) => {
      if (data.stats) {
        setStats(data.stats);
      }
    },
  });

  useEffect(() => {
    const playerName = player.player_name ?? player.name;
    if (!playerName) {
      return;
    }

    setStats(player);
    statsMutation.mutate({
      name: playerName,
      season: resolvedInitialSeason,
    });
  }, [player.player_name, player.name, resolvedInitialSeason]);

  const handleSeasonChange = (season: string) => {
    setSelectedSeason(season);
    statsMutation.mutate({
      name: player.player_name ?? player.name,
      season,
    });
  };

  const name = stats.player_name ?? stats.name ?? 'Unknown';
  const pos = stats.position ?? '—';

  const toNumber = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : 0);
  const formatRate = (value: number) => (Number.isFinite(value) ? value.toFixed(2) : '0.00');

  const totalPoints = toNumber(stats.total_points);
  const goals = toNumber(stats.goals);
  const assists = toNumber(stats.assists);
  const cleanSheets = toNumber(stats.clean_sheets);
  const bonus = toNumber(stats.bonus);
  const minutes = toNumber(stats.minutes);
  const games = toNumber(stats.games);
  const avgIct = toNumber(stats.avg_ict_index ?? stats.avg_ict);
  const valueMillions = toNumber(stats.avg_value_millions ?? stats.max_value);
  const maxSelected = toNumber(stats.max_selected);

  const pointsPerGame = games > 0 ? totalPoints / games : 0;
  const goalsPerGame = games > 0 ? goals / games : 0;
  const assistsPerGame = games > 0 ? assists / games : 0;
  const minutesPerGame = games > 0 ? minutes / games : 0;

  const statFields = [
    { key: 'total_points', label: 'Points' },
    { key: 'goals', label: 'Goals' },
    { key: 'assists', label: 'Assists' },
    { key: 'clean_sheets', label: 'CS' },
    { key: 'bonus', label: 'Bonus' },
  ];
  const chartData = statFields
    .filter((s) => stats[s.key] != null && stats[s.key] > 0)
    .map((s) => ({ name: s.label, value: stats[s.key] }));

  const efficiencyData = [
    { name: 'Pts/G', value: pointsPerGame },
    { name: 'G/G', value: goalsPerGame },
    { name: 'A/G', value: assistsPerGame },
    { name: 'Mins/G', value: minutesPerGame },
  ].filter((item) => item.value > 0);

  const summaryCards = [
    { label: 'Total Points', value: totalPoints, suffix: 'pts', icon: Target, accent: 'from-violet-500 to-fuchsia-500', note: `${formatRate(pointsPerGame)} pts/game` },
    { label: 'Goals', value: goals, suffix: '', icon: Aperture, accent: 'from-amber-400 to-orange-500', note: `${formatRate(goalsPerGame)} per game` },
    { label: 'Assists', value: assists, suffix: '', icon: ArrowLeftRight, accent: 'from-sky-400 to-cyan-500', note: `${formatRate(assistsPerGame)} per game` },
    { label: 'Minutes', value: minutes, suffix: 'm', icon: Gauge, accent: 'from-emerald-400 to-teal-500', note: `${formatRate(minutesPerGame)} mins/game` },
    { label: 'Clean Sheets', value: cleanSheets, suffix: '', icon: Shield, accent: 'from-blue-400 to-indigo-500', note: 'Defensive contribution' },
    { label: 'Bonus', value: bonus, suffix: '', icon: Flame, accent: 'from-rose-400 to-red-500', note: 'Bonus point haul' },
  ];

  const snapshotRows = [
    { label: 'Season', value: selectedSeason },
    { label: 'Position', value: pos },
    { label: 'Team', value: stats.team_name ?? player.team_name ?? 'FPL' },
    { label: 'Games', value: games.toLocaleString() },
    { label: 'Avg ICT', value: avgIct ? avgIct.toFixed(2) : '—' },
    { label: 'Value (£m)', value: valueMillions ? valueMillions.toFixed(2) : '—' },
    { label: 'Max Selected', value: maxSelected ? maxSelected.toLocaleString() : '—' },
  ];

  const seasons = ['All seasons', '2025-26', '2024-25', '2023-24', '2022-23', '2021-22', '2020-21'];

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <button
          onClick={onBack}
          className="flex items-center gap-1 text-sm text-purple-600 hover:text-purple-800"
        >
          ← {backLabel}
        </button>

        <div className="flex items-center gap-3">
          <span className={cn('text-xs font-bold uppercase tracking-wider', mutedText)}>Select Season:</span>
          <select
            value={selectedSeason}
            onChange={(e) => handleSeasonChange(e.target.value)}
            disabled={statsMutation.isPending}
            className={cn('w-40 cursor-pointer rounded-lg border px-3 py-2 text-sm font-medium transition-all focus:ring-2 disabled:opacity-50', controlClass)}
          >
            {seasons.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </div>

      <div className={cn('relative overflow-hidden rounded-2xl border p-5 sm:p-6', isDark ? 'border-slate-800 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-slate-100 shadow-[0_20px_80px_rgba(2,6,23,0.45)]' : 'border-slate-200 bg-gradient-to-br from-purple-600 via-indigo-700 to-slate-900 text-white shadow-lg')}>
        <div className="pointer-events-none absolute inset-0 opacity-40">
          <div className="absolute -right-16 -top-12 h-40 w-40 rounded-full bg-white/10 blur-3xl" />
          <div className="absolute left-1/3 top-8 h-28 w-28 rounded-full bg-fuchsia-400/20 blur-3xl" />
        </div>
        {statsMutation.isPending && (
          <div className="absolute inset-0 bg-black/10 backdrop-blur-[1px] flex items-center justify-center z-10">
            <Loader2 className="w-8 h-8 animate-spin text-white" />
          </div>
        )}
        <div className="relative z-10 flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-4">
            {stats.avatar ? (
              <img src={stats.avatar} alt={name} className="h-20 w-20 rounded-full border-2 border-white/30 object-cover shadow-lg" />
            ) : (
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-white/20 text-3xl font-bold shadow-lg">
                {name[0].toUpperCase()}
              </div>
            )}
            <div>
              <p className={cn('text-xs font-semibold uppercase tracking-[0.24em]', isDark ? 'text-slate-300' : 'text-purple-100')}>
                Player Dashboard
              </p>
              <h3 className="mt-1 text-3xl font-black tracking-tight sm:text-4xl">{name}</h3>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-sm">
                <span className={cn('rounded-full px-3 py-1 font-medium', isDark ? 'bg-white/10 text-slate-100' : 'bg-white/15 text-white')}>
                  {pos}
                </span>
                <span className={cn('rounded-full px-3 py-1 font-medium', isDark ? 'bg-white/10 text-slate-100' : 'bg-white/15 text-white')}>
                  {stats.team_name ?? player.team_name ?? 'FPL'}
                </span>
                <span className={cn('rounded-full px-3 py-1 font-medium', isDark ? 'bg-white/10 text-slate-100' : 'bg-white/15 text-white')}>
                  {selectedSeason}
                </span>
              </div>
            </div>
          </div>
          {stats.total_points != null && (
            <div className={cn('rounded-2xl border px-4 py-3 text-left lg:text-right', isDark ? 'border-white/10 bg-white/5' : 'border-white/20 bg-white/10')}>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-white/70">FPL Points</p>
              <p className="mt-1 text-4xl font-black leading-none">{stats.total_points}</p>
              <p className="mt-1 text-sm text-white/70">{formatRate(pointsPerGame)} pts/game</p>
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.55fr_0.95fr]">
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {summaryCards.map((card) => {
              const Icon = card.icon;
              return (
                <div key={card.label} className={cn('rounded-2xl border p-4 shadow-sm', panelClass)}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className={cn('text-xs font-semibold uppercase tracking-[0.22em]', mutedText)}>{card.label}</p>
                      <p className={cn('mt-2 text-3xl font-black', pageText)}>
                        {typeof card.value === 'number' ? card.value.toLocaleString() : card.value}
                        <span className="ml-1 text-sm font-semibold opacity-70">{card.suffix}</span>
                      </p>
                    </div>
                    <div className={cn('rounded-2xl bg-gradient-to-br p-3 text-white shadow-lg', card.accent)}>
                      <Icon className="h-5 w-5" />
                    </div>
                  </div>
                  <p className={cn('mt-3 text-sm', mutedText)}>{card.note}</p>
                </div>
              );
            })}
          </div>

          {chartData.length > 0 && (
            <div className={cn('rounded-2xl border p-4 sm:p-5', panelClass)}>
              <h4 className={cn('mb-3 flex items-center gap-2 text-sm font-semibold', mutedText)}>
                <TrendingUp className="h-4 w-4" /> Production Breakdown
              </h4>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDark ? 'rgba(148,163,184,0.12)' : '#f0f0f0'} />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: isDark ? '#cbd5e1' : '#475569' }} />
                  <YAxis tick={{ fontSize: 12, fill: isDark ? '#cbd5e1' : '#475569' }} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, background: isDark ? '#020617' : '#ffffff', borderColor: isDark ? 'rgba(148,163,184,0.18)' : '#e2e8f0', color: isDark ? '#e2e8f0' : '#0f172a' }} cursor={{ fill: isDark ? 'rgba(124,58,237,0.12)' : '#f3f0ff' }} />
                  <Bar dataKey="value" fill="#7c3aed" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {efficiencyData.length > 0 && (
            <div className={cn('rounded-2xl border p-4 sm:p-5', panelClass)}>
              <h4 className={cn('mb-3 flex items-center gap-2 text-sm font-semibold', mutedText)}>
                <Gauge className="h-4 w-4" /> Efficiency Snapshot
              </h4>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={efficiencyData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDark ? 'rgba(148,163,184,0.12)' : '#f0f0f0'} />
                  <XAxis dataKey="name" tick={{ fontSize: 12, fill: isDark ? '#cbd5e1' : '#475569' }} />
                  <YAxis tick={{ fontSize: 12, fill: isDark ? '#cbd5e1' : '#475569' }} />
                  <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8, background: isDark ? '#020617' : '#ffffff', borderColor: isDark ? 'rgba(148,163,184,0.18)' : '#e2e8f0', color: isDark ? '#e2e8f0' : '#0f172a' }} cursor={{ fill: isDark ? 'rgba(14,165,233,0.12)' : '#e0f2fe' }} />
                  <Bar dataKey="value" fill="#06b6d4" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className={cn('rounded-2xl border p-4 sm:p-5', panelClass)}>
            <h4 className={cn('mb-4 text-sm font-semibold', mutedText)}>Season Snapshot</h4>
            <div className="space-y-3">
              {snapshotRows.map((row) => (
                <div key={row.label} className={cn('flex items-center justify-between rounded-xl px-3 py-2', isDark ? 'bg-white/5' : 'bg-slate-50')}>
                  <span className={cn('text-sm', mutedText)}>{row.label}</span>
                  <span className={cn('text-sm font-semibold', pageText)}>{row.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className={cn('rounded-2xl border p-4 sm:p-5', panelClass)}>
            <h4 className={cn('mb-4 text-sm font-semibold', mutedText)}>Quick Insight</h4>
            <div className={cn('rounded-2xl p-4', isDark ? 'bg-violet-500/10' : 'bg-violet-50')}>
              <p className={cn('text-xs font-semibold uppercase tracking-[0.22em]', isDark ? 'text-violet-200' : 'text-violet-700')}>
                Fantasy summary
              </p>
              <p className={cn('mt-2 text-sm leading-6', pageText)}>
                {name} has produced {totalPoints.toLocaleString()} points across {games.toLocaleString()} games, averaging {formatRate(pointsPerGame)} points per appearance.
                {goals > 0 || assists > 0 ? ` The production mix is ${goals.toLocaleString()} goals and ${assists.toLocaleString()} assists.` : ''}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
