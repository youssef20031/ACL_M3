import { useEffect, useRef, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { AlertTriangle, ArrowLeft, ChevronRight, Loader2, Search, Sparkles, UserX } from 'lucide-react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/appStore';
import { apiService, handleApiError } from '../services/api';
import { PlayerWithClub } from '../components/PlayerWithClub';
import { usePlayerAvatars } from '../hooks/usePlayerAvatars';
import {
  Button,
  Card,
  Chip,
  ClubCrest,
  Container,
  DataTable,
  EmptyState,
  ErrorState,
  PageHeader,
  PlayerAvatar,
  PositionBadge,
  SearchInput,
  SectionHeading,
  Select,
  SkeletonRows,
  StatCard,
  StatusBadge,
  type Column,
} from '../components/ui';
import { StatBarChart } from '../components/players/StatBarChart';

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
  const { neo4jConnected } = useAppStore();
  const location = useLocation();
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<PlayerResult[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<PlayerResult | null>(null);
  const [detailOrigin, setDetailOrigin] = useState<'history' | 'results' | null>(null);
  const [detailSeason, setDetailSeason] = useState<string>('All seasons');
  const [detailCompareState, setDetailCompareState] = useState<PlayerSearchRouteState['compare'] | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const queryClient = useQueryClient();
  // Only the most recently submitted search may write results (guards against out-of-order responses).
  const latestSearchRef = useRef('');

  const searchMutation = useMutation({
    // Go through the query cache so repeating a recent search is served without a new request.
    mutationFn: (q: string) =>
      queryClient.fetchQuery({
        queryKey: ['players', 'search', q.toLowerCase(), 20],
        queryFn: ({ signal }) => apiService.searchPlayers(q, { signal }),
        staleTime: 5 * 60 * 1000,
      }),
    onMutate: (q) => {
      latestSearchRef.current = q;
    },
    onSuccess: (data, q) => {
      if (q !== latestSearchRef.current) return;
      setResults(data.players);
      setSelectedPlayer(null);
      setHasSearched(true);
    },
  });

  // Results carry an avatar when the backend knows it instantly; the rest load per row
  const avatarByName = usePlayerAvatars(
    selectedPlayer ? [] : results.filter((p) => !p.avatar).map((p) => p.player_name ?? p.name)
  );

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

  const showSkeleton = searchMutation.isPending && results.length === 0 && !selectedPlayer;

  return (
    <Container className="pb-10">
      <PageHeader
        eyebrow="Player stats"
        title="Players"
        description="Search any FPL player since 2020-21 for season totals, per-game rates and next-gameweek predictions."
      />

      <form onSubmit={handleSearch} role="search" className="flex flex-col gap-2 sm:flex-row">
        <SearchInput
          value={query}
          onChange={setQuery}
          label="Player name"
          placeholder={neo4jConnected ? 'Search by player name…' : 'Connecting to the database…'}
          disabled={!neo4jConnected}
          className="flex-1"
        />
        <Button
          type="submit"
          disabled={!neo4jConnected || searchMutation.isPending || !query.trim()}
          loading={searchMutation.isPending}
          icon={<Search className="h-4 w-4" aria-hidden="true" />}
          className="sm:w-32"
        >
          Search
        </Button>
      </form>

      <div className="mt-4 space-y-4">
        {!neo4jConnected && (
          <div role="status" className="flex items-start gap-3 rounded-lg border border-warning/30 bg-warning/10 p-3 text-data text-warning">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <p>Connecting to Neo4j… Please wait.</p>
          </div>
        )}

        {!hasSearched && neo4jConnected && !selectedPlayer && (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-data text-fg-subtle">Try:</span>
            {EXAMPLE_SEARCHES.map((name) => (
              <button
                key={name}
                type="button"
                onClick={() => handleExample(name)}
                className="rounded-full border border-line-strong px-3 py-1 text-data text-fg-muted transition-colors hover:border-accent hover:text-fg"
              >
                {name}
              </button>
            ))}
          </div>
        )}

        {searchMutation.isError && (
          <Card padded={false}>
            <ErrorState
              title="Search failed"
              description={handleApiError(searchMutation.error)}
              onRetry={latestSearchRef.current ? () => searchMutation.mutate(latestSearchRef.current) : undefined}
            />
          </Card>
        )}

        {showSkeleton && <SkeletonRows rows={5} />}

        {hasSearched && results.length === 0 && !searchMutation.isPending && !selectedPlayer && (
          <Card padded={false}>
            <EmptyState
              icon={<UserX className="h-8 w-8" aria-hidden="true" />}
              title={`No players found for "${query}"`}
              description="Try a partial name or check the spelling."
            />
          </Card>
        )}

        {results.length > 0 && !selectedPlayer && (
          <ResultsList
            results={results}
            avatarByName={avatarByName}
            onSelect={(player) => setSelectedPlayer(player)}
          />
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
            backLabel={
              detailOrigin === 'history'
                ? detailCompareState?.player1 && detailCompareState?.player2
                  ? 'Back to compare players'
                  : 'Back'
                : 'Back to results'
            }
          />
        )}
      </div>
    </Container>
  );
}

function ResultsList({
  results,
  avatarByName,
  onSelect,
}: {
  results: PlayerResult[];
  avatarByName: Record<string, string | null | undefined>;
  onSelect: (player: PlayerResult) => void;
}) {
  return (
    <Card padded={false}>
      <p className="border-b border-line px-4 py-2.5 text-micro font-semibold uppercase text-fg-subtle" aria-live="polite">
        {results.length} player{results.length !== 1 ? 's' : ''} found
      </p>
      <ul>
        {results.map((player, i) => {
          const name: string = player.player_name ?? player.name ?? 'Unknown';
          return (
            <li key={`${player.element_id ?? name}-${i}`} className="border-b border-line last:border-0">
              <button
                type="button"
                onClick={() => onSelect(player)}
                className="group flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors hover:bg-surface-2"
              >
                <PlayerWithClub
                  playerName={name}
                  teamName={player.team_name}
                  showAbbreviation={false}
                  avatar={player.avatar ?? avatarByName[name]}
                  avatarSize={40}
                  className="min-w-0 flex-1"
                  nameClassName="group-hover:text-accent"
                  meta={
                    <>
                      {player.season && <span className="whitespace-nowrap">· {player.season}</span>}
                      <StatusBadge status={player.status} chance={player.chance_of_playing_next_round} news={player.news} />
                    </>
                  }
                />
                <PositionBadge position={player.position} />
                <span className="hidden w-20 text-right sm:block">
                  {player.total_points != null && (
                    <span className="block text-data font-bold tabular-nums text-fg">{player.total_points} pts</span>
                  )}
                  {player.goals != null && (
                    <span className="block text-micro tabular-nums text-fg-subtle">
                      {player.goals}G {player.assists ?? 0}A
                    </span>
                  )}
                </span>
                {typeof player.predicted_points === 'number' && (
                  <Chip tone="accent" icon={<Sparkles className="h-3 w-3" aria-hidden="true" />}>
                    <span className="sr-only">Predicted next gameweek: </span>
                    {player.predicted_points.toFixed(1)}
                  </Chip>
                )}
                <ChevronRight className="h-4 w-4 shrink-0 text-fg-subtle group-hover:text-fg" aria-hidden="true" />
              </button>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}

type BreakdownRow = { stat: string; total: number; perGame: number | null; per90: number | null; decimals?: number };

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
  const resolvedInitialSeason = initialSeason === 'All seasons' ? '2026-27' : initialSeason;
  const [selectedSeason, setSelectedSeason] = useState<string>(resolvedInitialSeason);

  const playerName: string | undefined = player.player_name ?? player.name;

  // Reset the season picker when the detail view is re-targeted (e.g. via route state).
  useEffect(() => {
    setSelectedSeason(resolvedInitialSeason);
  }, [playerName, resolvedInitialSeason]);

  // Cached per (player, season): switching back to a season or reopening a player is instant.
  const statsQuery = useQuery({
    queryKey: ['players', 'stats', playerName, selectedSeason],
    queryFn: ({ signal }) =>
      apiService.getPlayerStats(playerName as string, selectedSeason === 'All seasons' ? undefined : selectedSeason, signal),
    enabled: !!playerName,
    staleTime: 10 * 60 * 1000,
    // Keep showing the previous season's numbers (under the loading overlay) for the same player only.
    placeholderData: (prev, prevQuery) => (prevQuery?.queryKey[2] === playerName ? prev : undefined),
  });
  const stats: PlayerResult = statsQuery.data?.stats ? statsQuery.data.stats : player;
  const statsLoading = statsQuery.isFetching;
  const noSeasonData = statsQuery.isSuccess && !statsQuery.isPlaceholderData && !statsQuery.data?.stats;

  const handleSeasonChange = (season: string) => {
    setSelectedSeason(season);
  };

  const name: string = stats.player_name ?? stats.name ?? 'Unknown';
  const pos: string = stats.position ?? player.position ?? '—';
  const team: string | undefined = stats.team_name ?? player.team_name;

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
  const hasAvgValue = typeof stats.avg_value_millions === 'number';
  const rawValue = toNumber(stats.avg_value_millions ?? stats.max_value);
  // max_value comes in FPL's tenths of a million (e.g. 145 = £14.5m).
  const valueMillions = rawValue > 25 ? rawValue / 10 : rawValue;
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
    .map((s) => ({ name: s.label, value: stats[s.key] as number }));

  const efficiencyData = [
    { name: 'Pts/G', value: pointsPerGame },
    { name: 'G/G', value: goalsPerGame },
    { name: 'A/G', value: assistsPerGame },
    // Minutes per game (~0-90) would flatten the other bars; it is in the stat cards and table instead.
  ].filter((item) => item.value > 0);

  const per = (v: number) => ({
    perGame: games > 0 ? v / games : null,
    per90: minutes > 0 ? (v / minutes) * 90 : null,
  });
  const breakdown: BreakdownRow[] = [
    { stat: 'Points', total: totalPoints, ...per(totalPoints) },
    { stat: 'Goals', total: goals, ...per(goals) },
    { stat: 'Assists', total: assists, ...per(assists) },
    { stat: 'Clean sheets', total: cleanSheets, ...per(cleanSheets) },
    { stat: 'Bonus', total: bonus, ...per(bonus) },
    { stat: 'Minutes', total: minutes, perGame: games > 0 ? minutes / games : null, per90: null, decimals: 0 },
  ];
  const fmt = (v: number | null, d = 2) => (v == null ? '–' : v.toFixed(d));
  const breakdownColumns: Column<BreakdownRow>[] = [
    { key: 'stat', header: 'Stat', render: (r) => <span className="font-semibold">{r.stat}</span> },
    { key: 'total', header: 'Total', align: 'right', render: (r) => <span className="font-bold">{r.total.toLocaleString()}</span> },
    { key: 'perGame', header: 'Per game', align: 'right', render: (r) => fmt(r.perGame, r.decimals ?? 2) },
    { key: 'per90', header: 'Per 90', description: 'Per 90 minutes played', align: 'right', render: (r) => fmt(r.per90) },
  ];

  const snapshotRows = [
    { label: 'Season', value: selectedSeason },
    { label: 'Position', value: pos },
    { label: 'Team', value: team ?? 'FPL' },
    { label: 'Games', value: games.toLocaleString() },
    { label: 'Avg ICT', value: avgIct ? avgIct.toFixed(2) : '—' },
    { label: hasAvgValue ? 'Avg price' : 'Peak price', value: valueMillions ? `£${valueMillions.toFixed(1)}m` : '—' },
    { label: 'Max selected', value: maxSelected ? maxSelected.toLocaleString() : '—' },
  ];

  const seasons = ['All seasons', '2026-27', '2025-26', '2024-25', '2023-24', '2022-23', '2021-22', '2020-21'];
  const status: string | undefined = stats.status ?? player.status;
  const news: string | undefined = stats.news ?? player.news;
  const chance: number | null | undefined = stats.chance_of_playing_next_round ?? player.chance_of_playing_next_round;
  const predicted = typeof stats.predicted_points === 'number' ? (stats.predicted_points as number) : null;

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <Button variant="ghost" size="sm" onClick={onBack} icon={<ArrowLeft className="h-4 w-4" aria-hidden="true" />} className="self-start">
          {backLabel}
        </Button>

        <label className="flex items-center gap-3">
          <span className="text-micro font-semibold uppercase text-fg-muted">Season</span>
          <Select
            value={selectedSeason}
            onChange={(e) => handleSeasonChange(e.target.value)}
            disabled={statsLoading}
            className="w-44"
          >
            {seasons.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </Select>
        </label>
      </div>

      <Card tone="brand" className="relative overflow-hidden" aria-busy={statsLoading}>
        {statsLoading && (
          <div role="status" className="absolute right-3 top-3 z-10 inline-flex items-center gap-2 rounded-full bg-bg/70 px-3 py-1 text-micro text-fg-muted">
            <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
            Loading {selectedSeason}
          </div>
        )}
        <div className="flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex min-w-0 items-center gap-4">
            <PlayerAvatar
              name={name}
              src={stats.avatar ?? player.avatar}
              size={96}
              loading="eager"
              className="border border-brand-2 bg-surface"
            />
            <div className="min-w-0">
              <p className="text-micro font-semibold uppercase text-accent">{selectedSeason}</p>
              <h2 className="mt-0.5 font-display text-h1 font-extrabold leading-tight text-fg">{name}</h2>
              <div className="mt-2 flex flex-wrap items-center gap-2 text-data text-fg-muted">
                {team && <ClubCrest team={team} size={20} label="full" className="font-semibold text-fg" />}
                <PositionBadge position={pos} />
                <StatusBadge status={status} chance={chance} news={news} />
                {predicted != null && (
                  <Chip tone="accent" icon={<Sparkles className="h-3 w-3" aria-hidden="true" />}>
                    Next GW: {predicted.toFixed(1)} pts
                  </Chip>
                )}
              </div>
              {news && (
                <p className="mt-2 flex items-start gap-1.5 text-data text-warning">
                  <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
                  <span>{news}</span>
                </p>
              )}
            </div>
          </div>
          {stats.total_points != null && (
            <div className="shrink-0 rounded-md border border-brand-2 bg-bg/30 px-4 py-3 sm:text-right">
              <p className="text-micro font-semibold uppercase text-fg-subtle">FPL points</p>
              <p className="font-display text-display font-extrabold leading-none tabular-nums text-accent">{stats.total_points}</p>
              <p className="mt-1 text-data text-fg-muted">{formatRate(pointsPerGame)} pts/game</p>
            </div>
          )}
        </div>
      </Card>

      {noSeasonData && (
        <p role="status" className="rounded-md border border-line bg-surface px-3 py-2 text-data text-fg-subtle">
          No {selectedSeason} stats for {name}. Pick another season.
        </p>
      )}

      {statsQuery.isError && (
        <Card padded={false}>
          <ErrorState title="Couldn't load stats" description={handleApiError(statsQuery.error)} onRetry={() => statsQuery.refetch()} />
        </Card>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Total points" value={totalPoints.toLocaleString()} detail={`${formatRate(pointsPerGame)} pts/game`} highlight />
        <StatCard label="Goals" value={goals.toLocaleString()} detail={`${formatRate(goalsPerGame)} per game`} />
        <StatCard label="Assists" value={assists.toLocaleString()} detail={`${formatRate(assistsPerGame)} per game`} />
        <StatCard label="Minutes" value={minutes.toLocaleString()} detail={`${formatRate(minutesPerGame)} mins/game`} />
        <StatCard label="Clean sheets" value={cleanSheets.toLocaleString()} detail="Defensive contribution" />
        <StatCard label="Bonus" value={bonus.toLocaleString()} detail="Bonus point haul" />
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <section aria-labelledby="season-breakdown">
          <SectionHeading as="h3" id="season-breakdown" title="Season breakdown" subtitle={`${selectedSeason} · ${games.toLocaleString()} games`} />
          <DataTable
            caption={`${name} season breakdown, ${selectedSeason}`}
            columns={breakdownColumns}
            rows={breakdown}
            rowKey={(r) => r.stat}
            loading={statsQuery.isLoading}
            skeletonRows={6}
            dense
          />
        </section>

        <section aria-labelledby="season-snapshot">
          <SectionHeading as="h3" id="season-snapshot" title="Season snapshot" />
          <Card padded={false}>
            <dl>
              {snapshotRows.map((row) => (
                <div key={row.label} className="flex items-center justify-between gap-3 border-b border-line px-4 py-2 text-data last:border-0">
                  <dt className="text-fg-subtle">{row.label}</dt>
                  <dd className="truncate font-semibold tabular-nums text-fg">{row.value}</dd>
                </div>
              ))}
            </dl>
            <p className="border-t border-line px-4 py-3 text-data leading-relaxed text-fg-muted">
              {name} has produced {totalPoints.toLocaleString()} points across {games.toLocaleString()} games, averaging {formatRate(pointsPerGame)} points per appearance.
              {goals > 0 || assists > 0 ? ` The production mix is ${goals.toLocaleString()} goals and ${assists.toLocaleString()} assists.` : ''}
            </p>
          </Card>
        </section>
      </div>

      {(chartData.length > 0 || efficiencyData.length > 0) && (
        <div className="grid gap-4 lg:grid-cols-2">
          {chartData.length > 0 && (
            <Card>
              <SectionHeading as="h3" title="Production breakdown" subtitle="Season totals" />
              <StatBarChart data={chartData} label={`${name} production breakdown, ${selectedSeason}`} height={240} />
            </Card>
          )}
          {efficiencyData.length > 0 && (
            <Card>
              <SectionHeading as="h3" title="Efficiency" subtitle="Per game played" />
              <StatBarChart
                data={efficiencyData}
                label={`${name} per-game efficiency, ${selectedSeason}`}
                height={240}
                tone="accent-2"
                decimals={2}
              />
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
