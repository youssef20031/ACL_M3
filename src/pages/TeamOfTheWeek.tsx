import { useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import {
  apiService,
  handleApiError,
  type PositionCode,
  type TeamOfWeekMode,
  type TeamOfWeekPlayer,
} from '../services/api';
import {
  Card,
  ClubCrest,
  Container,
  ErrorState,
  PageHeader,
  Pitch,
  PitchPlayer,
  PitchRow,
  PlayerAvatar,
  PositionBadge,
  SectionHeading,
  Select,
  Skeleton,
  StatCard,
  TabPanel,
  Tabs,
} from '../components/ui';

const LINES: { position: PositionCode; label: string }[] = [
  { position: 'GK', label: 'Goalkeeper' },
  { position: 'DEF', label: 'Defenders' },
  { position: 'MID', label: 'Midfielders' },
  { position: 'FWD', label: 'Forwards' },
];

function fmtPoints(points: number, mode: TeamOfWeekMode) {
  return mode === 'predicted' ? points.toFixed(1) : String(Math.round(points));
}

function TeamPitch({ starters, mode, gameweek }: { starters: TeamOfWeekPlayer[]; mode: TeamOfWeekMode; gameweek: number }) {
  return (
    <Pitch label={`${mode === 'predicted' ? 'Predicted' : 'Dream'} team, gameweek ${gameweek}`}>
      {LINES.map((line) => (
        <PitchRow key={line.position} label={line.label}>
          {starters
            .filter((p) => p.position === line.position)
            .map((p) => (
              <PitchPlayer
                key={p.name}
                name={p.name}
                team={p.team}
                photo={p.photo}
                opponents={p.opponents}
                captain={p.captain}
                viceCaptain={p.vice_captain}
                status={p.status}
                statusNote={p.news}
                value={`${fmtPoints(p.points, mode)} pts`}
              />
            ))}
        </PitchRow>
      ))}
    </Pitch>
  );
}

function PitchSkeleton() {
  return <Skeleton className="h-[30rem] w-full rounded-lg sm:h-[38rem]" />;
}

export function TeamOfTheWeek() {
  const [params, setParams] = useSearchParams();
  const mode: TeamOfWeekMode = params.get('mode') === 'actual' ? 'actual' : 'predicted';
  const requestedGw = Number(params.get('gw')) || undefined;

  const { data: season } = useQuery({
    queryKey: ['season-current'],
    queryFn: () => apiService.getCurrentSeason(),
    staleTime: 5 * 60_000,
  });
  const finished = season?.finished_gameweeks ?? [];

  const gameweek = mode === 'actual' ? requestedGw : undefined;
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['team-of-the-week', mode, gameweek ?? 'default'],
    queryFn: () => apiService.getTeamOfTheWeek(mode, gameweek),
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
  });

  // Keep the gw param out of the URL in predicted mode (it only applies to dream teams).
  useEffect(() => {
    if (mode === 'predicted' && params.has('gw')) {
      params.delete('gw');
      setParams(params, { replace: true });
    }
  }, [mode, params, setParams]);

  const setMode = (next: TeamOfWeekMode) => setParams(next === 'actual' ? { mode: 'actual' } : {});
  const team = data && data.mode === mode ? data : undefined;
  const captain = team?.starters.find((p) => p.captain);

  return (
    <Container>
      <PageHeader
        eyebrow={season?.season ?? 'This season'}
        title="Team of the Week"
        description={
          mode === 'predicted'
            ? 'The highest-scoring XI our model expects next gameweek, picked like a real FPL squad: £100m, 3 players per club at most.'
            : 'The best-performing XI of a finished gameweek, by actual FPL points.'
        }
      />

      <Tabs
        label="Team of the Week view"
        idPrefix="totw"
        value={mode}
        onChange={setMode}
        items={[
          { value: 'predicted', label: `Predicted${season?.next_gameweek ? ` · GW${season.next_gameweek}` : ''}` },
          { value: 'actual', label: 'Dream team' },
        ]}
      />

      <TabPanel idPrefix="totw" value={mode} className="pt-5">
        {isError ? (
          <Card>
            <ErrorState title="Couldn't build the team" description={handleApiError(error)} onRetry={() => refetch()} />
          </Card>
        ) : (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(18rem,1fr)]">
            <div aria-busy={isFetching}>
              {isLoading || !team ? <PitchSkeleton /> : <TeamPitch starters={team.starters} mode={mode} gameweek={team.gameweek} />}
            </div>

            <aside className="space-y-4">
              {mode === 'actual' && finished.length > 0 && (
                <label className="block">
                  <span className="mb-1.5 block text-micro font-semibold uppercase text-fg-muted">Gameweek</span>
                  <Select
                    value={team?.gameweek ?? requestedGw ?? ''}
                    onChange={(e) => setParams({ mode: 'actual', gw: e.target.value })}
                  >
                    {[...finished].reverse().map((n) => (
                      <option key={n} value={n}>
                        Gameweek {n}
                      </option>
                    ))}
                  </Select>
                </label>
              )}

              <div className="grid grid-cols-2 gap-3">
                <StatCard
                  label={mode === 'predicted' ? 'Predicted points' : 'Points'}
                  value={team ? fmtPoints(team.total_points, mode) : '–'}
                  detail={mode === 'predicted' ? 'Captain counted twice' : `Gameweek ${team?.gameweek ?? ''}`}
                  highlight
                />
                <StatCard label="Formation" value={team?.formation ?? '–'} />
                {mode === 'predicted' && (
                  <StatCard
                    label="Squad cost"
                    value={team?.total_cost != null ? `£${team.total_cost.toFixed(1)}m` : '–'}
                    detail={team?.budget != null ? `of £${team.budget.toFixed(1)}m` : undefined}
                  />
                )}
                <StatCard
                  label="Captain"
                  value={captain ? <span className="text-h3">{captain.name.split(' ').slice(-1)[0]}</span> : '–'}
                  detail={captain ? `${fmtPoints(captain.points, mode)} pts${mode === 'predicted' ? ' predicted' : ''}` : undefined}
                />
              </div>

              {mode === 'predicted' && team && team.bench.length > 0 && (
                <Card padded={false}>
                  <div className="px-4 pt-4">
                    <SectionHeading as="h3" title="Bench" subtitle="Cheapest picks that complete the 15-man squad" />
                  </div>
                  <ul>
                    {team.bench.map((p) => (
                      <li key={p.name} className="flex items-center gap-3 border-t border-line px-4 py-2.5">
                        <PlayerAvatar name={p.name} src={p.photo} size={32} />
                        <span className="min-w-0 flex-1">
                          <span className="block truncate text-sm font-semibold text-fg">{p.name}</span>
                          <ClubCrest team={p.team} size={14} label="full" className="text-data text-fg-subtle" />
                        </span>
                        <PositionBadge position={p.position} />
                        <span className="w-14 text-right text-data tabular-nums text-fg-muted">
                          {p.cost != null ? `£${p.cost.toFixed(1)}m` : ''}
                        </span>
                      </li>
                    ))}
                  </ul>
                </Card>
              )}

              <p className="text-data text-fg-subtle">
                {mode === 'predicted'
                  ? 'Picked by an optimiser from the points model’s next-gameweek predictions. Injured and suspended players are left out; doubtful players are marked in yellow.'
                  : 'Like FPL’s Dream Team: the best valid formation by points scored, with no budget or club limit.'}
              </p>
            </aside>
          </div>
        )}
      </TabPanel>
    </Container>
  );
}
