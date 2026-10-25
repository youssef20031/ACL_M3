import { useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { ShieldCheck } from 'lucide-react';
import {
  apiService,
  handleApiError,
  isNotFoundError,
  isServiceUnavailableError,
  type TeamOfWeekMode,
  type TeamOfWeekPick,
  type TeamOfWeekPlayer,
} from '../services/api';
import {
  Button,
  Card,
  ClubCrest,
  Container,
  EmptyState,
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
import { ModelReview } from '../components/totw/ModelReview';
import { PITCH_LINES } from '../components/totw/pitchLines';
import { SegmentedControl, type SegmentOption } from '../components/predictions/SegmentedControl';
import { usePageMeta } from '../hooks/usePageMeta';

// The page's tabs: the two squads from /api/team-of-the-week, plus the model's past picks
// against what they scored (/api/model-review). `mode` in the URL; predicted is the default.
// The predicted and review tabs also switch between the default pick and the safer one
// (`pick=safer` in the URL), which the backend saves next to it before every deadline.
type View = TeamOfWeekMode | 'review';

const parseView = (value: string | null): View => (value === 'actual' || value === 'review' ? value : 'predicted');
const parsePick = (value: string | null): TeamOfWeekPick => (value === 'safer' ? 'safer' : 'default');

const PICK_OPTIONS: SegmentOption<TeamOfWeekPick>[] = [
  { value: 'default', label: 'Most points' },
  { value: 'safer', label: 'Safer pick', shortLabel: 'Safer' },
];

function fmtPoints(points: number, mode: TeamOfWeekMode) {
  return mode === 'predicted' ? points.toFixed(1) : String(Math.round(points));
}

interface TeamPitchProps {
  starters: TeamOfWeekPlayer[];
  mode: TeamOfWeekMode;
  gameweek: number;
  /** The safer pick: each player's chance of playing 60 minutes under his points. */
  safer?: boolean;
}

function TeamPitch({ starters, mode, gameweek, safer = false }: TeamPitchProps) {
  const label = mode === 'predicted' ? (safer ? 'Safer predicted' : 'Predicted') : 'Dream';
  return (
    <Pitch label={`${label} team, gameweek ${gameweek}`}>
      {PITCH_LINES.map((line) => (
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
                // Availability is today's; only meaningful for the upcoming gameweek.
                status={mode === 'predicted' ? p.status : null}
                statusNote={mode === 'predicted' ? p.news : null}
                value={`${fmtPoints(p.points, mode)} pts`}
                detail={safer && p.play60 != null ? `${Math.round(p.play60 * 100)}% to play 60` : undefined}
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
  usePageMeta({ title: "Team of the Week", description: "The highest-scoring FPL squad our model predicts for the next gameweek (£100m, 3 per club), a safer pick that avoids risky stacks, how past picks scored, and each week's dream team." });
  const [params, setParams] = useSearchParams();
  const view = parseView(params.get('mode'));
  const mode: TeamOfWeekMode = view === 'actual' ? 'actual' : 'predicted';
  const pick: TeamOfWeekPick = view === 'actual' ? 'default' : parsePick(params.get('pick'));
  const requestedGw = Number(params.get('gw')) || undefined;

  const { data: season } = useQuery({
    queryKey: ['season-current'],
    queryFn: () => apiService.getCurrentSeason(),
    staleTime: 5 * 60_000,
  });
  const finished = season?.finished_gameweeks ?? [];

  const gameweek = view === 'actual' ? requestedGw : undefined;
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['team-of-the-week', mode, gameweek ?? 'default', pick],
    queryFn: () => apiService.getTeamOfTheWeek(mode, gameweek, pick),
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
    enabled: view !== 'review',
  });

  const reviewGw = view === 'review' ? requestedGw : undefined;
  const review = useQuery({
    queryKey: ['model-review', reviewGw ?? 'default', pick],
    queryFn: () => apiService.getModelReview(reviewGw, pick),
    placeholderData: keepPreviousData,
    staleTime: 5 * 60_000,
    enabled: view === 'review',
    // A 404 only means no saved pick for that gameweek: show it at once rather than retrying.
    retry: (failures, err) => !isNotFoundError(err) && failures < 1,
  });

  // Keep the gw param out of the URL in predicted mode (it only applies to past gameweeks), and
  // the pick out of it on the dream team tab (there's only one dream team).
  useEffect(() => {
    const stale = (view === 'predicted' && params.has('gw')) || (view === 'actual' && params.has('pick'));
    if (stale) {
      params.delete(view === 'predicted' ? 'gw' : 'pick');
      setParams(params, { replace: true });
    }
  }, [view, params, setParams]);

  // Switching tabs drops the gameweek (each tab has its own default) but keeps the safer pick.
  const withPick = (next: Record<string, string>) => (pick === 'safer' ? { ...next, pick } : next);
  const setView = (next: View) =>
    setParams(next === 'actual' ? { mode: next } : withPick(next === 'predicted' ? {} : { mode: next }));
  const setPick = (next: TeamOfWeekPick) => {
    const nextParams = new URLSearchParams(params);
    if (next === 'safer') nextParams.set('pick', next);
    else nextParams.delete('pick');
    setParams(nextParams);
  };
  const team = data && data.mode === mode && (data.pick ?? 'default') === pick ? data : undefined;
  const captain = team?.starters.find((p) => p.captain);
  const safer = pick === 'safer';
  // An older backend ignores pick=safer and answers with the default team, without a `pick`.
  const saferUnsupported = safer && !isFetching && data?.mode === mode && data.pick === undefined;

  return (
    <Container>
      <PageHeader
        eyebrow={season?.season ?? 'This season'}
        title="Team of the Week"
        description={
          view === 'predicted'
            ? safer
              ? 'A safer version of the predicted squad, under the same £100m and 3-per-club rules: players who might not play 60 minutes count for less, most of all when clubmates start together, so nailed stars stay and doubtful teammates make way.'
              : 'The highest-scoring XI our model expects next gameweek, picked like a real FPL squad: £100m, 3 players per club at most.'
            : view === 'review'
              ? 'How the model’s pick for a past gameweek actually did: each player’s predicted and real FPL points, against the average FPL score.'
              : 'The best-performing XI of a finished gameweek, by actual FPL points.'
        }
      />

      <Tabs
        label="Team of the Week view"
        idPrefix="totw"
        value={view}
        onChange={setView}
        items={[
          { value: 'predicted', label: `Predicted${season?.next_gameweek ? ` · GW${season.next_gameweek}` : ''}` },
          { value: 'review', label: 'Model vs actual' },
          { value: 'actual', label: 'Dream team' },
        ]}
      />

      <TabPanel idPrefix="totw" value={view} className="pt-5">
        {view !== 'actual' && (
          <div className="mb-5 flex flex-wrap items-center gap-x-4 gap-y-2">
            <SegmentedControl label="Which predicted squad" options={PICK_OPTIONS} value={pick} onChange={setPick} />
            <p className="text-data text-fg-muted">
              {safer
                ? 'Nailed players first: doubtful players, and clubmates who start together, count for less.'
                : 'The squad with the most expected points.'}
            </p>
          </div>
        )}
        {view === 'review' ? (
          <ModelReview
            pick={pick}
            review={review.data}
            error={review.isError ? review.error : undefined}
            selectedGameweek={reviewGw}
            updating={review.isFetching && review.isPlaceholderData}
            onGameweekChange={(gw) => setParams(withPick({ mode: 'review', gw: String(gw) }))}
            onRetry={() => review.refetch()}
            onShowLatest={reviewGw ? () => setParams(withPick({ mode: 'review' })) : undefined}
          />
        ) : saferUnsupported || (isError && safer && isServiceUnavailableError(error)) ? (
          <Card>
            <EmptyState
              icon={<ShieldCheck className="h-8 w-8" aria-hidden="true" />}
              title="The safer pick isn’t ready yet"
              description={saferUnsupported ? 'This server doesn’t offer the safer pick yet.' : handleApiError(error)}
              action={
                <Button variant="secondary" size="sm" onClick={() => setPick('default')}>
                  Show the main pick
                </Button>
              }
            />
          </Card>
        ) : isError ? (
          <Card>
            <ErrorState title="Couldn't build the team" description={handleApiError(error)} onRetry={() => refetch()} />
          </Card>
        ) : (
          <div className="grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(18rem,1fr)]">
            <div aria-busy={isFetching}>
              {isLoading || !team ? (
                <PitchSkeleton />
              ) : (
                <TeamPitch starters={team.starters} mode={mode} gameweek={team.gameweek} safer={safer} />
              )}
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
                    <SectionHeading as="h3" title="Bench" subtitle="Picked to come on when a starter doesn’t play" />
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
                {mode !== 'predicted'
                  ? 'Like FPL’s Dream Team: the best valid formation by points scored, with no budget or club limit.'
                  : safer
                    ? 'The same predictions, chosen differently: each player counts as his predicted points minus a quarter of the points he’d lose by not playing 60 minutes (from a chance-of-playing model and FPL’s injury news), and clubmates who start together pay a further share of that risk. Tested on 121 past gameweeks it scored about the same as the main pick (0.4 points a week fewer, within the noise) with far fewer same-club pairs.'
                    : 'Picked by an optimiser from the points model’s next-gameweek predictions. Injured and suspended players are left out; doubtful players are marked in yellow.'}
              </p>
            </aside>
          </div>
        )}
      </TabPanel>
    </Container>
  );
}
