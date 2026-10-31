import type { FixturePicksAnswerCard as FixturePicksCardData } from '../../../services/api';
import { getClubAbbreviation } from '../../../utils/clubLogos';
import { Chip, ClubCrest, PlayerAvatar } from '../../ui';
import { CardFrame, CardLink, PlayerAction, PlayerMeta, predictionQuestion, useCardPhotos } from './shared';

function kickoff(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleString(undefined, { weekday: 'short', hour: '2-digit', minute: '2-digit' });
}

const abbr = (team: string) => getClubAbbreviation(team) || team.slice(0, 3).toUpperCase();

/** Each match of the next gameweek with its top predicted players, in kickoff order. */
export function FixturePicksAnswerCard({ card }: { card: FixturePicksCardData }) {
  const photoOf = useCardPhotos(card.fixtures.flatMap((f) => f.players.map((p) => p.player)));
  return (
    <CardFrame
      icon="calendar"
      eyebrow={`${card.gameweek ? `Gameweek ${card.gameweek} · ` : ''}top ${card.per_fixture} per match`}
      title={card.title}
      footer={
        <>
          <CardLink to="/predictions">All predictions</CardLink>
          <CardLink to="/fixtures">Fixtures</CardLink>
        </>
      }
    >
      <ol className="grid gap-px bg-line [@container(min-width:36rem)]:grid-cols-2">
        {card.fixtures.map((f) => {
          const when = kickoff(f.kickoff_time);
          return (
            <li key={`${f.home}-${f.away}`} className="bg-surface px-4 py-3 sm:px-5">
              <p className="flex items-center gap-1.5 text-data font-semibold text-fg">
                <ClubCrest team={f.home} size={18} />
                <span>{abbr(f.home)}</span>
                <span className="text-fg-subtle">v</span>
                <span>{abbr(f.away)}</span>
                <ClubCrest team={f.away} size={18} />
                <span className="sr-only">
                  , {f.home} against {f.away}
                </span>
                {when && <span className="ml-auto text-micro font-normal text-fg-subtle">{when}</span>}
              </p>
              {f.players.length ? (
                <ol className="mt-2 space-y-1">
                  {f.players.map((p) => (
                    <li key={p.player.name}>
                      <PlayerAction
                        player={p.player}
                        season={null}
                        question={predictionQuestion(p.player.name)}
                        className="grid grid-cols-[1.75rem_minmax(0,1fr)_auto] items-center gap-2 rounded-md px-1 py-1 transition-colors hover:bg-surface-2"
                      >
                        <PlayerAvatar name={p.player.name} src={photoOf(p.player)} size={28} />
                        <span className="min-w-0">
                          <span className="block truncate text-data font-semibold text-fg group-hover:underline">{p.player.name}</span>
                          <span className="flex flex-wrap items-center gap-1">
                            <PlayerMeta player={p.player} />
                            {p.double && <Chip tone="info">2 matches</Chip>}
                          </span>
                        </span>
                        <span className="font-display text-body font-bold tabular-nums text-accent">
                          {p.points.toFixed(1)}
                          <span className="sr-only"> predicted points</span>
                        </span>
                      </PlayerAction>
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="mt-2 text-data text-fg-subtle">No predictions for this match.</p>
              )}
            </li>
          );
        })}
      </ol>
    </CardFrame>
  );
}
