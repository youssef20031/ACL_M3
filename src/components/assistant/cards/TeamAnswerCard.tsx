import { Link } from 'react-router-dom';
import type { TeamAnswerCard as TeamCardData } from '../../../services/api';
import { ClubCrest, PlayerAvatar, StatusBadge } from '../../ui';
import {
  CardFrame,
  CardLink,
  Highlights,
  OpponentChip,
  ResultPill,
  Section,
  formatKickoff,
  ordinal,
  playerProfileState,
  teamCode,
  useCardPhotos,
} from './shared';

/** A club this season: table position, form, next fixtures with FDR, top performers and team news. */
export function TeamAnswerCard({ card }: { card: TeamCardData }) {
  const photoOf = useCardPhotos([...card.top_players.map((p) => p.player), ...card.flagged]);
  const s = card.standing;
  const tiles: [string, string, number][] = s
    ? [
        ['P', 'Played', s.played],
        ['W', 'Won', s.won],
        ['D', 'Drawn', s.drawn],
        ['L', 'Lost', s.lost],
        ['GF', 'Goals for', s.goals_for],
        ['GA', 'Goals against', s.goals_against],
        ['GD', 'Goal difference', s.goal_difference],
        ['Pts', 'Points', s.points],
      ]
    : [];

  return (
    <CardFrame
      eyebrow={`Team · ${card.season}`}
      identity={(titleId) => (
        <div className="flex items-center gap-3">
          <span className="inline-flex h-16 w-16 shrink-0 items-center justify-center rounded-lg bg-surface-2 ring-1 ring-line">
            <ClubCrest team={card.team} size={48} />
          </span>
          <div className="min-w-0">
            <h3 id={titleId} className="break-words font-display text-h3 font-extrabold text-fg sm:text-h2">
              {card.team}
            </h3>
            {s && (
              <p className="text-data text-fg-muted">
                <span className="font-semibold text-fg">{s.points} pts</span> from {s.played} {s.played === 1 ? 'match' : 'matches'}
              </p>
            )}
          </div>
        </div>
      )}
      aside={
        s ? (
          <p className="rounded-md bg-brand px-3 py-2 text-center ring-1 ring-brand-2">
            <span className="block font-display text-h1 font-extrabold leading-none text-accent">{ordinal(s.position)}</span>
            <span className="text-micro uppercase text-fg-subtle">of {card.teams_in_table}</span>
          </p>
        ) : undefined
      }
      footer={
        <>
          <CardLink to="/table">League table</CardLink>
          <CardLink to="/fixtures">Fixtures</CardLink>
        </>
      }
    >
      {s && (
        <dl className="grid grid-cols-4 gap-px border-t border-line bg-line [@container(min-width:34rem)]:grid-cols-8">
          {tiles.map(([short, long, value]) => (
            <div key={short} className="bg-surface px-2 py-2 text-center">
              <dt className="text-micro font-semibold uppercase text-fg-subtle">
                <abbr title={long} className="no-underline">
                  {short}
                </abbr>
              </dt>
              <dd className="font-display text-h3 font-bold text-fg">{short === 'GD' && value > 0 ? `+${value}` : value}</dd>
            </div>
          ))}
        </dl>
      )}

      {card.form.length > 0 && (
        <Section title="Form" aside="Oldest to newest">
          <ol className="flex flex-wrap gap-1.5">
            {card.form.map((f, i) => (
              <li key={`${f.gameweek}-${i}`} className="flex items-center gap-1.5 rounded-md bg-surface-2 py-1 pl-1 pr-2 text-data">
                <ResultPill result={f.result} title={`GW ${f.gameweek ?? '?'}: ${f.score} v ${f.opponent}`} />
                <span className="text-fg">{f.score}</span>
                <span className="text-fg-subtle">
                  <span aria-hidden="true">v {teamCode(f.opponent)}</span>
                  <span className="sr-only">v {f.opponent}</span>
                  {f.home ? ' (H)' : ' (A)'}
                </span>
              </li>
            ))}
          </ol>
        </Section>
      )}

      {card.next_fixtures.length > 0 && (
        <Section
          title="Next fixtures"
          aside={card.next_fixtures.some((f) => f.difficulty != null) ? 'Colour = FPL difficulty' : undefined}
        >
          <ol className="flex flex-wrap gap-2">
            {card.next_fixtures.map((f, i) => (
              <li key={`${f.team}-${i}`} className="flex flex-col items-start gap-0.5">
                <span className="text-micro text-fg-subtle">{f.gameweek ? `GW ${f.gameweek}` : formatKickoff(f.kickoff_time)}</span>
                <OpponentChip opponent={f} />
              </li>
            ))}
          </ol>
        </Section>
      )}

      {card.top_players.length > 0 && (
        <Section title="Top performers" aside="This season">
          <ol className="grid gap-2 [@container(min-width:36rem)]:grid-cols-3">
            {card.top_players.map(({ player, points, goals, assists }) => (
              <li key={player.name}>
                <Link
                  to="/players"
                  state={playerProfileState(player, card.season)}
                  className="group flex min-w-0 items-center gap-2 rounded-md border border-line bg-bg/40 p-2 hover:border-line-strong"
                >
                  <PlayerAvatar name={player.name} src={photoOf(player)} size={40} />
                  <span className="min-w-0">
                    <span className="block truncate text-data font-semibold text-fg group-hover:underline">{player.name}</span>
                    <span className="block text-micro text-fg-subtle">
                      {points} pts · {goals} G · {assists} A
                    </span>
                  </span>
                </Link>
              </li>
            ))}
          </ol>
        </Section>
      )}

      {card.flagged.length > 0 && (
        <Section title="Team news" aside={card.flagged_total > card.flagged.length ? `${card.flagged_total} flagged` : undefined}>
          <ul className="space-y-1.5">
            {card.flagged.map((p) => (
              <li key={p.name} className="flex min-w-0 items-center gap-2">
                <PlayerAvatar name={p.name} src={photoOf(p)} size={28} />
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-1.5">
                    <span className="truncate text-data font-semibold text-fg">{p.name}</span>
                    <StatusBadge status={p.status} chance={p.chance} />
                  </span>
                  {p.news && <span className="block truncate text-micro text-fg-muted">{p.news}</span>}
                </span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      <Highlights items={card.highlights} />
    </CardFrame>
  );
}
