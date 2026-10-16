import { Link } from 'react-router-dom';
import type { SeasonAnswerCard as SeasonCardData } from '../../../services/api';
import { Chip, PlayerAvatar } from '../../ui';
import { CardFrame, CardLink, Highlights, PlayerMeta, Section, formatValue, playerProfileState, useCardPhotos } from './shared';

/** A season in numbers, with its points leaders. */
export function SeasonAnswerCard({ card }: { card: SeasonCardData }) {
  const photoOf = useCardPhotos(card.leaders.map((l) => l.player));
  return (
    <CardFrame
      icon="calendar"
      eyebrow={card.in_progress ? 'Season so far' : 'Season summary'}
      title={card.season}
      aside={card.in_progress ? <Chip tone="info">In progress</Chip> : undefined}
      footer={<CardLink to="/players">Season leaders</CardLink>}
    >
      <dl className="grid grid-cols-3 gap-px border-t border-line bg-line [@container(min-width:30rem)]:grid-cols-5">
        {card.stats.map((s) => (
          <div key={s.key} className="bg-surface px-3 py-2.5">
            <dt className="truncate text-micro font-semibold uppercase text-fg-subtle">{s.label}</dt>
            <dd className="font-display text-h3 font-bold text-fg">{formatValue(s.value, s.format)}</dd>
          </div>
        ))}
      </dl>

      {card.leaders.length > 0 && (
        <Section title="Most points">
          <ol className="grid gap-2 [@container(min-width:36rem)]:grid-cols-3">
            {card.leaders.map(({ player, points, goals, assists }, i) => (
              <li key={player.name}>
                <Link
                  to="/players"
                  state={playerProfileState(player, card.season)}
                  className="group flex min-w-0 items-center gap-2 rounded-md border border-line bg-bg/40 p-2 hover:border-line-strong"
                >
                  <span className="w-4 shrink-0 text-center text-data font-bold text-fg-subtle">{i + 1}</span>
                  <PlayerAvatar name={player.name} src={photoOf(player)} size={40} />
                  <span className="min-w-0">
                    <span className="block truncate text-data font-semibold text-fg group-hover:underline">{player.name}</span>
                    <PlayerMeta player={player} />
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

      <Highlights items={card.highlights} />
    </CardFrame>
  );
}
