import type { CardPrediction, PlayerAnswerCard as PlayerCardData } from '../../../services/api';
import { Chip, ClubCrest, PlayerAvatar } from '../../ui';
import { cn } from '../../../utils/cn';
import {
  CardFrame,
  CardLink,
  FormBars,
  Highlights,
  MiniBars,
  OpponentChip,
  PlayerMeta,
  ResultPill,
  Section,
  formatKickoff,
  formatValue,
  playerProfileState,
  useCardPhotos,
} from './shared';

const POSITION_PLURAL: Record<string, string> = { GK: 'GKs', DEF: 'DEFs', MID: 'MIDs', FWD: 'FWDs' };

function PredictionStrip({ prediction, position, emphasised }: { prediction: CardPrediction; position: string | null; emphasised: boolean }) {
  const kickoff = formatKickoff(prediction.opponents[0]?.kickoff_time);
  return (
    <Section
      title={`Next: gameweek ${prediction.gameweek}`}
      aside="Model prediction"
      className={cn(emphasised && 'bg-brand/70')}
    >
      <div className="flex items-center gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            {prediction.opponents.length > 0 ? (
              prediction.opponents.map((o, i) => <OpponentChip key={`${o.team}-${i}`} opponent={o} />)
            ) : (
              <Chip tone="warning">No fixture</Chip>
            )}
            {prediction.opponents.length > 1 && <Chip tone="info">Double gameweek</Chip>}
          </div>
          {kickoff && <p className="mt-1 text-data text-fg-muted">{kickoff}</p>}
        </div>
        <div className="shrink-0 text-right">
          <p className={cn('font-display font-extrabold leading-none text-fg', emphasised ? 'text-display' : 'text-h1')}>
            {prediction.points.toFixed(1)}
            <span className="ml-1 text-data font-semibold text-fg-subtle">pts</span>
          </p>
          <p className="mt-1 text-micro text-fg-subtle">
            #{prediction.rank} of {prediction.pool}
            {position && ` · #${prediction.position_rank} ${POSITION_PLURAL[position] ?? position}`}
          </p>
        </div>
      </div>
    </Section>
  );
}

/** A player's profile for the answer: identity, totals, form, next-gameweek prediction, notable stats. */
export function PlayerAnswerCard({ card }: { card: PlayerCardData }) {
  const { player, prediction, match } = card;
  const photo = useCardPhotos([player])(player);
  const predictionFirst = card.emphasis === 'prediction' && prediction;
  const best = card.seasons.length ? Math.max(...card.seasons.map((s) => s.points)) : null;

  return (
    <CardFrame
      eyebrow={predictionFirst ? `Gameweek ${prediction.gameweek} prediction` : `Player · ${card.season_label}`}
      identity={(titleId) => (
        <div className="flex items-center gap-3">
          <span className="relative shrink-0">
            <PlayerAvatar name={player.name} src={photo} size={72} loading="eager" className="ring-1 ring-line" />
            {player.team && (
              <span className="absolute -bottom-1.5 -right-1.5 rounded-full bg-surface p-0.5 ring-1 ring-line">
                <ClubCrest team={player.team} size={22} />
              </span>
            )}
          </span>
          <div className="min-w-0">
            <h3 id={titleId} className="break-words font-display text-h3 font-extrabold text-fg sm:text-h2">
              {player.name}
            </h3>
            <PlayerMeta player={player} showTeamName className="mt-1 text-data" />
            {player.news && <p className="mt-1 line-clamp-2 text-data text-fg-muted">{player.news}</p>}
          </div>
        </div>
      )}
      footer={
        <>
          <CardLink to="/players" state={playerProfileState(player, card.season)} label={`Full profile of ${player.name}`}>
            Full profile
          </CardLink>
          {prediction && <CardLink to="/predictions">All predictions</CardLink>}
        </>
      }
    >
      {predictionFirst && <PredictionStrip prediction={prediction} position={player.position} emphasised />}

      <dl className="grid grid-cols-3 gap-px border-t border-line bg-line [@container(min-width:34rem)]:grid-cols-6">
        {card.stats.map((s, i) => (
          <div key={s.key} className="bg-surface px-3 py-2.5">
            <dt className="truncate text-micro font-semibold uppercase text-fg-subtle">{s.label}</dt>
            <dd className={cn('font-display text-h3 font-bold', i === 0 ? 'text-accent' : 'text-fg')}>{formatValue(s.value, s.format)}</dd>
          </div>
        ))}
      </dl>
      {card.season === null && (
        <p className="border-t border-line px-4 py-1.5 text-micro text-fg-subtle sm:px-5">Totals across every season since 2020-21</p>
      )}

      {match && (
        <Section title={`Gameweek ${match.gameweek ?? '?'} · ${match.season}`}>
          <div className="flex items-center gap-3">
            <ResultPill result={match.result} />
            <p className="min-w-0 flex-1 text-body text-fg">
              {match.opponent ? `v ${match.opponent}` : 'Match'}
              {match.home != null && <span className="text-fg-muted">{match.home ? ' (H)' : ' (A)'}</span>}
              {match.score && <span className="text-fg-muted"> · {match.score}</span>}
            </p>
            <p className="shrink-0 font-display text-h2 font-extrabold text-fg">
              {match.points}
              <span className="ml-1 text-data font-semibold text-fg-subtle">pts</span>
            </p>
          </div>
          <p className="mt-1 text-data text-fg-muted">
            {match.minutes > 0 ? `${match.minutes} min` : 'Did not play'}
            {match.goals > 0 && ` · ${match.goals} goal${match.goals > 1 ? 's' : ''}`}
            {match.assists > 0 && ` · ${match.assists} assist${match.assists > 1 ? 's' : ''}`}
          </p>
        </Section>
      )}

      {card.form.length > 0 && (
        <Section title="Recent form" aside={`Points, last ${card.form.length} matches`}>
          <FormBars matches={card.form} focusGameweek={match?.gameweek} caption={`${player.name}'s points in recent matches`} />
        </Section>
      )}

      {card.seasons.length > 1 && (
        <Section title="Points by season">
          <MiniBars
            caption={`${player.name}'s points per season`}
            height={56}
            floor={50}
            bars={card.seasons.map((s) => ({
              key: s.season,
              value: s.points,
              display: String(s.points),
              label: s.season.slice(2),
              title: `${s.season}${s.team ? ` (${s.team})` : ''}: ${s.points} pts, ${s.goals} goals, ${s.assists} assists in ${s.games} appearances`,
              focus: s.points === best && card.seasons.length > 2,
            }))}
          />
        </Section>
      )}

      {prediction && !predictionFirst && <PredictionStrip prediction={prediction} position={player.position} emphasised={false} />}

      <Highlights items={card.highlights} />
    </CardFrame>
  );
}
