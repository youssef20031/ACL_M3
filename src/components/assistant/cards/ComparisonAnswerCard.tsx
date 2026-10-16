import { Check } from 'lucide-react';
import type {
  CardPlayer,
  CardPrediction,
  ComparisonAnswerCard as ComparisonCardData,
  ComparisonMetric,
} from '../../../services/api';
import { Chip, PlayerAvatar } from '../../ui';
import { cn } from '../../../utils/cn';
import {
  CardFrame,
  CardLink,
  FixtureStrip,
  FormBars,
  Highlights,
  OpponentChip,
  PlayerAction,
  PlayerMeta,
  Section,
  formatValue,
  useCardPhotos,
} from './shared';

type Side = 0 | 1;

// Player 1 mint, player 2 cyan, as on the Compare page; position (left/right) and the
// names carry identity too, so colour is never the only cue.
const BAR = ['bg-accent', 'bg-accent-2'] as const;
const BAR_DIM = ['bg-accent/40', 'bg-accent-2/40'] as const;

function PlayerSide({ player, photo, side, season }: { player: CardPlayer; photo: string | null | undefined; side: Side; season: string | null }) {
  return (
    <div className={cn('flex min-w-0 flex-col gap-1.5', side === 1 ? 'items-end text-right' : 'items-start')}>
      <span aria-hidden="true" className={cn('h-1 w-10 rounded-full', BAR[side])} />
      <PlayerAction
        player={player}
        season={season}
        layout="stack"
        iconSide={side === 1 ? 'left' : 'right'}
        className={cn('flex flex-col gap-1.5', side === 1 ? 'items-end text-right' : 'items-start')}
      >
        <PlayerAvatar name={player.name} src={photo} size={56} loading="eager" className="ring-1 ring-line" />
        <span className="block w-full break-words font-display text-body font-bold leading-tight text-fg group-hover:underline sm:text-h3">
          {player.name}
        </span>
      </PlayerAction>
      <PlayerMeta player={player} className={cn(side === 1 && 'justify-end')} />
    </div>
  );
}

function MetricRow({ metric, names }: { metric: ComparisonMetric; names: [string, string] }) {
  const [a, b] = metric.values;
  const max = Math.max(Math.abs(a ?? 0), Math.abs(b ?? 0));
  const width = (v: number | null) => (v != null && max > 0 ? `${Math.max(2, (Math.abs(v) / max) * 100)}%` : '0%');

  const value = (v: number | null, side: Side) => {
    const better = metric.better === side;
    return (
      <span
        className={cn(
          'inline-flex items-center gap-1 text-data tabular-nums',
          side === 1 && 'flex-row-reverse',
          better ? 'font-bold text-fg' : 'text-fg-muted',
        )}
      >
        {better && <Check className="h-3.5 w-3.5 text-fg" aria-hidden="true" />}
        {formatValue(v, metric.format)}
      </span>
    );
  };

  return (
    <li>
      <p className="sr-only">
        {metric.label}: {names[0]} {formatValue(a, metric.format)}, {names[1]} {formatValue(b, metric.format)}
        {metric.better != null ? `; ${names[metric.better]} higher` : ''}.
      </p>
      <div aria-hidden="true">
        <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-baseline gap-2">
          <span className="text-left">{value(a, 0)}</span>
          <span className="text-center text-micro font-semibold uppercase text-fg-subtle">{metric.label}</span>
          <span className="text-right">{value(b, 1)}</span>
        </div>
        <div className="mt-1 flex gap-[2px]">
          <span className="flex h-1.5 flex-1 justify-end overflow-hidden rounded-l-full bg-surface-2">
            <span className={cn('h-full rounded-l-full', metric.better === 1 || !metric.scored ? BAR_DIM[0] : BAR[0])} style={{ width: width(a) }} />
          </span>
          <span className="flex h-1.5 flex-1 overflow-hidden rounded-r-full bg-surface-2">
            <span className={cn('h-full rounded-r-full', metric.better === 0 || !metric.scored ? BAR_DIM[1] : BAR[1])} style={{ width: width(b) }} />
          </span>
        </div>
      </div>
    </li>
  );
}

function PredictionSide({ prediction, side, leads }: { prediction: CardPrediction | null; side: Side; leads: boolean }) {
  if (!prediction) {
    return <p className={cn('text-data text-fg-subtle', side === 1 && 'text-right')}>Not in the current game</p>;
  }
  return (
    <div className={cn('flex min-w-0 flex-col gap-1', side === 1 ? 'items-end text-right' : 'items-start')}>
      <p className={cn('font-display text-h2 font-extrabold leading-none', leads ? 'text-fg' : 'text-fg-muted')}>
        {prediction.points.toFixed(1)}
        <span className="ml-1 text-data font-semibold text-fg-subtle">pts</span>
      </p>
      <div className={cn('flex flex-wrap gap-1', side === 1 && 'justify-end')}>
        {prediction.opponents.length ? (
          prediction.opponents.map((o, i) => <OpponentChip key={`${o.team}-${i}`} opponent={o} />)
        ) : (
          <Chip tone="warning">No fixture</Chip>
        )}
      </div>
    </div>
  );
}

/** Two players side by side: mirrored bars per stat, next-gameweek predictions, recent form. */
export function ComparisonAnswerCard({ card }: { card: ComparisonCardData }) {
  const photoOf = useCardPhotos(card.players);
  const [a, b] = card.players;
  const names: [string, string] = [a.name, b.name];
  const [pa, pb] = card.predictions;
  const predictionLead = pa && pb && pa.points !== pb.points ? (pa.points > pb.points ? 0 : 1) : null;
  const nextFixtures = card.next_fixtures ?? [[], []];
  const scored = card.metrics.filter((m) => m.scored).length;
  const predictionsSection = (pa || pb) && (
    <Section
      title={`Next gameweek${(pa ?? pb) ? ` · GW ${(pa ?? pb)!.gameweek}` : ''}`}
      aside="Model prediction"
      className={cn(card.emphasis === 'prediction' && 'bg-brand/70')}
    >
      <div className="grid grid-cols-2 gap-3">
        <PredictionSide prediction={pa} side={0} leads={predictionLead === 0} />
        <PredictionSide prediction={pb} side={1} leads={predictionLead === 1} />
      </div>
    </Section>
  );

  return (
    <CardFrame
      eyebrow={`Comparison · ${card.season_label}`}
      identity={(titleId) => (
        <>
          <h3 id={titleId} className="sr-only">
            {a.name} compared with {b.name}
          </h3>
          <div className="grid grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)] items-start gap-2">
            <PlayerSide player={a} photo={photoOf(a)} side={0} season={card.season} />
            <span className="mt-8 rounded-full bg-brand px-2 py-1 text-micro font-bold text-fg-muted ring-1 ring-brand-2" aria-hidden="true">
              VS
            </span>
            <PlayerSide player={b} photo={photoOf(b)} side={1} season={card.season} />
          </div>
          {scored > 0 && (
            <p className="mt-3 text-data text-fg-muted">
              <span className="font-semibold text-fg">{a.name}</span> leads {card.leads[0]} of {scored} stats ·{' '}
              <span className="font-semibold text-fg">{b.name}</span> leads {card.leads[1]}
            </p>
          )}
        </>
      )}
      footer={
        <CardLink to="/compare" state={{ player1: a.name, player2: b.name, season: card.season ?? '' }}>
          Open in Compare
        </CardLink>
      }
    >
      {card.emphasis === 'prediction' && predictionsSection}

      <Section title="Head to head" aside={card.season_label}>
        <ul className="space-y-3">
          {card.metrics.map((m) => (
            <MetricRow key={m.key} metric={m} names={names} />
          ))}
        </ul>
      </Section>

      {card.emphasis !== 'prediction' && predictionsSection}

      {nextFixtures.some((side) => side.length > 0) && (
        <Section
          title="Next fixtures"
          aside={nextFixtures.some((side) => side.some((f) => f.difficulty != null)) ? 'Colour = FPL difficulty' : undefined}
        >
          <div className="grid grid-cols-2 gap-3">
            {nextFixtures.map((fixtures, side) => (
              <div key={side} className={cn('min-w-0', side === 1 && 'flex justify-end')}>
                {fixtures.length ? (
                  <FixtureStrip fixtures={fixtures} className={cn(side === 1 && 'justify-end')} />
                ) : (
                  <p className="text-data text-fg-subtle">No fixtures</p>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {(card.form[0].length > 0 || card.form[1].length > 0) && (
        <Section title="Recent form" aside="Points per match">
          <div className="grid grid-cols-2 gap-4">
            {card.form.map((matches, side) =>
              matches.length ? (
                <FormBars
                  key={side}
                  matches={matches}
                  tone={side === 0 ? 'accent' : 'accent-2'}
                  height={40}
                  caption={`${names[side]}'s points in recent matches`}
                />
              ) : (
                <p key={side} className="self-end text-data text-fg-subtle">
                  No matches
                </p>
              ),
            )}
          </div>
        </Section>
      )}

      <Highlights items={card.highlights} />
    </CardFrame>
  );
}
