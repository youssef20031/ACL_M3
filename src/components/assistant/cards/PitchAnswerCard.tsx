import type { PitchAnswerCard as PitchCardData, PitchSlot } from '../../../services/api';
import { Pitch, PitchPlayer, PitchRow, PlayerAvatar, PositionBadge } from '../../ui';
import { CardFrame, CardLink, Highlights, Section, useCardPhotos } from './shared';

const LINES = [
  { position: 'GK', label: 'Goalkeeper' },
  { position: 'DEF', label: 'Defenders' },
  { position: 'MID', label: 'Midfielders' },
  { position: 'FWD', label: 'Forwards' },
] as const;

/** Best XI on a compact pitch, with the bench and the squad's totals. */
export function PitchAnswerCard({ card }: { card: PitchCardData }) {
  const photoOf = useCardPhotos([...card.starters, ...card.bench].map((s) => s.player));
  const predicted = card.mode === 'predicted';
  const pts = (v: number | null) => (v == null ? '–' : predicted ? v.toFixed(1) : String(Math.round(v)));
  const label = (s: PitchSlot) => `${pts(s.points)}${predicted ? '' : ' pts'}`;

  return (
    <CardFrame
      icon={predicted ? 'star' : 'trophy'}
      eyebrow={`Gameweek ${card.gameweek} · ${predicted ? 'predicted' : 'actual points'}`}
      title={predicted ? 'Best squad' : 'Dream team'}
      aside={
        <p className="text-right">
          <span className="block font-display text-h2 font-extrabold leading-none text-accent">{pts(card.total_points)}</span>
          <span className="text-micro uppercase text-fg-subtle">{predicted ? 'pts, captain ×2' : 'pts'}</span>
        </p>
      }
      footer={
        <CardLink to={predicted ? '/team-of-the-week' : `/team-of-the-week?mode=actual&gw=${card.gameweek}`}>
          Team of the week
        </CardLink>
      }
    >
      <Section className="px-2 sm:px-3">
        <p className="mb-2 px-2 text-data text-fg-muted">
          {card.formation}
          {predicted && card.total_cost != null && card.budget != null
            ? ` · £${card.total_cost.toFixed(1)}m of £${card.budget.toFixed(1)}m, 3 per club at most`
            : ' · best XI by points scored'}
        </p>
        <Pitch compact label={`${predicted ? 'Predicted best' : 'Dream'} team, gameweek ${card.gameweek}`}>
          {LINES.map((line) => (
            <PitchRow compact key={line.position} label={line.label}>
              {card.starters
                .filter((s) => s.player.position === line.position)
                .map((s) => (
                  <PitchPlayer
                    compact
                    key={s.player.name}
                    name={s.player.name}
                    team={s.player.team}
                    photo={photoOf(s.player)}
                    opponents={s.opponents}
                    captain={s.captain}
                    viceCaptain={s.vice_captain}
                    status={s.player.status}
                    statusNote={s.player.news}
                    value={label(s)}
                  />
                ))}
            </PitchRow>
          ))}
        </Pitch>
      </Section>

      {card.bench.length > 0 && (
        <Section title="Bench" aside="Cheapest picks that complete the squad">
          <ol className="grid grid-cols-2 gap-2 [@container(min-width:36rem)]:grid-cols-4">
            {card.bench.map((s) => (
              <li key={s.player.name} className="flex min-w-0 items-center gap-2 rounded-md border border-line bg-bg/40 p-1.5">
                <PlayerAvatar name={s.player.name} src={photoOf(s.player)} size={28} />
                <span className="min-w-0">
                  <span className="block truncate text-data font-semibold text-fg">{s.player.name}</span>
                  <span className="flex items-center gap-1 text-micro text-fg-subtle">
                    <PositionBadge position={s.player.position} className="w-auto px-1" />
                    {s.cost != null && `£${s.cost.toFixed(1)}m`}
                  </span>
                </span>
              </li>
            ))}
          </ol>
        </Section>
      )}

      <Highlights items={card.highlights} />
    </CardFrame>
  );
}
