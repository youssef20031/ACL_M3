import type { CardPlayer, LeaderboardAnswerCard as LeaderboardCardData, LeaderboardRow } from '../../../services/api';
import { Chip, PlayerAvatar, PositionBadge } from '../../ui';
import { cn } from '../../../utils/cn';
import { CardFrame, CardLink, Highlights, PlayerAction, PlayerMeta, Section, ValueBar, formatValue, useCardPhotos } from './shared';

const GROUP_NAMES: Record<string, string> = { GK: 'Goalkeepers', DEF: 'Defenders', MID: 'Midfielders', FWD: 'Forwards' };

type PhotoOf = (player: CardPlayer) => string | null | undefined;

function secondary(card: LeaderboardCardData, row: LeaderboardRow): string {
  const parts = card.columns.map((c) => `${c.label} ${formatValue(row.values[c.key], c.format)}`);
  if (row.detail) parts.unshift(row.detail);
  return parts.join(' · ');
}

/** Top three as a podium (2-1-3 on screen, 1-2-3 for screen readers); block height follows the value. */
function Podium({ card, rows, photoOf, max }: { card: LeaderboardCardData; rows: LeaderboardRow[]; photoOf: PhotoOf; max: number }) {
  const order = ['order-2', 'order-1', 'order-3'];
  return (
    <ol className="flex items-end gap-2 px-4 pb-4 sm:gap-3 sm:px-5" aria-label={`Top three by ${card.metric.label.toLowerCase()}`}>
      {rows.map((row, i) => {
        const first = i === 0;
        const share = max > 0 && row.value != null ? Math.abs(row.value) / max : 0;
        return (
          <li key={row.player.name} className={cn('flex min-w-0 flex-1 flex-col items-center text-center', order[i])}>
            <PlayerAction
              player={row.player}
              season={card.season}
              layout="stack"
              className="flex w-full min-w-0 flex-col items-center rounded-md pb-2"
            >
              <span className="relative">
                <PlayerAvatar
                  name={row.player.name}
                  src={photoOf(row.player)}
                  size={first ? 64 : 52}
                  className={cn('ring-2', first ? 'ring-accent' : 'ring-line')}
                />
                <span
                  className={cn(
                    'absolute -top-2 left-1/2 inline-flex h-5 min-w-5 -translate-x-1/2 items-center justify-center rounded-full px-1 text-micro font-extrabold',
                    first ? 'bg-accent text-accent-on' : 'bg-fg text-brand',
                  )}
                >
                  <span className="sr-only">Rank </span>
                  {row.rank}
                </span>
              </span>
              <span className="mt-1.5 line-clamp-2 w-full break-words text-data font-semibold leading-tight text-fg group-hover:underline">
                {row.player.name}
              </span>
              <PlayerMeta player={row.player} className="mt-1 justify-center" />
            </PlayerAction>
            <div
              className={cn(
                'flex w-full flex-col items-center justify-start rounded-t-md pt-2 ring-1',
                first ? 'bg-brand ring-accent/50' : 'bg-brand/70 ring-brand-2',
              )}
              style={{ height: 52 + Math.round(share * 44) }}
            >
              <span className="font-display text-h2 font-extrabold leading-none text-fg">{formatValue(row.value, card.metric.format)}</span>
              <span className="mt-0.5 text-micro uppercase text-fg-subtle">{card.metric.label}</span>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function RowItem({ card, row, photoOf, max }: { card: LeaderboardCardData; row: LeaderboardRow; photoOf: PhotoOf; max: number }) {
  return (
    <li>
      <PlayerAction
        player={row.player}
        season={card.season}
        className="grid grid-cols-[1.5rem_2rem_minmax(0,1fr)_auto] items-center gap-2 rounded-md px-1 py-1.5 transition-colors hover:bg-surface-2 sm:gap-3"
      >
        <span className="text-center text-data font-semibold tabular-nums text-fg-subtle">
          {row.rank != null && (
            <>
              <span className="sr-only">Rank </span>
              {row.rank}
            </>
          )}
        </span>
        <PlayerAvatar name={row.player.name} src={photoOf(row.player)} size={32} />
        <span className="min-w-0">
          <span className="block truncate text-data font-semibold text-fg group-hover:underline">{row.player.name}</span>
          <PlayerMeta player={row.player} />
          <span className="mt-0.5 block truncate text-micro text-fg-subtle">{secondary(card, row)}</span>
        </span>
        <span className="flex w-16 flex-col items-end">
          <span className="font-display text-body font-bold tabular-nums text-fg">{formatValue(row.value, card.metric.format)}</span>
          <ValueBar value={row.value} max={max} className="mt-1" />
        </span>
      </PlayerAction>
    </li>
  );
}

/** A ranking: podium for the top three, then the rest with value bars; grouped by position for "top per position". */
export function LeaderboardAnswerCard({ card }: { card: LeaderboardCardData }) {
  const photoOf = useCardPhotos(card.rows.map((r) => r.player));
  const max = Math.max(0, ...card.rows.map((r) => Math.abs(r.value ?? 0)));
  const podium = !card.grouped && card.rows.length >= 3;
  const rest = podium ? card.rows.slice(3) : card.rows;
  const groups = card.grouped
    ? Object.entries(
        card.rows.reduce<Record<string, LeaderboardRow[]>>((acc, row) => {
          (acc[row.group ?? 'Other'] ??= []).push(row);
          return acc;
        }, {}),
      )
    : [];

  return (
    <CardFrame
      icon={card.icon}
      eyebrow={card.subtitle}
      title={card.title}
      aside={<Chip tone="brand">{card.metric.label}</Chip>}
      footer={<CardLink to="/players">Season leaders table</CardLink>}
    >
      {podium && <Podium card={card} rows={card.rows.slice(0, 3)} photoOf={photoOf} max={max} />}

      {card.grouped ? (
        <Section>
          <div className="grid gap-4 [@container(min-width:36rem)]:grid-cols-2">
            {groups.map(([group, rows]) => (
              <div key={group} className="min-w-0">
                <h4 className="mb-1 flex items-center gap-2 text-micro font-semibold uppercase text-fg-subtle">
                  <PositionBadge position={group} />
                  {GROUP_NAMES[group] ?? group}
                </h4>
                <ol>
                  {rows.map((row) => (
                    <RowItem key={row.player.name} card={card} row={row} photoOf={photoOf} max={max} />
                  ))}
                </ol>
              </div>
            ))}
          </div>
        </Section>
      ) : (
        rest.length > 0 && (
          <Section className={cn(podium && 'pt-2')}>
            <ol start={podium ? 4 : 1}>
              {rest.map((row) => (
                <RowItem key={row.player.name} card={card} row={row} photoOf={photoOf} max={max} />
              ))}
            </ol>
          </Section>
        )
      )}

      <Highlights items={card.highlights} />
    </CardFrame>
  );
}
