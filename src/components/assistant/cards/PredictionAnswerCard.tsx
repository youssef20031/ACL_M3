import { Link } from 'react-router-dom';
import type { CardPlayer, PredictionAnswerCard as PredictionCardData, PredictionRow } from '../../../services/api';
import { Chip, ClubCrest, PlayerAvatar } from '../../ui';
import { cn } from '../../../utils/cn';
import {
  CardFrame,
  CardLink,
  Highlights,
  OpponentChip,
  PlayerMeta,
  Section,
  ValueBar,
  playerProfileState,
  useCardPhotos,
} from './shared';

const POSITION_NAMES: Record<string, string> = { GK: 'Goalkeepers', DEF: 'Defenders', MID: 'Midfielders', FWD: 'Forwards' };

function Opponents({ row }: { row: PredictionRow }) {
  if (!row.opponents.length) return <Chip tone="warning">No fixture</Chip>;
  return (
    <>
      {row.opponents.map((o, i) => (
        <OpponentChip key={`${o.team}-${i}`} opponent={o} />
      ))}
    </>
  );
}

/** "in 1d 4h" until the deadline, worked out when the card renders. */
function Deadline({ iso }: { iso: string }) {
  const target = new Date(iso);
  const ms = target.getTime() - Date.now();
  if (Number.isNaN(ms)) return null;
  const hours = Math.floor(ms / 3_600_000);
  const text = ms <= 0 ? 'passed' : hours >= 24 ? `in ${Math.floor(hours / 24)}d ${hours % 24}h` : `in ${hours}h ${Math.floor((ms % 3_600_000) / 60_000)}m`;
  return (
    <p className="text-micro uppercase text-fg-subtle">
      Deadline
      <time dateTime={target.toISOString()} className="block font-display text-body font-bold normal-case text-fg">
        {text}
      </time>
    </p>
  );
}

function TopPick({ card, row, photo }: { card: PredictionCardData; row: PredictionRow; photo: string | null | undefined }) {
  return (
    <div className="mx-4 mb-4 rounded-lg border border-brand-2 bg-brand p-3 sm:mx-5 sm:p-4">
      <div className="flex items-center gap-3">
        <span className="relative shrink-0">
          <PlayerAvatar name={row.player.name} src={photo} size={72} loading="eager" className="ring-2 ring-accent" />
          {row.player.team && (
            <span className="absolute -bottom-1.5 -right-1.5 rounded-full bg-brand p-0.5 ring-1 ring-brand-2">
              <ClubCrest team={row.player.team} size={22} />
            </span>
          )}
          {card.captain && (
            <span className="absolute -left-1 -top-1 flex h-6 w-6 items-center justify-center rounded-full bg-accent text-micro font-extrabold text-accent-on ring-2 ring-brand">
              <span aria-hidden="true">C</span>
              <span className="sr-only">Captain pick</span>
            </span>
          )}
        </span>
        <div className="min-w-0 flex-1">
          <p className="text-micro font-semibold uppercase text-accent">{card.captain ? 'Captain pick' : 'Top pick'}</p>
          <Link
            to="/players"
            state={playerProfileState(row.player, null)}
            className="block break-words font-display text-h3 font-bold leading-tight text-fg hover:underline sm:text-h2"
          >
            {row.player.name}
          </Link>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <PlayerMeta player={row.player} />
            <Opponents row={row} />
          </div>
        </div>
        <p className="shrink-0 text-right font-display text-display font-extrabold leading-none text-accent">
          {row.points.toFixed(1)}
          <span className="block text-micro font-semibold uppercase text-fg-subtle">pts</span>
        </p>
      </div>
    </div>
  );
}

function RowItem({ row, photo, max }: { row: PredictionRow; photo: string | null | undefined; max: number }) {
  return (
    <li>
      <Link
        to="/players"
        state={playerProfileState(row.player, null)}
        className="group grid grid-cols-[1.5rem_2rem_minmax(0,1fr)_auto] items-center gap-2 rounded-md px-1 py-1.5 transition-colors hover:bg-surface-2 sm:gap-3"
      >
        <span className="text-center text-data font-semibold tabular-nums text-fg-subtle">
          <span className="sr-only">Rank </span>
          {row.rank}
        </span>
        <PlayerAvatar name={row.player.name} src={photo} size={32} />
        <span className="min-w-0">
          <span className="block truncate text-data font-semibold text-fg group-hover:underline">{row.player.name}</span>
          <span className="mt-0.5 flex flex-wrap items-center gap-1">
            <PlayerMeta player={row.player} />
            <Opponents row={row} />
          </span>
        </span>
        <span className="flex w-16 flex-col items-end">
          <span className="font-display text-body font-bold tabular-nums text-fg">
            {row.points.toFixed(1)}
            <span className="sr-only"> predicted points</span>
          </span>
          {row.price != null && <span className="text-micro text-fg-subtle">£{row.price.toFixed(1)}m</span>}
          <ValueBar value={row.points} max={max} className="mt-1" />
        </span>
      </Link>
    </li>
  );
}

/** Next-gameweek predictions: the top pick as a hero, the next best with fixtures (FDR) and prices. */
export function PredictionAnswerCard({ card }: { card: PredictionCardData }) {
  const photoOf = useCardPhotos(card.rows.map((r) => r.player));
  const [top, ...rest] = card.rows;
  const max = Math.max(0, ...card.rows.map((r) => r.points));
  const photo = (p: CardPlayer) => photoOf(p);

  return (
    <CardFrame
      icon="star"
      eyebrow={`Gameweek ${card.gameweek}${card.position ? ` · ${POSITION_NAMES[card.position] ?? card.position}` : ''}`}
      title={card.title}
      aside={card.deadline ? <Deadline iso={card.deadline} /> : undefined}
      footer={
        <>
          <CardLink to="/predictions">All predictions</CardLink>
          <CardLink to="/team-of-the-week">Team of the week</CardLink>
        </>
      }
    >
      {top && <TopPick card={card} row={top} photo={photo(top.player)} />}
      {rest.length > 0 && (
        <Section title="Next best" aside="Predicted points">
          <ol start={2}>
            {rest.map((row) => (
              <RowItem key={row.player.name} row={row} photo={photo(row.player)} max={max} />
            ))}
          </ol>
        </Section>
      )}
      <Highlights items={card.highlights} />
      {card.model && (
        <p className={cn('border-t border-line px-4 py-2 text-micro text-fg-subtle sm:px-5')}>
          Predicted by the Pitch Me model ({card.model}). Points are scaled by FPL&rsquo;s chance of playing.
        </p>
      )}
    </CardFrame>
  );
}
