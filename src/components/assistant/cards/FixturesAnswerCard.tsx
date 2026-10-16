import type { ReactNode } from 'react';
import type { FixturesAnswerCard as FixturesCardData } from '../../../services/api';
import { ClubCrest, FixtureRow } from '../../ui';
import { cn } from '../../../utils/cn';
import { CardFrame, CardLink, Highlights, ResultPill, Section, teamCode } from './shared';

/** "Arsenal wins" from sm up, "ARS wins" on phones, where the full name truncated. */
function WinsLabel({ team }: { team: string }) {
  return (
    <>
      <span className="sm:hidden" aria-hidden="true">
        {teamCode(team)} wins
      </span>
      <span className="sr-only sm:not-sr-only">{team} wins</span>
    </>
  );
}

function Count({ value, label }: { value: number; label: ReactNode }) {
  // dt before dd in the DOM; the number shows on top
  return (
    <div className="flex min-w-0 flex-col-reverse">
      <dt className="mt-1 truncate text-micro uppercase text-fg-subtle">{label}</dt>
      <dd className="font-display text-h2 font-extrabold leading-none text-fg">{value}</dd>
    </div>
  );
}

/** A team's results, or two teams' head-to-head: record, a W/D/L split bar, recent scorelines. */
export function FixturesAnswerCard({ card }: { card: FixturesCardData }) {
  const [team, opponent] = card.teams;
  const h2h = card.teams.length > 1;
  const r = card.record;
  // Head to head: each side's wins in its identity colour (mint / cyan). One team's results:
  // losses in the danger tone. The counts above always say which is which.
  const segments = [
    { key: 'won', value: r.won, className: 'bg-accent' },
    { key: 'drawn', value: r.drawn, className: 'bg-fg-subtle/40' },
    { key: 'lost', value: r.lost, className: h2h ? 'bg-accent-2' : 'bg-danger' },
  ].filter((s) => s.value > 0);

  return (
    <CardFrame
      icon="calendar"
      eyebrow={card.subtitle}
      title={card.title}
      footer={<CardLink to="/fixtures">Fixtures and results</CardLink>}
    >
      <Section>
        {h2h && (
          <div className="mb-3 flex items-center justify-between gap-3 text-data font-semibold text-fg">
            <span className="flex min-w-0 items-center gap-2">
              <ClubCrest team={team} size={28} />
              <span className="truncate">{team}</span>
            </span>
            <span className="flex min-w-0 items-center justify-end gap-2 text-right">
              <span className="truncate">{opponent}</span>
              <ClubCrest team={opponent} size={28} />
            </span>
          </div>
        )}
        <dl className="grid grid-cols-3 gap-2 text-center">
          <Count value={r.won} label={h2h ? <WinsLabel team={team} /> : 'Won'} />
          <Count value={r.drawn} label={r.drawn === 1 ? 'Draw' : 'Draws'} />
          <Count value={r.lost} label={h2h ? <WinsLabel team={opponent} /> : 'Lost'} />
        </dl>
        {segments.length > 0 && (
          <div aria-hidden="true" className="mt-3 flex h-2 gap-[2px] overflow-hidden rounded-full">
            {segments.map((s) => (
              <span key={s.key} className={cn('h-full', s.className)} style={{ flexGrow: s.value }} />
            ))}
          </div>
        )}
        <p className="mt-2 text-center text-data text-fg-muted">
          {r.played} {r.played === 1 ? 'match' : 'matches'} · goals {r.goals_for}–{r.goals_against}
          {h2h ? ` (${team} first)` : ''}
        </p>
      </Section>

      <Section title={card.total > card.matches.length ? `Latest ${card.matches.length} of ${card.total}` : 'Matches'} className="pb-0">
        <ol className="-mx-4 sm:-mx-5">
          {card.matches.map((m, i) => (
            <li key={`${m.season}-${m.gameweek}-${i}`} className="border-t border-line first:border-0">
              <p className="flex items-center justify-between gap-2 px-4 pt-2 text-micro text-fg-subtle sm:px-5">
                <span>
                  {m.season} · GW {m.gameweek ?? '?'}
                </span>
                {!h2h && <ResultPill result={m.result} className="h-5 w-5" />}
              </p>
              <FixtureRow
                home={m.home_team}
                away={m.away_team}
                homeScore={m.home_score}
                awayScore={m.away_score}
                status="finished"
                className="border-0 pt-1"
              />
            </li>
          ))}
        </ol>
      </Section>

      <Highlights items={card.highlights} />
    </CardFrame>
  );
}
