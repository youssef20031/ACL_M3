import { useId, type ReactNode } from 'react';
import { ArrowRight, CalendarOff, UserRound } from 'lucide-react';
import type { ExplainFixture, PlayerExplanation } from '../../services/api';
import { usePlayerAvatars } from '../../hooks/usePlayerAvatars';
import { cn } from '../../utils/cn';
import { getClubAbbreviation } from '../../utils/clubLogos';
import { ButtonLink, Card, ClubCrest, EmptyState, PlayerAvatar, PositionBadge, StatusBadge } from '../ui';
import { LimePanel } from './LimePanel';
import { ShapVsLime } from './ShapVsLime';
import { Waterfall } from './Waterfall';
import { formatDateTime, points } from './format';

// The "Explain a player" view: who he is and his predicted points, how the fixtures' model
// outputs become that number, then a SHAP waterfall and a LIME panel per fixture.

function PanelHeading({ title, children }: { title: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-4">
      <h4 className="font-display text-h3 font-bold text-fg">{title}</h4>
      {children && <p className="mt-1 text-data text-fg-subtle">{children}</p>}
    </div>
  );
}

interface AdjustStep {
  label: string;
  value: number;
}

/** Fixtures' model outputs → floored at 0 per fixture → × chance of playing → the prediction. Unchanged steps are skipped. */
function Adjustments({ explanation }: { explanation: PlayerExplanation }) {
  const { fixtures, availability_factor: factor, predicted_points: predicted } = explanation;
  if (!fixtures.length) return null;
  const raw = fixtures.reduce((s, f) => s + f.model_points, 0);
  const floored = explanation.model_points;
  const steps: AdjustStep[] = [
    {
      label: fixtures.length > 1 ? `Model output, ${fixtures.length} fixtures (${fixtures.map((f) => points(f.model_points)).join(' + ')})` : 'Model output',
      value: raw,
    },
  ];
  if (Math.abs(floored - raw) >= 0.005) {
    steps.push({ label: fixtures.length > 1 ? 'Negative fixtures count as 0' : 'Negative output counts as 0', value: floored });
  }
  if (factor < 0.995) {
    steps.push({ label: `× ${Math.round(factor * 100)}% chance of playing`, value: floored * factor });
  }

  if (steps.length === 1 && Math.abs(raw - predicted) < 0.05) {
    return (
      <p className="mt-4 border-t border-line pt-3 text-data text-fg-muted">
        The prediction is the model’s output for {fixtures.length > 1 ? 'his fixtures' : 'this fixture'}, with no adjustments.
      </p>
    );
  }

  return (
    <div className="mt-4 border-t border-line pt-3">
      <p className="mb-2 text-micro font-semibold uppercase text-fg-subtle">From model output to prediction</p>
      <ol className="flex flex-wrap items-center gap-x-2 gap-y-2 text-data">
        {steps.map((s) => (
          <li key={s.label} className="flex items-center gap-2">
            <span className="rounded-md border border-line bg-surface-2 px-2 py-1">
              <span className="text-fg-muted">{s.label}</span> <span className="font-semibold tabular-nums text-fg">{points(s.value)}</span>
            </span>
            <ArrowRight className="h-4 w-4 shrink-0 text-fg-subtle" aria-hidden="true" />
          </li>
        ))}
        <li className="rounded-md border border-accent/40 bg-accent/10 px-2 py-1">
          <span className="text-fg-muted">Predicted</span> <span className="font-bold tabular-nums text-accent">{points(predicted)}</span>
        </li>
      </ol>
    </div>
  );
}

function FixtureTitle({ fixture }: { fixture: ExplainFixture }) {
  const kickoff = formatDateTime(fixture.kickoff_time);
  const abbr = getClubAbbreviation(fixture.opponent) || fixture.opponent;
  return (
    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <span className="text-fg-subtle">{fixture.is_home ? 'vs' : 'at'}</span>
      <ClubCrest team={fixture.opponent} size={22} />
      <span>
        <span className="sm:hidden">{abbr}</span>
        <span className="hidden sm:inline">{fixture.opponent}</span>
        <span className="text-fg-muted" aria-hidden="true">
          {' '}
          ({fixture.is_home ? 'H' : 'A'})
        </span>
        <span className="sr-only">{fixture.is_home ? ', at home' : ', away'}</span>
      </span>
      {kickoff && <span className="text-data font-normal text-fg-subtle">· {kickoff}</span>}
    </span>
  );
}

function FixtureSection({
  fixture,
  playerName,
  descriptions,
}: {
  fixture: ExplainFixture;
  playerName: string;
  descriptions: Record<string, string>;
}) {
  const headingId = useId();
  const where = `${fixture.is_home ? 'against' : 'away at'} ${fixture.opponent}`;
  return (
    <section aria-labelledby={headingId} className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-1">
        <h3 id={headingId} className="font-display text-h2 font-bold text-fg">
          <FixtureTitle fixture={fixture} />
        </h3>
        <p className="text-data text-fg-muted">
          Model output <span className="font-bold tabular-nums text-fg">{points(fixture.model_points)}</span>
        </p>
      </div>
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <Card>
          <PanelHeading title="SHAP: where the points come from">
            Starts at the base value and adds each feature’s contribution. They sum exactly to the model’s output.
          </PanelHeading>
          <Waterfall fixture={fixture} label={`SHAP waterfall for ${playerName} ${where}`} descriptions={descriptions} />
        </Card>
        <Card>
          <PanelHeading title="LIME: a local approximation">
            LIME fits a simple weighted linear model to the points model’s outputs on{' '}
            {fixture.lime.num_samples.toLocaleString('en-GB')} perturbed copies of this player (features resampled from this
            gameweek’s players), so its weights describe the model’s behaviour near him. Unlike SHAP they don’t have to add up
            to the prediction, and a low R² means the approximation is rough.
          </PanelHeading>
          <LimePanel lime={fixture.lime} label={`LIME weights for ${playerName} ${where}`} />
        </Card>
        <Card className="lg:col-span-2">
          <PanelHeading title="SHAP and LIME side by side">
            The two methods measure different things, so compare the direction, not the size.
          </PanelHeading>
          <ShapVsLime fixture={fixture} />
        </Card>
      </div>
    </section>
  );
}

interface PlayerExplainerProps {
  explanation: PlayerExplanation;
  /** Feature descriptions (from the summary) for hover text. */
  descriptions?: Record<string, string>;
  /** Off on /design, whose sample players are made up. */
  lookupPhotos?: boolean;
  /** Another player's explanation is on screen while this one loads. */
  updating?: boolean;
}

export function PlayerExplainer({ explanation, descriptions = {}, lookupPhotos = true, updating = false }: PlayerExplainerProps) {
  const { player, fixtures, gameweek } = explanation;
  const photos = usePlayerAvatars(lookupPhotos ? [player.name] : []);
  const unavailable = player.status != null && player.status !== 'a';

  return (
    <div className={cn('space-y-8 transition-opacity', updating && 'opacity-60')} aria-busy={updating}>
      <Card>
        <div className="flex items-start gap-3 sm:gap-4">
          <PlayerAvatar name={player.name} src={photos[player.name]} size={56} />
          <div className="min-w-0 flex-1">
            <h2 className="font-display text-h2 font-bold text-fg">{player.name}</h2>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-data text-fg-muted">
              <ClubCrest team={player.team} size={18} label="full" />
              <PositionBadge position={player.position} />
              <StatusBadge status={player.status} chance={player.chance_of_playing_next_round} news={player.news} />
            </div>
          </div>
          <div className="shrink-0 text-right">
            <span className="block font-display text-h1 font-extrabold tabular-nums text-accent sm:text-display">
              {explanation.predicted_points.toFixed(1)}
            </span>
            <span className="block text-micro uppercase text-fg-subtle">
              Pred. pts<span className="hidden sm:inline"> · GW{gameweek}</span>
            </span>
          </div>
        </div>
        {unavailable && player.news && <p className="mt-3 text-data text-warning">{player.news}</p>}
        <Adjustments explanation={explanation} />
        <div className="mt-4 flex flex-wrap gap-2">
          <ButtonLink
            to="/players"
            state={{ player: { player_name: player.name, team_name: player.team, position: player.position }, returnTo: 'history' }}
            variant="secondary"
            size="sm"
            icon={<UserRound className="h-4 w-4" aria-hidden="true" />}
          >
            Player profile
          </ButtonLink>
        </div>
      </Card>

      {fixtures.length === 0 ? (
        <Card padded={false}>
          <EmptyState
            icon={<CalendarOff className="h-8 w-8" aria-hidden="true" />}
            title="No fixture this gameweek"
            description={`${player.team} don’t play in gameweek ${gameweek}, so the model has nothing to explain and he’s predicted 0 points.`}
          />
        </Card>
      ) : (
        fixtures.map((f) => <FixtureSection key={f.fixture} fixture={f} playerName={player.name} descriptions={descriptions} />)
      )}
    </div>
  );
}
