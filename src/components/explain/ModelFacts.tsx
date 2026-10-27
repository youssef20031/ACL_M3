import type { ReactNode } from 'react';
import type { ExplainMetrics, ExplainSummary } from '../../services/api';
import { Card, Chip, SectionHeading } from '../ui';
import { ImportanceBars, type ImportanceItem } from './ImportanceBars';
import { formatDate, formatDateTime, seasonRange } from './format';

// The model's vital statistics and its held-out test results against simple baselines.

/** Methods compared, in display order; keys missing from the payload are skipped. */
const METHODS: Array<{ key: string; label: string }> = [
  { key: 'v3', label: 'Points model' },
  { key: 'last3_avg', label: 'Average of last 3 matches' },
  { key: 'last6_avg', label: 'Average of last 6 matches' },
  { key: 'season_ppg', label: 'Points per match this season' },
];

const METRICS: Array<{ key: keyof Pick<ExplainMetrics, 'top20_points' | 'captain_points'>; title: string; caption: string }> = [
  {
    key: 'top20_points',
    title: 'Top 20 picks',
    caption: 'Average actual points of each method’s top 20 picks per gameweek.',
  },
  {
    key: 'captain_points',
    title: 'Captain pick',
    caption: 'Average actual points of each method’s single top pick per gameweek.',
  },
];

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0 bg-surface px-3 py-2">
      <dt className="text-micro font-semibold uppercase text-fg-subtle">{label}</dt>
      <dd className="mt-0.5 truncate text-data font-semibold text-fg">{children}</dd>
    </div>
  );
}

export function ModelFacts({ summary }: { summary: ExplainSummary }) {
  const { model } = summary;
  const methods = METHODS.filter((m) => model.test_metrics[m.key]);
  const trained = formatDate(model.trained_at);
  const generated = formatDateTime(summary.generated_at);
  const seasons = seasonRange(model.seasons);

  return (
    <Card>
      <SectionHeading
        as="h2"
        size="md"
        title="The model"
        subtitle="Gradient-boosted trees (XGBoost) predicting each player’s points per fixture."
        action={<Chip tone={model.source === 'live' ? 'accent' : 'neutral'}>{model.source === 'live' ? 'Retrained weekly' : 'Bundled'}</Chip>}
      />
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line">
        <Fact label="Version">{model.version}</Fact>
        <Fact label="Trained">{trained ?? '–'}</Fact>
        <Fact label="Trees">{model.n_trees != null ? model.n_trees.toLocaleString('en-GB') : '–'}</Fact>
        <Fact label="Training rows">{model.rows != null ? model.rows.toLocaleString('en-GB') : '–'}</Fact>
        <div className="col-span-2 min-w-0 bg-surface px-3 py-2">
          <dt className="text-micro font-semibold uppercase text-fg-subtle">Seasons</dt>
          <dd className="mt-0.5 text-data font-semibold text-fg">
            {seasons ?? '–'}
            {model.seasons.length > 1 && <span className="font-normal text-fg-subtle"> · {model.seasons.length} seasons</span>}
          </dd>
        </div>
      </dl>
      <p className="mt-2 text-micro text-fg-subtle">
        Explaining {summary.players.toLocaleString('en-GB')} players across {summary.rows.toLocaleString('en-GB')} fixtures in gameweek{' '}
        {summary.gameweek}
        {generated ? `, worked out ${generated}` : ''}.
      </p>

      {methods.length > 0 && (
        <div className="mt-5 space-y-5 border-t border-line pt-4">
          <div>
            <h3 className="font-display text-h3 font-bold text-fg">How it tested</h3>
            <p className="mt-0.5 text-data text-fg-subtle">On held-out matches the tested model never saw in training, against simple rules of thumb.</p>
          </div>
          {METRICS.map((metric) => {
            const items: ImportanceItem[] = methods.map((m) => ({
              key: m.key,
              label: m.key === 'v3' ? `${m.label} (${model.version})` : m.label,
              value: model.test_metrics[m.key][metric.key],
              tone: m.key === 'v3' ? 'accent' : 'baseline',
            }));
            return (
              <section key={metric.key} aria-label={metric.title}>
                <h4 className="text-micro font-semibold uppercase text-fg-muted">{metric.title}</h4>
                <p className="mb-2 text-micro text-fg-subtle">{metric.caption}</p>
                <ImportanceBars items={items} label={`${metric.title}: ${metric.caption}`} axis={false} />
              </section>
            );
          })}
        </div>
      )}
    </Card>
  );
}
