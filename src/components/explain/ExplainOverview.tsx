import { useId } from 'react';
import type { ExplainPlayerHit, ExplainSummary } from '../../services/api';
import { Card, SectionHeading } from '../ui';
import { Beeswarm } from './Beeswarm';
import { ImportanceBars, type ImportanceItem } from './ImportanceBars';
import { ModelFacts } from './ModelFacts';
import { PlayerChips } from './PlayerPicker';
import { points } from './format';

// The Overview tab: what drives this gameweek's predictions overall (global SHAP), the model's
// facts and test results, and quick picks into the player view.

interface ExplainOverviewProps {
  summary: ExplainSummary;
  onPickPlayer: (player: ExplainPlayerHit) => void;
}

export function HowToRead({ baseValue }: { baseValue?: number }) {
  return (
    <Card>
      <SectionHeading as="h2" size="md" title="How to read this" />
      <ul className="space-y-2 text-data text-fg-muted">
        <li>
          <span className="font-semibold text-fg">SHAP</span> splits each prediction into additive contributions, one per feature, that
          sum exactly to the model’s output. They start from the <span className="font-semibold text-fg">base value</span>, the model’s
          average prediction{baseValue != null ? ` (${points(baseValue)} points)` : ''}.
        </li>
        <li>A positive contribution adds points to that player’s prediction; a negative one takes points away.</li>
        <li>
          Importance here is the average size of a feature’s contributions over every player and fixture this gameweek, whichever way
          they push.
        </li>
        <li>
          <span className="font-semibold text-fg">LIME</span> (in the player view) is a second opinion: a simple model fitted around one
          player to see which rules matter near him.
        </li>
      </ul>
    </Card>
  );
}

export function ExplainOverview({ summary, onPickPlayer }: ExplainOverviewProps) {
  const groupsId = useId();
  const featuresId = useId();
  const descriptions = Object.fromEntries(summary.features.map((f) => [f.feature, f.description]));
  const groupItems: ImportanceItem[] = summary.groups.map((g) => ({ key: g.group, label: g.group, value: g.importance }));
  const featureItems: ImportanceItem[] = [...summary.features]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 15)
    .map((f) => ({ key: f.feature, label: f.label, detail: f.description, tooltip: `${f.label} (${f.group}): ${f.description}`, value: f.importance }));

  return (
    <div className="space-y-6">
      <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(18rem,1fr)]">
        <Card>
          <SectionHeading
            as="h2"
            title="What drives this gameweek’s predictions"
            subtitle={`Average size of each feature’s SHAP contribution, in points, over ${summary.rows.toLocaleString('en-GB')} player-fixtures in gameweek ${summary.gameweek}.`}
          />
          <section aria-labelledby={groupsId} className="mt-4">
            <h3 id={groupsId} className="mb-1 font-display text-h3 font-bold text-fg">
              By kind of feature
            </h3>
            <p className="mb-3 text-data text-fg-subtle">Each family’s features added together.</p>
            <ImportanceBars items={groupItems} label="Importance by feature family, points" tone="accent-2" axisCaption="Mean |SHAP|, points" />
          </section>
          <section aria-labelledby={featuresId} className="mt-8">
            <h3 id={featuresId} className="mb-1 font-display text-h3 font-bold text-fg">
              Top 15 features
            </h3>
            <p className="mb-3 text-data text-fg-subtle">
              Out of {summary.features.length}, each with what it measures.
            </p>
            <ImportanceBars items={featureItems} label="The 15 most important features, points" axisCaption="Mean |SHAP|, points" />
          </section>
        </Card>

        <aside className="space-y-4">
          {summary.top_players.length > 0 && (
            <Card>
              <SectionHeading as="h2" size="md" title="Explain a player" subtitle="The highest predicted scorers this gameweek" />
              <PlayerChips players={summary.top_players} onSelect={onPickPlayer} />
            </Card>
          )}
          <ModelFacts summary={summary} />
          <HowToRead baseValue={summary.base_value} />
        </aside>
      </div>

      {summary.beeswarm.length > 0 && (
        <Card>
          <SectionHeading
            as="h2"
            title="How each feature pushes predictions"
            subtitle="One dot per player and fixture. Right of the line adds points, left takes them away; the colour is the feature’s value, so you can see whether high values help or hurt."
          />
          <Beeswarm rows={summary.beeswarm} descriptions={descriptions} />
        </Card>
      )}
    </div>
  );
}
