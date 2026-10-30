import { useId, type ReactNode } from 'react';
import type { ExplainDependence, ExplainFeature } from '../../services/api';
import { cn } from '../../utils/cn';
import { Card, Select, Skeleton } from '../ui';
import { DependencePlot } from './DependencePlot';
import { points } from './format';

// The Feature effects tab: pick a feature, see how its value moves every player's prediction.

interface FeatureEffectsProps {
  features: ExplainFeature[];
  feature: string;
  onFeatureChange: (feature: string) => void;
  dependence?: ExplainDependence;
  /** Shown in place of the plot (error or unavailable state). */
  problem?: ReactNode;
  /** An older feature's plot is on screen while the selected one loads. */
  updating?: boolean;
}

/** Features grouped for the picker, groups in order of their most important feature. */
function grouped(features: ExplainFeature[]): Array<[string, ExplainFeature[]]> {
  const groups = new Map<string, ExplainFeature[]>();
  for (const f of [...features].sort((a, b) => b.importance - a.importance)) {
    const list = groups.get(f.group) ?? [];
    list.push(f);
    groups.set(f.group, list);
  }
  return [...groups.entries()];
}

export function FeatureEffects({ features, feature, onFeatureChange, dependence, problem, updating = false }: FeatureEffectsProps) {
  const selectId = useId();
  const meta = features.find((f) => f.feature === feature);
  const shown = dependence && dependence.feature === feature ? dependence : updating ? dependence : undefined;

  return (
    <Card>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,22rem)_minmax(0,1fr)] md:items-end">
        <div>
          <label htmlFor={selectId} className="mb-1.5 block text-micro font-semibold uppercase text-fg-muted">
            Feature
          </label>
          <Select id={selectId} value={feature} onChange={(e) => onFeatureChange(e.target.value)}>
            {grouped(features).map(([group, list]) => (
              <optgroup key={group} label={group}>
                {list.map((f) => (
                  <option key={f.feature} value={f.feature}>
                    {f.label}
                  </option>
                ))}
              </optgroup>
            ))}
          </Select>
        </div>
        {meta && (
          <p className="text-data text-fg-muted">
            {meta.description}{' '}
            <span className="text-fg-subtle">
              {meta.group} · average impact {points(meta.importance)} pts
            </span>
          </p>
        )}
      </div>

      <div className="mt-5">
        {problem ? (
          problem
        ) : shown ? (
          <div className={cn('transition-opacity', updating && 'opacity-60')} aria-busy={updating}>
            <DependencePlot data={shown} />
          </div>
        ) : (
          <div role="status">
            <span className="sr-only">Loading the feature’s effect…</span>
            <Skeleton className="h-[340px] w-full rounded-md" />
          </div>
        )}
      </div>

      <p className="mt-4 border-t border-line pt-3 text-data text-fg-subtle">
        Each dot is one player’s fixture this gameweek: across is his value for the feature, up or down is how many points it adds to or
        takes from his prediction. Spread at the same value comes from the other features it interacts with.
      </p>
    </Card>
  );
}
