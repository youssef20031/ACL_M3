import type { LimeExplanation } from '../../services/api';
import { cn } from '../../utils/cn';
import { AxisRow, BarSpan, BarTrack, ChartRow, ChartRows } from './ChartRows';
import { DataDisclosure } from './DataDisclosure';
import { SignedValue } from './Waterfall';
import { domainOf, featureValue, fitQuality, linearScale, niceTicks, points } from './format';

/*
 * LIME for one fixture: the weights of a simple linear model fitted to the points model's
 * outputs on perturbed copies of this player. Bars start at zero; each is the points the rule
 * adds (mint) or takes away (pink) in that local model.
 */

interface LimePanelProps {
  lime: LimeExplanation;
  /** How many rules get their own bar. */
  top?: number;
  /** Accessible name of the chart. */
  label: string;
}

/**
 * A rule as text. The backend writes conditions with the feature's label ("Minutes in last
 * match > 76"); if one arrives with the raw column name instead ("minutes_r1 > 76"), the label
 * leads and the raw rule follows in monospace.
 */
function RuleText({ label, condition, className }: { label: string; condition: string; className?: string }) {
  if (condition.includes(label)) return <span className={cn('block truncate', className)}>{condition}</span>;
  return (
    <>
      <span className={cn('block truncate', className)}>{label}</span>
      <code className="block truncate font-mono text-micro text-fg-subtle">{condition}</code>
    </>
  );
}

function Stat({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div className="min-w-0 bg-surface px-3 py-2">
      <dt className="text-micro font-semibold uppercase text-fg-subtle">{label}</dt>
      <dd className="mt-0.5 text-data font-semibold tabular-nums text-fg">
        {value}
        {detail && <span className="block text-micro font-normal text-fg-subtle">{detail}</span>}
      </dd>
    </div>
  );
}

export function LimePanel({ lime, top = 10, label }: LimePanelProps) {
  const sorted = [...lime.weights].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const shown = sorted.slice(0, top);
  const [lo, hi] = domainOf(shown.map((w) => w.weight));
  const ticks = niceTicks(lo, hi, 4);
  const toPercent = linearScale(lo, hi);
  const zero = toPercent(0);
  const r2 = Math.max(0, lime.score);

  return (
    <figure className="min-w-0">
      {shown.length > 0 ? (
        <ChartRows label={label}>
          {shown.map((w) => (
            <ChartRow
              key={`${w.feature}-${w.condition}`}
              label={
                <span className="block min-w-0" title={w.condition.includes(w.label) ? w.condition : `${w.label}: ${w.condition}`}>
                  <RuleText label={w.label} condition={w.condition} className="text-data font-semibold text-fg" />
                  <span className="block truncate text-micro text-fg-subtle">His value: {featureValue(w.feature, w.value)}</span>
                </span>
              }
              value={<SignedValue value={w.weight} />}
              bar={
                <BarTrack zero={zero}>
                  <BarSpan from={zero} to={toPercent(w.weight)} tone={w.weight >= 0 ? 'positive' : 'negative'} rounded="end" />
                </BarTrack>
              }
            />
          ))}
          <AxisRow ticks={ticks} toPercent={toPercent} caption="Weight in the local model, points" />
        </ChartRows>
      ) : (
        <p className="text-data text-fg-subtle">LIME found no rule that matters for this fixture.</p>
      )}

      <dl className="mt-4 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line">
        <Stat label="Local model" value={points(lime.local_prediction)} detail="LIME’s prediction" />
        <Stat label="Points model" value={points(lime.model_prediction)} detail="What it approximates" />
        <Stat label="Intercept" value={points(lime.intercept)} detail="Local model’s starting point" />
        <Stat label="Local fit R²" value={r2.toFixed(2)} detail={fitQuality(r2)} />
      </dl>

      <DataDisclosure summary={`All ${sorted.length} rules`}>
        <table className="w-full text-left text-data">
          <caption className="sr-only">Every LIME rule and its weight in points, largest first</caption>
          <thead className="text-micro uppercase text-fg-subtle">
            <tr>
              <th scope="col" className="py-1.5 pr-3 font-semibold">Rule</th>
              <th scope="col" className="py-1.5 pr-3 text-right font-semibold">His value</th>
              <th scope="col" className="py-1.5 text-right font-semibold">Weight</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((w) => (
              <tr key={`${w.feature}-${w.condition}`} className="border-t border-line">
                <th scope="row" className="max-w-[16rem] py-1.5 pr-3 font-normal text-fg">
                  <RuleText label={w.label} condition={w.condition} />
                </th>
                <td className="whitespace-nowrap py-1.5 pr-3 text-right tabular-nums text-fg-muted">{featureValue(w.feature, w.value)}</td>
                <td className="py-1.5 text-right tabular-nums">
                  <SignedValue value={w.weight} digits={3} className="font-normal" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </DataDisclosure>
    </figure>
  );
}
