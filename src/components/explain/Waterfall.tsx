import type { ExplainContribution, ExplainFixture } from '../../services/api';
import { cn } from '../../utils/cn';
import { AxisRow, BarMark, BarSpan, BarTrack, ChartRow, ChartRows, RowLabel, type BarTone } from './ChartRows';
import { DataDisclosure } from './DataDisclosure';
import { domainOf, featureValue, linearScale, niceTicks, points, signed, spokenSigned } from './format';

/*
 * SHAP waterfall for one fixture: from the base value (the model's average prediction), each of
 * the biggest contributions moves the running total right (adds points, mint) or left (takes
 * points away, pink); the rest are lumped together, and the last row is the model's output.
 * SHAP values add up exactly, so the remainder is worked out from the output itself.
 */

interface Step {
  key: string;
  label: string;
  detail?: string;
  tooltip?: string;
  from: number;
  to: number;
  shap: number;
  tone: BarTone;
}

export function SignedValue({ value, digits = 2, className }: { value: number; digits?: number; className?: string }) {
  const rounded = Number(value.toFixed(digits));
  return (
    <span className={cn('text-data font-semibold', rounded > 0 ? 'text-accent' : rounded < 0 ? 'text-danger' : 'text-fg-muted', className)}>
      <span aria-hidden="true">{signed(value, digits)}</span>
      <span className="sr-only">{spokenSigned(value, digits)}</span>
    </span>
  );
}

interface WaterfallProps {
  fixture: ExplainFixture;
  /** How many contributions get their own bar. */
  top?: number;
  /** Accessible name of the chart. */
  label: string;
  /** Feature descriptions for hover text. */
  descriptions?: Record<string, string>;
}

export function Waterfall({ fixture, top = 10, label, descriptions = {} }: WaterfallProps) {
  const sorted = [...fixture.contributions].sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  const shown = sorted.slice(0, top);
  const base = fixture.base_value;
  const output = fixture.model_points;
  const shownSum = shown.reduce((s, c) => s + c.shap, 0);
  const rest = output - base - shownSum;
  const restCount = sorted.length - shown.length;

  const steps: Step[] = [];
  let running = base;
  for (const c of shown) {
    steps.push({
      key: c.feature,
      label: c.label,
      detail: `Value: ${featureValue(c.feature, c.value)}`,
      tooltip: descriptions[c.feature] ?? c.label,
      from: running,
      to: running + c.shap,
      shap: c.shap,
      tone: c.shap >= 0 ? 'positive' : 'negative',
    });
    running += c.shap;
  }
  if (restCount > 0 || Math.abs(rest) >= 0.005) {
    steps.push({
      key: '__rest',
      label: restCount > 0 ? `${restCount} other ${restCount === 1 ? 'feature' : 'features'}` : 'Other features',
      detail: 'Combined',
      from: running,
      to: running + rest,
      shap: rest,
      tone: rest >= 0 ? 'positive-muted' : 'negative-muted',
    });
  }

  const [lo, hi] = domainOf([base, output, ...steps.flatMap((s) => [s.from, s.to])]);
  const ticks = niceTicks(lo, hi, 4);
  const toPercent = linearScale(lo, hi);
  const zero = toPercent(0);

  return (
    <figure className="min-w-0">
      <ChartRows label={label}>
        <ChartRow
          label={<RowLabel title="Base value" detail="The model’s average prediction" />}
          value={<span className="text-data font-semibold text-fg-muted">{points(base)}</span>}
          bar={
            <BarTrack zero={zero}>
              <BarSpan from={zero} to={toPercent(base)} tone="base" rounded="end" />
              <BarMark at={toPercent(base)} />
            </BarTrack>
          }
        />
        {steps.map((s) => (
          <ChartRow
            key={s.key}
            label={<RowLabel title={s.label} detail={s.detail} tooltip={s.tooltip} />}
            value={<SignedValue value={s.shap} />}
            bar={
              <BarTrack zero={zero}>
                <BarSpan from={toPercent(s.from)} to={toPercent(s.to)} tone={s.tone} />
              </BarTrack>
            }
          />
        ))}
        <ChartRow
          className="border-t border-line pt-2.5"
          label={<RowLabel title="Model output" detail="Points for this fixture, before adjustments" />}
          value={
            <span className="text-data font-bold text-fg">
              {points(output)}
              <span className="sr-only"> points</span>
            </span>
          }
          bar={
            <BarTrack zero={zero}>
              <BarSpan from={zero} to={toPercent(output)} tone="total" rounded="end" />
              <BarMark at={toPercent(output)} />
            </BarTrack>
          }
        />
        <AxisRow ticks={ticks} toPercent={toPercent} caption="Points" />
      </ChartRows>

      <DataDisclosure summary={`All ${sorted.length} contributions`}>
        <ContributionsTable contributions={sorted} base={base} output={output} />
      </DataDisclosure>
    </figure>
  );
}

function ContributionsTable({ contributions, base, output }: { contributions: ExplainContribution[]; base: number; output: number }) {
  return (
    <table className="w-full text-left text-data">
      <caption className="sr-only">Every feature’s SHAP contribution, largest first, from the base value to the model output</caption>
      <thead className="text-micro uppercase text-fg-subtle">
        <tr>
          <th scope="col" className="py-1.5 pr-3 font-semibold">Feature</th>
          <th scope="col" className="py-1.5 pr-3 font-semibold">Group</th>
          <th scope="col" className="py-1.5 pr-3 text-right font-semibold">Value</th>
          <th scope="col" className="py-1.5 text-right font-semibold">Points</th>
        </tr>
      </thead>
      <tbody>
        <tr className="border-t border-line">
          <th scope="row" className="py-1.5 pr-3 font-semibold text-fg">Base value</th>
          <td className="py-1.5 pr-3 text-fg-subtle">–</td>
          <td className="py-1.5 pr-3 text-right text-fg-subtle">–</td>
          <td className="py-1.5 text-right tabular-nums text-fg">{points(base)}</td>
        </tr>
        {contributions.map((c) => (
          <tr key={c.feature} className="border-t border-line">
            <th scope="row" className="py-1.5 pr-3 font-normal text-fg">{c.label}</th>
            <td className="py-1.5 pr-3 text-fg-subtle">{c.group}</td>
            <td className="whitespace-nowrap py-1.5 pr-3 text-right tabular-nums text-fg-muted">{featureValue(c.feature, c.value)}</td>
            <td className="py-1.5 text-right tabular-nums">
              <SignedValue value={c.shap} digits={3} className="font-normal" />
            </td>
          </tr>
        ))}
        <tr className="border-t-2 border-line-strong">
          <th scope="row" className="py-1.5 pr-3 font-semibold text-fg">Model output</th>
          <td className="py-1.5 pr-3 text-fg-subtle">–</td>
          <td className="py-1.5 pr-3 text-right text-fg-subtle">–</td>
          <td className="py-1.5 text-right font-bold tabular-nums text-fg">{points(output)}</td>
        </tr>
      </tbody>
    </table>
  );
}
