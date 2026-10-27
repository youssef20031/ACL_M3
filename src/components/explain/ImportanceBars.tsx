import { AxisRow, BarSpan, BarTrack, ChartRow, ChartRows, RowLabel, type BarTone } from './ChartRows';
import { linearScale, niceTicks, points } from './format';

export interface ImportanceItem {
  key: string;
  label: string;
  /** Quieter second line, e.g. the feature's group or description. */
  detail?: string;
  /** Hover text with the full description. */
  tooltip?: string;
  value: number;
  /** Bar colour; defaults to the chart's tone. */
  tone?: BarTone;
}

interface ImportanceBarsProps {
  items: ImportanceItem[];
  /** Accessible name of the chart. */
  label: string;
  tone?: BarTone;
  digits?: number;
  /** Shown under the axis, e.g. "Mean |SHAP|, points". */
  axisCaption?: string;
  /** Hide the axis (when the values are labelled and the scale doesn't matter). */
  axis?: boolean;
  /** Spoken unit after each value. */
  unit?: string;
  className?: string;
}

/** Horizontal magnitude bars from zero, each labelled with its value. */
export function ImportanceBars({ items, label, tone = 'accent', digits = 2, axisCaption, axis = true, unit = 'points', className }: ImportanceBarsProps) {
  const max = Math.max(0, ...items.map((i) => i.value));
  const ticks = niceTicks(0, max, 4);
  const top = Math.max(max, ticks[ticks.length - 1] ?? max) || 1;
  const toPercent = linearScale(0, top);
  return (
    <ChartRows label={label} className={className}>
      {items.map((item) => (
        <ChartRow
          key={item.key}
          label={<RowLabel title={item.label} detail={item.detail} tooltip={item.tooltip ?? item.label} />}
          value={
            <span className="text-data font-semibold text-fg">
              {points(item.value, digits)}
              <span className="sr-only"> {unit}</span>
            </span>
          }
          bar={
            <BarTrack filled>
              <BarSpan from={0} to={toPercent(item.value)} tone={item.tone ?? tone} rounded="end" />
            </BarTrack>
          }
        />
      ))}
      {axis && items.length > 0 && <AxisRow ticks={ticks} toPercent={toPercent} caption={axisCaption} />}
    </ChartRows>
  );
}
