import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

export interface StatBarDatum {
  name: string;
  value: number;
}

interface StatBarChartProps {
  data: StatBarDatum[];
  /** Accessible summary of the chart. The values are also listed for screen readers. */
  label: string;
  height?: number;
  /** Bar colour token, e.g. 'accent' or 'accent-2'. */
  tone?: 'accent' | 'accent-2';
  /** Decimal places in the tooltip and screen-reader list. */
  decimals?: number;
}

const tick = { fontSize: 12, fill: 'rgb(var(--text-subtle))' };

/** Simple bar chart coloured from the design tokens (follows the theme). */
export function StatBarChart({ data, label, height = 220, tone = 'accent', decimals = 0 }: StatBarChartProps) {
  const fmt = (v: number) => (decimals ? v.toFixed(decimals) : v.toLocaleString());
  return (
    <figure>
      <div role="img" aria-label={label}>
        <ResponsiveContainer width="100%" height={height}>
          <BarChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgb(var(--border))" vertical={false} />
            <XAxis dataKey="name" tick={tick} axisLine={{ stroke: 'rgb(var(--border))' }} tickLine={false} />
            <YAxis tick={tick} axisLine={false} tickLine={false} />
            <Tooltip
              formatter={(v: number) => fmt(v)}
              contentStyle={{
                fontSize: 12,
                borderRadius: 6,
                background: 'rgb(var(--surface-2))',
                borderColor: 'rgb(var(--border-strong))',
                color: 'rgb(var(--text))',
              }}
              labelStyle={{ color: 'rgb(var(--text-muted))' }}
              itemStyle={{ color: 'rgb(var(--text))' }}
              cursor={{ fill: 'rgb(var(--brand-2) / 0.45)' }}
            />
            <Bar dataKey="value" name="Value" fill={`rgb(var(--${tone}))`} radius={[4, 4, 0, 0]} maxBarSize={56} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="sr-only">
        {data.map((d) => `${d.name}: ${fmt(d.value)}`).join(', ')}
      </figcaption>
    </figure>
  );
}
