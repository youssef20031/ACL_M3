// Formatting and small pure helpers shared by the model-insights components.

/** Signed with a real minus sign and fixed decimals: "+1.23", "−0.40", "0.00". */
export function signed(value: number, digits = 2): string {
  const rounded = Number(value.toFixed(digits)); // -0.004 rounds to -0, shown unsigned
  if (rounded === 0) return (0).toFixed(digits);
  return `${rounded > 0 ? '+' : '−'}${Math.abs(rounded).toFixed(digits)}`;
}

/** Spoken form of a signed number, for screen-reader text. */
export function spokenSigned(value: number, digits = 2, unit = 'points'): string {
  const rounded = Number(value.toFixed(digits));
  if (rounded === 0) return `no change`;
  return `${rounded > 0 ? 'plus' : 'minus'} ${Math.abs(rounded).toFixed(digits)} ${unit}`;
}

/** Points with up to `digits` decimals; a real minus sign for negatives. */
export function points(value: number, digits = 2): string {
  const text = value.toFixed(digits);
  return value < 0 && Number(text) !== 0 ? `−${text.slice(1)}` : text.replace(/^-/, '');
}

/** A number with at most 2 decimals and no trailing zeros: 84.333 → "84.33", 2.5 → "2.5", 3 → "3". */
export function trimNumber(value: number): string {
  const text = Number(value.toFixed(2)).toString();
  return text.startsWith('-') ? `−${text.slice(1)}` : text;
}

/** Features that are 0/1 flags in the model, shown as Yes/No. */
function isFlag(feature: string): boolean {
  return (
    feature === 'was_home' ||
    feature === 'started_r1' ||
    feature.startsWith('pos_') ||
    feature === 'snap_unavailable' ||
    feature === 'snap_doubtful' ||
    feature === 'snap_news'
  );
}

/**
 * A feature's raw value for display: flags as Yes/No, price in £m, ownership rank as "top N% owned",
 * anything else as a number with at most 2 decimals. Null (missing) is an em dash.
 */
export function featureValue(feature: string, value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—';
  if (isFlag(feature) && (value === 0 || value === 1)) return value === 1 ? 'Yes' : 'No';
  if (feature === 'price') return `£${value.toFixed(1)}m`;
  if (feature === 'selected_rank' && value >= 0 && value <= 1) {
    // Percentile rank of ownership: 1 = the most selected player.
    return `top ${Math.max(1, Math.round((1 - value) * 100))}% owned`;
  }
  return trimNumber(value);
}

/**
 * Parses the backend's timestamps, which come as ISO ("2026-10-22T12:29:58.734103203+00:00") or
 * pandas-style ("2026-10-24 11:30:00+00:00"). Safari rejects the space and more than 3 fraction
 * digits, so both are normalised first.
 */
export function parseTimestamp(value: string | null | undefined): Date | null {
  if (!value) return null;
  const iso = value.trim().replace(/^(\d{4}-\d{2}-\d{2}) (?=\d)/, '$1T').replace(/(\.\d{3})\d+/, '$1');
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function formatDateTime(value: string | null | undefined): string | null {
  const d = parseTimestamp(value);
  return d ? d.toLocaleString(undefined, { weekday: 'short', day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) : null;
}

export function formatDate(value: string | null | undefined): string | null {
  const d = parseTimestamp(value);
  return d ? d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : null;
}

/** "2020-21 to 2026-27", or the one season, or null. */
export function seasonRange(seasons: string[]): string | null {
  if (!seasons.length) return null;
  const sorted = [...seasons].sort();
  return sorted.length === 1 ? sorted[0] : `${sorted[0]} to ${sorted[sorted.length - 1]}`;
}

/** [lo, hi] covering every value (and zero, unless told otherwise), padded by `pad` of the span. */
export function domainOf(values: number[], { pad = 0.04, includeZero = true } = {}): [number, number] {
  const finite = values.filter(Number.isFinite);
  let lo = Math.min(...finite, ...(includeZero ? [0] : []));
  let hi = Math.max(...finite, ...(includeZero ? [0] : []));
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (lo === hi) {
    lo -= 0.5;
    hi += 0.5;
  }
  const span = hi - lo;
  return [lo - (includeZero && lo === 0 ? 0 : span * pad), hi + (includeZero && hi === 0 ? 0 : span * pad)];
}

/** Maps a value in [lo, hi] to 0..`size` (percent by default). */
export function linearScale(lo: number, hi: number, size = 100): (v: number) => number {
  const span = hi - lo || 1;
  return (v: number) => ((v - lo) / span) * size;
}

/** "Nice" tick values covering [min, max], about `count` of them. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const span = max - min;
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const start = Math.ceil(min / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-9; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

/** Tick label: as many decimals as the step needs (0.5 → 1 dp, 0.25 → 2 dp). */
export function tickLabel(value: number, ticks: number[]): string {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  const digits = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  const text = value.toFixed(digits);
  return Number(text) === 0 ? (0).toFixed(digits) : text.replace(/^-/, '−');
}

/** Spearman-free direction of a beeswarm row: do high feature values push predictions up or down? */
export function valueDirection(points: Array<[number | null, number]>): 'up' | 'down' | 'mixed' {
  const known = points.filter((p): p is [number, number] => p[0] != null);
  if (known.length < 3) return 'mixed';
  const mx = known.reduce((s, p) => s + p[0], 0) / known.length;
  const my = known.reduce((s, p) => s + p[1], 0) / known.length;
  let cov = 0;
  let vx = 0;
  let vy = 0;
  for (const [x, y] of known) {
    cov += (x - mx) * (y - my);
    vx += (x - mx) ** 2;
    vy += (y - my) ** 2;
  }
  if (vx === 0 || vy === 0) return 'mixed';
  const r = cov / Math.sqrt(vx * vy);
  return r > 0.2 ? 'up' : r < -0.2 ? 'down' : 'mixed';
}

export function ordinal(n: number): string {
  const rem100 = n % 100;
  const suffix = rem100 >= 11 && rem100 <= 13 ? 'th' : ({ 1: 'st', 2: 'nd', 3: 'rd' } as Record<number, string>)[n % 10] ?? 'th';
  return `${n}${suffix}`;
}

/** How well LIME's local linear model fits, in words. */
export function fitQuality(r2: number): string {
  if (r2 >= 0.7) return 'a close fit';
  if (r2 >= 0.4) return 'a moderate fit';
  return 'a rough fit';
}
