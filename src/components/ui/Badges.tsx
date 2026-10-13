import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from '../../utils/cn';

type ChipTone = 'neutral' | 'accent' | 'info' | 'warning' | 'danger' | 'brand';

const CHIP_TONES: Record<ChipTone, string> = {
  neutral: 'bg-surface-2 text-fg-muted border-line',
  accent: 'bg-accent/15 text-accent border-accent/30',
  info: 'bg-info/15 text-info border-info/30',
  warning: 'bg-warning/15 text-warning border-warning/30',
  danger: 'bg-danger/15 text-danger border-danger/30',
  brand: 'bg-brand text-fg border-brand-2',
};

interface ChipProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: ChipTone;
  icon?: ReactNode;
}

export function Chip({ tone = 'neutral', icon, className, children, ...rest }: ChipProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 whitespace-nowrap rounded-sm border px-1.5 py-0.5 text-micro font-semibold',
        CHIP_TONES[tone],
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </span>
  );
}

// --- Positions ---------------------------------------------------------------

export type Position = 'GK' | 'DEF' | 'MID' | 'FWD';

const POSITION_CLASSES: Record<Position, string> = {
  GK: 'bg-pos-gk',
  DEF: 'bg-pos-def',
  MID: 'bg-pos-mid',
  FWD: 'bg-pos-fwd',
};

const POSITION_NAMES: Record<Position, string> = {
  GK: 'Goalkeeper',
  DEF: 'Defender',
  MID: 'Midfielder',
  FWD: 'Forward',
};

/** Normalises the spellings the backend uses (GKP, GK, Goalkeeper, …). */
export function normalizePosition(value: string | null | undefined): Position | null {
  const v = (value ?? '').trim().toUpperCase();
  if (v === 'GK' || v === 'GKP' || v.startsWith('GOAL')) return 'GK';
  if (v === 'DEF' || v.startsWith('DEFEN')) return 'DEF';
  if (v === 'MID' || v.startsWith('MIDF')) return 'MID';
  if (v === 'FWD' || v === 'FW' || v.startsWith('FORW') || v.startsWith('STRIK')) return 'FWD';
  return null;
}

export function PositionBadge({ position, className }: { position: string | null | undefined; className?: string }) {
  const pos = normalizePosition(position);
  if (!pos) return null;
  return (
    <span
      title={POSITION_NAMES[pos]}
      className={cn(
        'inline-flex w-9 items-center justify-center rounded-sm py-0.5 text-micro font-bold text-accent-on',
        POSITION_CLASSES[pos],
        className,
      )}
    >
      {pos}
    </span>
  );
}

// --- Availability ------------------------------------------------------------

/** FPL availability codes: a available, d doubtful, i injured, s suspended, u unavailable, n not in squad. */
export type AvailabilityStatus = 'a' | 'd' | 'i' | 's' | 'u' | 'n';

const STATUS_META: Record<AvailabilityStatus, { label: string; tone: ChipTone }> = {
  a: { label: 'Available', tone: 'accent' },
  d: { label: 'Doubtful', tone: 'warning' },
  i: { label: 'Injured', tone: 'danger' },
  s: { label: 'Suspended', tone: 'danger' },
  u: { label: 'Unavailable', tone: 'neutral' },
  n: { label: 'Not in squad', tone: 'neutral' },
};

interface StatusBadgeProps {
  status: string | null | undefined;
  /** chance_of_playing_next_round, 0–100. */
  chance?: number | null;
  news?: string | null;
  /** Available players render nothing unless this is set. */
  showAvailable?: boolean;
  className?: string;
}

/** Availability chip. The status is always spelled out, never colour alone. */
export function StatusBadge({ status, chance, news, showAvailable = false, className }: StatusBadgeProps) {
  const code = (status ?? '').toLowerCase() as AvailabilityStatus;
  const meta = STATUS_META[code];
  if (!meta || (code === 'a' && !showAvailable)) return null;
  const label = code === 'd' && chance != null ? `${chance}%` : meta.label;
  return (
    <Chip tone={meta.tone} title={news || meta.label} className={className}>
      {label}
      {news && <span className="sr-only">: {news}</span>}
    </Chip>
  );
}

// --- Fixture difficulty ------------------------------------------------------

const FDR_CLASSES: Record<number, string> = {
  1: 'bg-fdr-1 text-white',
  2: 'bg-fdr-2 text-accent-on',
  3: 'bg-fdr-3 text-accent-on',
  4: 'bg-fdr-4 text-white',
  5: 'bg-fdr-5 text-white',
};

interface FdrBadgeProps {
  difficulty: number;
  /** Text inside the chip, e.g. "ARS (H)". Defaults to the number. */
  children?: ReactNode;
  className?: string;
}

export function FdrBadge({ difficulty, children, className }: FdrBadgeProps) {
  const d = Math.min(5, Math.max(1, Math.round(difficulty)));
  return (
    <span
      title={`Difficulty ${d} of 5`}
      className={cn(
        'inline-flex min-w-[2rem] items-center justify-center rounded-sm px-1.5 py-0.5 text-micro font-bold',
        FDR_CLASSES[d],
        className,
      )}
    >
      {children ?? d}
    </span>
  );
}
