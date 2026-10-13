import { cn } from '../../utils/cn';

interface LogoProps {
  /** Icon height in px; the wordmark scales with it. */
  size?: number;
  /** Icon only (e.g. collapsed nav, favicon-like uses). */
  iconOnly?: boolean;
  className?: string;
}

/** Pitch Me brand mark: a minimal pitch outline plus the wordmark. Original artwork. */
export function Logo({ size = 28, iconOnly = false, className }: LogoProps) {
  return (
    <span className={cn('inline-flex items-center gap-2', className)}>
      <svg width={size * 1.4} height={size} viewBox="0 0 42 30" aria-hidden="true" className="shrink-0">
        <rect x="1.5" y="1.5" width="39" height="27" rx="4" fill="none" stroke="rgb(var(--accent))" strokeWidth="2.5" />
        <line x1="21" y1="1.5" x2="21" y2="28.5" stroke="rgb(var(--accent))" strokeWidth="2.5" />
        <circle cx="21" cy="15" r="5.5" fill="none" stroke="rgb(var(--accent))" strokeWidth="2.5" />
        <circle cx="21" cy="15" r="1.6" fill="rgb(var(--accent-2))" />
      </svg>
      {!iconOnly && (
        <span
          className="font-display font-extrabold leading-none tracking-tight text-fg"
          style={{ fontSize: size * 0.82 }}
        >
          Pitch<span className="text-accent">Me</span>
        </span>
      )}
      <span className="sr-only">Pitch Me</span>
    </span>
  );
}
