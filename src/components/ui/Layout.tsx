import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from '../../utils/cn';

/** Centred page-width container with responsive gutters. */
export function Container({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('mx-auto w-full max-w-page px-4 sm:px-6', className)} {...rest} />;
}

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Featured panels use the brand purple instead of the neutral surface. */
  tone?: 'default' | 'brand';
  padded?: boolean;
}

export function Card({ tone = 'default', padded = true, className, ...rest }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-lg border shadow-card',
        tone === 'brand' ? 'border-brand-2 bg-brand' : 'border-line bg-surface',
        padded && 'p-4 sm:p-5',
        className,
      )}
      {...rest}
    />
  );
}

interface SectionHeadingProps {
  title: ReactNode;
  subtitle?: ReactNode;
  /** Right-aligned action, e.g. a "See all" link. */
  action?: ReactNode;
  as?: 'h2' | 'h3';
  className?: string;
  id?: string;
}

export function SectionHeading({ title, subtitle, action, as: Tag = 'h2', className, id }: SectionHeadingProps) {
  return (
    <div className={cn('mb-3 flex items-end justify-between gap-3', className)}>
      <div className="min-w-0">
        <Tag id={id} className={cn('font-display font-bold text-fg', Tag === 'h2' ? 'text-h2' : 'text-h3')}>
          {title}
        </Tag>
        {subtitle && <p className="mt-0.5 text-data text-fg-subtle">{subtitle}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

interface PageHeaderProps {
  /** Small label above the title, e.g. "Gameweek 7". */
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({ eyebrow, title, description, actions, className }: PageHeaderProps) {
  return (
    <header className={cn('flex flex-col gap-4 py-6 sm:flex-row sm:items-end sm:justify-between sm:py-8', className)}>
      <div className="min-w-0">
        {eyebrow && <p className="mb-1 text-micro font-semibold uppercase text-accent">{eyebrow}</p>}
        <h1 className="font-display text-h1 font-extrabold text-fg sm:text-display">{title}</h1>
        {description && <p className="mt-2 max-w-2xl text-body text-fg-muted">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
    </header>
  );
}
