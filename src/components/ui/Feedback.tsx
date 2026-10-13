import type { ReactNode } from 'react';
import { AlertTriangle, Inbox } from 'lucide-react';
import { cn } from '../../utils/cn';
import { Button } from './Button';

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden="true" className={cn('animate-pulse rounded-sm bg-surface-2', className)} />;
}

/** Placeholder block for a loading list/table. Announced once to screen readers. */
export function SkeletonRows({ rows = 5, className }: { rows?: number; className?: string }) {
  return (
    <div role="status" className={cn('space-y-2', className)}>
      <span className="sr-only">Loading…</span>
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-9 w-full" />
      ))}
    </div>
  );
}

interface StateProps {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
  className?: string;
}

export function EmptyState({ title, description, action, icon, className }: StateProps) {
  return (
    <div className={cn('flex flex-col items-center justify-center px-4 py-10 text-center', className)}>
      <div className="mb-3 text-fg-subtle">{icon ?? <Inbox className="h-8 w-8" aria-hidden="true" />}</div>
      <p className="font-display text-h3 font-semibold text-fg">{title}</p>
      {description && <p className="mt-1 max-w-md text-data text-fg-subtle">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

interface ErrorStateProps extends Omit<StateProps, 'action' | 'title'> {
  title?: ReactNode;
  onRetry?: () => void;
}

export function ErrorState({ title = 'Something went wrong', description, onRetry, icon, className }: ErrorStateProps) {
  return (
    <div role="alert" className={cn('flex flex-col items-center justify-center px-4 py-10 text-center', className)}>
      <div className="mb-3 text-danger">{icon ?? <AlertTriangle className="h-8 w-8" aria-hidden="true" />}</div>
      <p className="font-display text-h3 font-semibold text-fg">{title}</p>
      {description && <p className="mt-1 max-w-md text-data text-fg-subtle">{description}</p>}
      {onRetry && (
        <Button variant="secondary" size="sm" className="mt-4" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}
