import { Hourglass, SearchX } from 'lucide-react';
import { handleApiError, isNotFoundError, isServiceUnavailableError } from '../../services/api';
import { Card, EmptyState, ErrorState, Skeleton } from '../ui';

/** No explanations yet (503): they're written by the daily prediction run. */
export function ExplainUnavailable({ bare = false }: { bare?: boolean }) {
  const state = (
    <EmptyState
      icon={<Hourglass className="h-8 w-8" aria-hidden="true" />}
      title="Explanations appear after the next prediction run"
      description="The model’s next-gameweek predictions are explained once a day, right after they’re made. Check back soon."
    />
  );
  return bare ? state : <Card padded={false}>{state}</Card>;
}

interface ExplainErrorProps {
  error: unknown;
  onRetry?: () => void;
  /** What failed to load, e.g. "the model insights". */
  what: string;
  /** Text for a 404, when it means something specific (a player not in this gameweek). */
  notFound?: { title: string; description: string };
  /** Without the card around it, for use inside another card. */
  bare?: boolean;
}

/** 503 → the friendly "not yet" state; a 404 with a meaning → its own message; anything else → an error with retry. */
export function ExplainError({ error, onRetry, what, notFound, bare = false }: ExplainErrorProps) {
  if (isServiceUnavailableError(error)) return <ExplainUnavailable bare={bare} />;
  const state =
    notFound && isNotFoundError(error) ? (
      <EmptyState icon={<SearchX className="h-8 w-8" aria-hidden="true" />} title={notFound.title} description={notFound.description} />
    ) : (
      <ErrorState title={`Couldn’t load ${what}`} description={handleApiError(error)} onRetry={onRetry} />
    );
  return bare ? state : <Card padded={false}>{state}</Card>;
}

/** Loading placeholder for a stack of chart cards. */
export function ExplainSkeleton({ label = 'Loading…', blocks = 2 }: { label?: string; blocks?: number }) {
  return (
    <div role="status" className="space-y-4">
      <span className="sr-only">{label}</span>
      {Array.from({ length: blocks }, (_, i) => (
        <Skeleton key={i} className={i === 0 ? 'h-32 w-full rounded-lg' : 'h-80 w-full rounded-lg'} />
      ))}
    </div>
  );
}
