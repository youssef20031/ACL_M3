import { Container, Skeleton } from '../ui';

/** Placeholder shown while a page's code chunk downloads: the shape of a typical page. */
export function PageSkeleton() {
  return (
    <Container role="status" aria-label="Loading page" className="py-8">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="mt-3 h-9 w-64 max-w-full" />
      <Skeleton className="mt-3 h-4 w-[32rem] max-w-full" />
      <div className="mt-8 grid gap-4 md:grid-cols-3">
        <Skeleton className="h-28 rounded-lg" />
        <Skeleton className="h-28 rounded-lg" />
        <Skeleton className="h-28 rounded-lg" />
      </div>
      <Skeleton className="mt-4 h-72 rounded-lg" />
      <span className="sr-only">Loading…</span>
    </Container>
  );
}
