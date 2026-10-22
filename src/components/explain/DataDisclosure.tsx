import { useState, type ReactNode } from 'react';
import { ChevronRight } from 'lucide-react';
import { cn } from '../../utils/cn';

interface DataDisclosureProps {
  summary: ReactNode;
  /** Rendered only while open, so long tables don't weigh on the page. */
  children: ReactNode;
  className?: string;
}

/** A chart's numbers as a table, behind a native disclosure (keyboard and screen-reader friendly). */
export function DataDisclosure({ summary, children, className }: DataDisclosureProps) {
  const [open, setOpen] = useState(false);
  return (
    <details className={cn('group mt-3', className)} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 rounded-sm text-data font-semibold text-fg-muted hover:text-fg [&::-webkit-details-marker]:hidden">
        <ChevronRight className="h-4 w-4 transition-transform group-open:rotate-90" aria-hidden="true" />
        {summary}
      </summary>
      {open && <div className="mt-2 overflow-x-auto">{children}</div>}
    </details>
  );
}
