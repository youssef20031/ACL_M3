import { Github } from 'lucide-react';
import { cn } from '../../utils/cn';

export const REPO_URL = 'https://github.com/youssef20031/ACL_M3';

/** Link to the project's source on GitHub (opens in a new tab). */
export function RepoLink({ label = 'GitHub', className }: { label?: string; className?: string }) {
  return (
    <a
      href={REPO_URL}
      target="_blank"
      rel="noopener noreferrer"
      className={cn(
        'inline-flex items-center gap-1.5 rounded-sm text-data font-semibold text-fg-muted transition-colors hover:text-accent',
        className,
      )}
    >
      <Github className="h-4 w-4" aria-hidden="true" />
      {label}
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  );
}
