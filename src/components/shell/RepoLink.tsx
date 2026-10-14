import { Github } from 'lucide-react';
import { cn } from '../../utils/cn';

export const REPO_URL = 'https://github.com/youssef20031/ACL_M3';

/** Link to the project's source on GitHub (opens in a new tab). */
interface RepoLinkProps {
  label?: string;
  /** Icon only on phones (the label shows from sm up). */
  compact?: boolean;
  className?: string;
}

export function RepoLink({ label = 'GitHub', compact = false, className }: RepoLinkProps) {
  return (
    <a
      href={REPO_URL}
      target="_blank"
      rel="noopener noreferrer"
      aria-label={`${label} (opens in a new tab)`}
      title={compact ? label : undefined}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-sm text-data font-semibold text-fg-muted transition-colors hover:text-accent',
        className,
      )}
    >
      <Github className="h-4 w-4" aria-hidden="true" />
      <span className={compact ? 'hidden sm:inline' : undefined}>{label}</span>
    </a>
  );
}
