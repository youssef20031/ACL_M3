import type { ReactNode } from 'react';
import { Bug, Code2, ExternalLink, Users } from 'lucide-react';
import { Card } from '../components/ui';
import { ProsePage, ProseSection } from '../components/legal/Prose';
import { REPO_URL } from '../components/shell/RepoLink';
import { usePageMeta } from '../hooks/usePageMeta';

function ContactCard({ icon, title, body, href, cta }: { icon: ReactNode; title: string; body: string; href: string; cta: string }) {
  return (
    <Card className="flex flex-col">
      <div className="mb-3 inline-flex h-10 w-10 items-center justify-center rounded-md bg-brand text-accent">{icon}</div>
      <h2 className="font-display text-h3 font-bold text-fg">{title}</h2>
      <p className="mt-1 flex-1 text-data text-fg-muted">{body}</p>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="mt-4 inline-flex items-center gap-1.5 text-data font-semibold text-accent hover:underline"
      >
        {cta}
        <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
        <span className="sr-only"> (opens GitHub in a new tab)</span>
      </a>
    </Card>
  );
}

export function Contact() {
  usePageMeta({
    title: 'Contact',
    description: 'Report a bug, suggest a feature or browse the code: Pitch Me is developed in the open on GitHub.',
  });

  return (
    <ProsePage
      eyebrow="Get in touch"
      title="Contact"
      description="Pitch Me is built in the open. The quickest way to reach the people behind it is GitHub."
    >
      <div className="grid gap-4 sm:grid-cols-3">
        <ContactCard
          icon={<Bug className="h-5 w-5" aria-hidden="true" />}
          title="Report a problem"
          body="Something wrong, a wrong answer or an idea? Open an issue and describe what happened."
          href={`${REPO_URL}/issues/new`}
          cta="Open an issue"
        />
        <ContactCard
          icon={<Code2 className="h-5 w-5" aria-hidden="true" />}
          title="Browse the code"
          body="The whole site, the knowledge graph and the prediction models are on GitHub."
          href={REPO_URL}
          cta="View the repository"
        />
        <ContactCard
          icon={<Users className="h-5 w-5" aria-hidden="true" />}
          title="The team"
          body="See who builds Pitch Me and what they've been working on."
          href={`${REPO_URL}/graphs/contributors`}
          cta="See contributors"
        />
      </div>

      <ProseSection title="Before you write">
        <ul>
          <li>Wrong numbers? Mention the player, season or gameweek so it can be checked against the data.</li>
          <li>Don't post personal information in an issue: issues are public.</li>
          <li>Pitch Me is a free fan project, so replies may take a while.</li>
        </ul>
      </ProseSection>
    </ProsePage>
  );
}
