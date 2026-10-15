import type { ReactNode } from 'react';
import { Container, PageHeader } from '../ui';

/** Readable single-column layout for text pages (privacy, terms, contact). */
export function ProsePage({
  eyebrow,
  title,
  description,
  updated,
  children,
}: {
  eyebrow: string;
  title: string;
  description: ReactNode;
  /** "Last updated" date shown under the header. */
  updated?: string;
  children: ReactNode;
}) {
  return (
    <Container className="max-w-3xl pb-10">
      <PageHeader eyebrow={eyebrow} title={title} description={description} />
      {updated && <p className="-mt-3 mb-6 text-micro text-fg-subtle">Last updated {updated}</p>}
      <div className="space-y-8">{children}</div>
    </Container>
  );
}

export function ProseSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="space-y-3 text-body text-fg-muted [&_a]:font-semibold [&_a]:text-accent hover:[&_a]:underline [&_strong]:text-fg [&_ul]:list-disc [&_ul]:space-y-1.5 [&_ul]:pl-5">
      <h2 className="font-display text-h3 font-bold text-fg">{title}</h2>
      {children}
    </section>
  );
}
