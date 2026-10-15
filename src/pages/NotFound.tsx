import { Link } from 'react-router-dom';
import { ArrowRight, Search } from 'lucide-react';
import { ButtonLink, Container } from '../components/ui';
import { usePageMeta } from '../hooks/usePageMeta';

const POPULAR = [
  { to: '/fixtures', label: 'Fixtures & results', hint: 'Every match by gameweek' },
  { to: '/table', label: 'League table', hint: 'Standings and form' },
  { to: '/team-of-the-week', label: 'Team of the Week', hint: 'The model’s best XI' },
  { to: '/predictions', label: 'Predictions', hint: 'Expected points next gameweek' },
  { to: '/players', label: 'Players', hint: 'Season leaders and profiles' },
  { to: '/assistant', label: 'Assistant', hint: 'Ask anything about FPL' },
];

/** Opens the header's player search (the app shell listens for "/"). */
function openPlayerSearch() {
  document.dispatchEvent(new KeyboardEvent('keydown', { key: '/' }));
}

export function NotFound() {
  usePageMeta({ title: "Page not found", description: "That page doesn't exist on Pitch Me. Head back to fixtures, the table, predictions or the assistant." });

  return (
    <Container className="max-w-3xl py-12 sm:py-16">
      <p className="font-display text-display font-extrabold leading-none text-accent sm:text-[4.5rem]">404</p>
      <h1 className="mt-3 font-display text-h1 font-extrabold text-fg">This page is offside</h1>
      <p className="mt-2 max-w-xl text-body text-fg-muted">
        The page you were looking for doesn&rsquo;t exist or has moved. Try one of these, or search for a player.
      </p>
      <div className="mt-6 flex flex-wrap gap-2">
        <ButtonLink to="/">Back to home</ButtonLink>
        <button
          type="button"
          onClick={openPlayerSearch}
          className="inline-flex h-10 items-center gap-2 rounded-md border border-line-strong bg-surface-2 px-4 text-sm text-fg transition-colors hover:bg-brand-2/60"
        >
          <Search className="h-4 w-4" aria-hidden="true" />
          Find a player
        </button>
      </div>

      <nav aria-label="Popular pages" className="mt-10">
        <h2 className="text-micro font-semibold uppercase text-fg-subtle">Popular pages</h2>
        <ul className="mt-3 grid gap-2 sm:grid-cols-2">
          {POPULAR.map((page) => (
            <li key={page.to}>
              <Link
                to={page.to}
                className="group flex items-center justify-between gap-3 rounded-lg border border-line bg-surface px-4 py-3 transition-colors hover:border-accent/60"
              >
                <span>
                  <span className="block font-semibold text-fg">{page.label}</span>
                  <span className="text-data text-fg-subtle">{page.hint}</span>
                </span>
                <ArrowRight className="h-4 w-4 shrink-0 text-fg-subtle transition-colors group-hover:text-accent" aria-hidden="true" />
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </Container>
  );
}
