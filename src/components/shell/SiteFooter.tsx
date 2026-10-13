import { Link } from 'react-router-dom';
import { Logo } from '../ui';
import { MORE_NAV, PRIMARY_NAV } from './SiteHeader';

export function SiteFooter() {
  return (
    <footer className="mt-12 border-t border-line bg-surface">
      <div className="mx-auto grid w-full max-w-page gap-8 px-4 py-8 sm:px-6 md:grid-cols-[1.4fr_1fr_1fr]">
        <div>
          <Logo size={24} />
          <p className="mt-3 max-w-sm text-data text-fg-subtle">
            Fantasy Premier League stats, fixtures and machine-learning predictions, with an assistant that answers
            questions from a knowledge graph of every season since 2020-21.
          </p>
        </div>
        <nav aria-label="Footer: main">
          <ul className="space-y-1.5 text-data">
            {PRIMARY_NAV.map((item) => (
              <li key={item.to}>
                <Link to={item.to} className="text-fg-muted hover:text-fg">
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        <nav aria-label="Footer: more">
          <ul className="space-y-1.5 text-data">
            {MORE_NAV.map((item) => (
              <li key={item.to}>
                <Link to={item.to} className="text-fg-muted hover:text-fg">
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
      <div className="border-t border-line">
        <p className="mx-auto w-full max-w-page px-4 py-4 text-micro text-fg-subtle sm:px-6">
          Pitch Me is a fan-made project. Not affiliated with, endorsed by or connected to the Premier League or
          Fantasy Premier League. Data from the public FPL API.
        </p>
      </div>
    </footer>
  );
}
