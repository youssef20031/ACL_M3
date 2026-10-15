import { Link } from 'react-router-dom';
import { BarChart3 } from 'lucide-react';
import { Button } from '../ui';
import { setConsent, useConsent } from '../../lib/consent';

/**
 * Asks once whether the cookie-free page-view counter may run. Nothing is counted until the
 * visitor accepts; the choice can be changed later from the footer or the privacy page.
 */
export function ConsentBanner() {
  const consent = useConsent();
  if (consent !== null) return null;

  return (
    <section
      aria-label="Privacy choices"
      className="fixed inset-x-0 bottom-0 z-50 border-t border-line bg-surface-2/95 shadow-pop backdrop-blur-sm"
    >
      <div className="mx-auto flex w-full max-w-page flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:gap-6 sm:px-6">
        <BarChart3 className="hidden h-6 w-6 shrink-0 text-accent sm:block" aria-hidden="true" />
        <p className="flex-1 text-data text-fg-muted">
          <strong className="font-semibold text-fg">Help improve Pitch Me?</strong> With your OK we count page views
          anonymously: no cookies, no tracking across sites, nothing that identifies you. Your settings are kept in
          your browser either way.{' '}
          <Link to="/privacy" className="font-semibold text-accent hover:underline">
            Privacy policy
          </Link>
        </p>
        <div className="flex shrink-0 gap-2">
          <Button variant="secondary" size="sm" onClick={() => setConsent('declined')}>
            Decline
          </Button>
          {/* Equal weight for both choices: no nudging towards Accept */}
          <Button variant="secondary" size="sm" onClick={() => setConsent('accepted')}>
            Accept
          </Button>
        </div>
      </div>
    </section>
  );
}
