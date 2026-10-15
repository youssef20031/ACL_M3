import { Link } from 'react-router-dom';
import { Button } from '../components/ui';
import { ProsePage, ProseSection } from '../components/legal/Prose';
import { usePageMeta } from '../hooks/usePageMeta';
import { setConsent, useConsent } from '../lib/consent';

const ISSUES_URL = 'https://github.com/youssef20031/ACL_M3/issues';

function ConsentControls() {
  const consent = useConsent();
  const label =
    consent === 'accepted'
      ? 'You allowed anonymous page-view counting.'
      : consent === 'declined'
        ? 'You declined page-view counting, so nothing is counted.'
        : "You haven't chosen yet, so nothing is counted.";
  return (
    <div className="flex flex-col gap-3 rounded-lg border border-line bg-surface p-4 sm:flex-row sm:items-center sm:justify-between">
      <p className="text-data text-fg" role="status">{label}</p>
      <div className="flex shrink-0 gap-2">
        <Button variant="secondary" size="sm" onClick={() => setConsent('declined')} disabled={consent === 'declined'}>
          Decline
        </Button>
        <Button variant="secondary" size="sm" onClick={() => setConsent('accepted')} disabled={consent === 'accepted'}>
          Allow
        </Button>
      </div>
    </div>
  );
}

export function Privacy() {
  usePageMeta({
    title: 'Privacy policy',
    description: 'What Pitch Me stores (very little), what is sent to third parties, and how to change your choices.',
  });

  return (
    <ProsePage
      eyebrow="Legal"
      title="Privacy policy"
      description="Pitch Me is a fan-made Fantasy Premier League project. It has no accounts and sets no cookies. Here is everything it does with data."
      updated="15 October 2026"
    >
      <ProseSection title="Your choices">
        <p>
          Anonymous page-view counting only runs if you allow it. You can change your answer here at any time, or from{' '}
          <strong>Privacy choices</strong> in the footer. If your browser sends a Global Privacy Control signal, we treat
          it as a no.
        </p>
        <ConsentControls />
      </ProseSection>

      <ProseSection title="What we store">
        <ul>
          <li>
            <strong>In your browser (local storage), not on our servers:</strong> your assistant settings (language
            model, retrieval method), your trivia score and your privacy choice. Clearing site data removes them.
          </li>
          <li>
            <strong>Page views, only if you allow them:</strong> a count of views per page per day and the domain of the
            website that sent you (for example google.com). No IP address, device details, cookies or identifiers are
            stored, so these counts can't be linked to you. They're kept for up to 400 days.
          </li>
          <li>
            <strong>Technical logs:</strong> like most websites, our server's access logs record your IP address, the
            address requested and the time. We use them for security and troubleshooting only. They're capped at a
            few megabytes, so older entries are deleted automatically. IP addresses are also used briefly in memory
            to rate-limit the assistant.
          </li>
        </ul>
      </ProseSection>

      <ProseSection title="What is sent to other services">
        <ul>
          <li>
            <strong>Assistant questions</strong> (and the last few messages of the conversation, for context) are sent
            to <a href="https://groq.com/privacy-policy/" target="_blank" rel="noopener noreferrer">Groq</a>, which runs
            the language model that writes the answer. Please don't include personal information in questions. We
            don't save your questions.
          </li>
          <li>
            <strong>Player photos</strong> load straight from the Premier League's image server, which, like any
            website, can see your IP address when your browser fetches them.
          </li>
          <li>
            <strong>Links to GitHub</strong> take you to GitHub, which has its own privacy policy.
          </li>
        </ul>
        <p>Football data comes from the public Fantasy Premier League API; nothing about you is sent there.</p>
      </ProseSection>

      <ProseSection title="Contact">
        <p>
          Questions about this policy? Open an issue on <a href={ISSUES_URL} target="_blank" rel="noopener noreferrer">GitHub</a>{' '}
          or see the <Link to="/contact">contact page</Link>. If this policy changes, the date at the top changes too.
        </p>
      </ProseSection>
    </ProsePage>
  );
}
