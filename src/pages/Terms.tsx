import { Link } from 'react-router-dom';
import { ProsePage, ProseSection } from '../components/legal/Prose';
import { usePageMeta } from '../hooks/usePageMeta';

export function Terms() {
  usePageMeta({
    title: 'Terms of use',
    description: 'The terms for using Pitch Me: a free, fan-made FPL project whose predictions and answers are for fun and information.',
  });

  return (
    <ProsePage
      eyebrow="Legal"
      title="Terms of use"
      description="Short and plain: what Pitch Me is, what it isn't, and how to use it fairly."
      updated="15 October 2026"
    >
      <ProseSection title="A fan project">
        <p>
          Pitch Me is a free, non-commercial project built by fans. It is <strong>not affiliated with, endorsed by or
          connected to</strong> the Premier League, Fantasy Premier League or any club. Club names and crests belong to
          their respective owners and are shown only to identify teams. Football data comes from the public Fantasy
          Premier League API.
        </p>
      </ProseSection>

      <ProseSection title="Predictions and answers">
        <p>
          Predicted points, the predicted Team of the Week and the assistant's answers come from statistical models and
          a language model. They can be wrong or out of date. Use them for fun and as one input to your own decisions.
          They are <strong>not betting, financial or professional advice</strong>.
        </p>
      </ProseSection>

      <ProseSection title="Fair use">
        <ul>
          <li>Don't overload the site or scrape it automatically. The assistant has per-visitor limits.</li>
          <li>Don't try to break, bypass or probe the site's security or admin features.</li>
          <li>Don't use the assistant to produce anything unlawful or abusive.</li>
        </ul>
      </ProseSection>

      <ProseSection title="No guarantees">
        <p>
          The site is provided as it is, without any warranty. It may change, be unavailable or stop at any time. As far
          as the law allows, we aren't liable for losses arising from using it, including FPL decisions made with it.
        </p>
      </ProseSection>

      <ProseSection title="Privacy and contact">
        <p>
          How data is handled is described in the <Link to="/privacy">privacy policy</Link>. Questions or problems? See
          the <Link to="/contact">contact page</Link>.
        </p>
      </ProseSection>
    </ProsePage>
  );
}
