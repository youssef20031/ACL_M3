import { useEffect } from 'react';

const SITE_NAME = 'Pitch Me';

interface PageMeta {
  /** Page name, e.g. "Fixtures". Omit on the home page to use the site name alone. */
  title?: string;
  /** One or two sentences, ~150 characters: shown in search results and link previews. */
  description: string;
}

function setMeta(selector: string, attr: 'name' | 'property', key: string, content: string) {
  let el = document.head.querySelector<HTMLMetaElement>(selector);
  if (!el) {
    el = document.createElement('meta');
    el.setAttribute(attr, key);
    document.head.appendChild(el);
  }
  el.setAttribute('content', content);
}

/**
 * Per-page <title>, meta description, Open Graph title/description/url and canonical link.
 *
 * Search engines render the SPA and read these; social scrapers don't run JS, so index.html
 * keeps site-level defaults for link previews.
 */
export function usePageMeta({ title, description }: PageMeta) {
  useEffect(() => {
    const fullTitle = title ? `${title} · ${SITE_NAME}` : `${SITE_NAME} · FPL stats, predictions & AI assistant`;
    const url = `${window.location.origin}${window.location.pathname}`;
    document.title = fullTitle;
    setMeta('meta[name="description"]', 'name', 'description', description);
    setMeta('meta[property="og:title"]', 'property', 'og:title', fullTitle);
    setMeta('meta[property="og:description"]', 'property', 'og:description', description);
    setMeta('meta[property="og:url"]', 'property', 'og:url', url);
    setMeta('meta[name="twitter:title"]', 'name', 'twitter:title', fullTitle);
    setMeta('meta[name="twitter:description"]', 'name', 'twitter:description', description);

    let canonical = document.head.querySelector<HTMLLinkElement>('link[rel="canonical"]');
    if (!canonical) {
      canonical = document.createElement('link');
      canonical.rel = 'canonical';
      document.head.appendChild(canonical);
    }
    canonical.href = url;
  }, [title, description]);
}
