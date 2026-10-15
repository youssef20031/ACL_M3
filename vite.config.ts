import { defineConfig, loadEnv, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';

// Public origin of the site, without a trailing slash. index.html's `%SITE_URL%` (canonical,
// Open Graph tags) and the generated robots.txt / sitemap.xml all use it. Override it with
// SITE_URL in the environment or a .env file; deploy/web.Dockerfile takes it as a build arg.
const DEFAULT_SITE_URL = 'https://20-215-33-13.sslip.io';

// Indexable routes for sitemap.xml. Keep in sync with the routes in src/App.tsx and the route
// list in deploy/Caddyfile. Left out on purpose: /settings (per-browser preferences), /design
// (internal reference page), and the /qa and /search redirects.
const SITEMAP_ROUTES: { path: string; changefreq: 'daily' | 'weekly' | 'monthly'; priority: number }[] = [
  { path: '/', changefreq: 'daily', priority: 1.0 },
  { path: '/fixtures', changefreq: 'daily', priority: 0.9 },
  { path: '/table', changefreq: 'daily', priority: 0.9 },
  { path: '/predictions', changefreq: 'daily', priority: 0.9 },
  { path: '/team-of-the-week', changefreq: 'daily', priority: 0.8 },
  { path: '/players', changefreq: 'daily', priority: 0.8 },
  { path: '/assistant', changefreq: 'weekly', priority: 0.7 },
  { path: '/compare', changefreq: 'weekly', priority: 0.6 },
  { path: '/trivia', changefreq: 'weekly', priority: 0.5 },
  { path: '/contact', changefreq: 'monthly', priority: 0.3 },
  { path: '/privacy', changefreq: 'monthly', priority: 0.2 },
  { path: '/terms', changefreq: 'monthly', priority: 0.2 },
];

function robotsTxt(siteUrl: string): string {
  return [
    'User-agent: *',
    'Allow: /',
    'Disallow: /api/',
    'Disallow: /docs',
    'Disallow: /openapi.json',
    'Disallow: /design',
    '',
    `Sitemap: ${siteUrl}/sitemap.xml`,
    '',
  ].join('\n');
}

function sitemapXml(siteUrl: string): string {
  const urls = SITEMAP_ROUTES.map(
    ({ path, changefreq, priority }) =>
      `  <url>\n    <loc>${siteUrl}${path}</loc>\n    <changefreq>${changefreq}</changefreq>\n    <priority>${priority.toFixed(1)}</priority>\n  </url>`,
  );
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...urls,
    '</urlset>',
    '',
  ].join('\n');
}

/** Fills `%SITE_URL%` in index.html and emits robots.txt and sitemap.xml into the build. */
function siteEssentials(siteUrl: string): Plugin {
  return {
    name: 'pitchme-site-essentials',
    transformIndexHtml: {
      order: 'pre',
      handler: (html) => html.replace(/%SITE_URL%/g, () => siteUrl),
    },
    generateBundle() {
      this.emitFile({ type: 'asset', fileName: 'robots.txt', source: robotsTxt(siteUrl) });
      this.emitFile({ type: 'asset', fileName: 'sitemap.xml', source: sitemapXml(siteUrl) });
    },
  };
}

// The dev server proxies the API so the app is same-origin, like production.
// API_PROXY_TARGET picks the backend: a local uvicorn by default, or the deployed site
// with `npm run dev:live` (mode "live" reads .env.live), so frontend work needs no
// Python, Neo4j or API keys.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.API_PROXY_TARGET || 'http://localhost:8000';
  const siteUrl = (env.SITE_URL || DEFAULT_SITE_URL).trim().replace(/\/+$/, '');
  if (!/^https?:\/\/[^/\s]+$/.test(siteUrl)) {
    throw new Error(`SITE_URL must be an origin like https://example.com (got "${siteUrl}")`);
  }
  return {
    plugins: [react(), siteEssentials(siteUrl)],
    server: {
      port: 3000,
      proxy: {
        '/api': { target, changeOrigin: true },
        '/health': { target, changeOrigin: true },
        '/ws': { target: target.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
      },
    },
  };
});
