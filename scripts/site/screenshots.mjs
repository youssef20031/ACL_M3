// Captures the README screenshots in docs/screenshots/ from a running site with a headless
// Chromium browser (Edge or Chrome, see cdp.mjs):
//
//   node scripts/site/screenshots.mjs [site URL]     (default: $SITE_URL, then the live site)
//
// The consent banner is answered "declined" up front, so these visits aren't counted. The
// assistant shot asks one real question, which goes through the LLM and the per-client rate limit
// like any other. Pages that take router state (/assistant, /players) get it through
// history.state, which is where React Router reads it on load.
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch, sleep } from './cdp.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, '..', '..', 'docs', 'screenshots');
const site = (process.argv[2] || process.env.SITE_URL || 'https://20-215-33-13.sslip.io').replace(/\/+$/, '');

const QUESTION = 'Who should I captain this gameweek?';
const PLAYER = 'Erling Haaland';
const SEASON = '2026-27';

/** The search result /players opens a profile from, as the search box would pass it. */
async function searchResult(name) {
  const res = await fetch(`${site}/api/players/search`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ query: name, limit: 1, include_avatars: true }),
  });
  const [player] = (await res.json()).players ?? [];
  if (!player) throw new Error(`No search result for ${name}`);
  return player;
}

const shots = [
  { out: 'home.webp', path: '/', width: 1280, height: 800 },
  {
    out: 'assistant.webp',
    path: '/assistant',
    state: async () => ({ question: QUESTION }),
    // The answer box only exists once the answer has started; aria-busy turns false when it ends
    ready: `!!document.querySelector('[aria-busy="false"]')`,
    // Start the view at the question rather than the bottom of the answer
    after: `[...document.querySelectorAll('main *')]
      .find((e) => e.childElementCount === 0 && e.textContent.trim() === ${JSON.stringify(QUESTION)})
      ?.scrollIntoView({ block: 'start' })`,
    width: 1280,
    height: 1240,
    timeout: 120_000,
  },
  { out: 'team-of-the-week.webp', path: '/team-of-the-week', width: 1280, height: 920 },
  {
    out: 'player.webp',
    path: '/players',
    state: async () => ({ player: await searchResult(PLAYER), season: SEASON }),
    width: 1280,
    height: 920,
  },
  { out: 'predictions.webp', path: '/predictions', width: 1280, height: 800 },
];

// Loaded, every image in view decoded (lazy ones further down never load), and no skeletons or
// streaming answers left on the page
const SETTLED = `document.readyState === 'complete'
  && [...document.images].every((i) => i.complete || i.getBoundingClientRect().top > innerHeight)
  && !document.querySelector('[aria-busy="true"], .animate-pulse')`;

async function evaluate(cdp, expression) {
  const { result } = await cdp.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  return result.value;
}

async function capture(cdp, { path, state, ready = 'true', after, width, height, mobile = false, timeout = 30_000 }) {
  await cdp.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 2, mobile });
  const setup =
    `try { localStorage.setItem('pm-analytics-consent', 'declined'); } catch {}` +
    (state ? `history.replaceState({ usr: ${JSON.stringify(await state())}, key: 'shot', idx: 0 }, '');` : '');
  const { identifier } = await cdp.send('Page.addScriptToEvaluateOnNewDocument', { source: setup });
  try {
    const loaded = cdp.once('Page.loadEventFired');
    await cdp.send('Page.navigate', { url: site + path });
    await loaded;
    await evaluate(cdp, 'document.fonts.ready.then(() => true)');
    // Settled twice in a row, a second apart, so late queries and images have landed
    const deadline = Date.now() + timeout;
    let streak = 0;
    while (streak < 2) {
      if (Date.now() > deadline) throw new Error(`${path} did not settle within ${timeout / 1000}s`);
      await sleep(1000);
      streak = (await evaluate(cdp, `(${ready}) && ${SETTLED}`)) ? streak + 1 : 0;
    }
    if (after) {
      await evaluate(cdp, after);
      await sleep(500);
    }
    const shot = await cdp.send('Page.captureScreenshot', {
      format: 'webp',
      quality: 85,
      clip: { x: 0, y: 0, width, height, scale: 1 },
    });
    return Buffer.from(shot.data, 'base64');
  } finally {
    await cdp.send('Page.removeScriptToEvaluateOnNewDocument', { identifier });
  }
}

mkdirSync(outDir, { recursive: true });
const { cdp, close } = await launch();
try {
  await cdp.send('Page.enable');
  for (const shot of shots) {
    const data = await capture(cdp, shot);
    writeFileSync(join(outDir, shot.out), data);
    console.log(`captured ${shot.out} (${Math.round(data.length / 1024)} KB)`);
  }
} finally {
  await close();
}
