// Renders the static brand images in public/ with a headless Chromium browser (Edge or Chrome):
//   og-image.png           1200x630 link preview, from scripts/site/og-image.html
//   favicon.ico            16/32/48, from public/favicon.svg (16px is a pixel-grid redraw, below)
//   icon-192.png, icon-512.png   rounded-square icon (transparent corners), from public/favicon.svg
//   apple-touch-icon.png   180x180, full-bleed square (iOS rounds it), no transparency
//   icon-maskable-512.png  full-bleed square, mark inside the 80% safe zone
//
//   node scripts/site/render.mjs            (from the repo root; needs `npm install` for the fonts)
//
// Set BROWSER_PATH to use a specific Chromium-based browser. The PNGs are then packed by
// pack_icons.py (Pillow): opaque images lose their alpha channel and the ICO gets exact-size
// frames. PYTHON picks the interpreter (default: the repo's .venv, then python3/python).
import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..', '..');
const publicDir = join(root, 'public');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- Icon artwork, derived from public/favicon.svg (the single source for the mark) ----------

const BRAND = '#37003C';
const faviconSvg = readFileSync(join(publicDir, 'favicon.svg'), 'utf8').trim();
const bgRect = faviconSvg.match(/<rect width="32" height="32"[^>]*\/>/);
if (!bgRect) throw new Error('public/favicon.svg: expected a 32x32 background <rect> as the first shape');
const markInner = faviconSvg
  .replace(/^<svg[^>]*>/, '')
  .replace(/<\/svg>$/, '')
  .replace(bgRect[0], '');

/** Full-bleed square background with the mark scaled about the centre. */
function squareIconSvg(scale) {
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">` +
    `<rect width="32" height="32" fill="${BRAND}"/>` +
    `<g transform="translate(16 16) scale(${scale}) translate(-16 -16)">${markInner}</g></svg>`
  );
}

function iconPage(svg, size) {
  return (
    `<!doctype html><html><head><style>html,body{margin:0;background:transparent;overflow:hidden}` +
    `svg{display:block;width:${size}px;height:${size}px}</style></head><body>${svg}</body></html>`
  );
}

// favicon.svg's 2-unit strokes fall between pixels at 16px and blur, so the 16px ICO frame is
// the same mark drawn on the pixel grid (half a pixel off-centre, so the halfway line is 1px).
const FAVICON_16_SVG =
  `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">` +
  `<rect width="16" height="16" rx="3.5" fill="${BRAND}"/>` +
  `<g fill="#00FF85" shape-rendering="crispEdges">` +
  `<rect x="2" y="4" width="13" height="1"/><rect x="2" y="12" width="13" height="1"/>` +
  `<rect x="2" y="5" width="1" height="7"/><rect x="14" y="5" width="1" height="7"/>` +
  `<rect x="8" y="5" width="1" height="7"/>` +
  `<rect x="7" y="6" width="3" height="1"/><rect x="7" y="10" width="3" height="1"/>` +
  `<rect x="6" y="7" width="1" height="3"/><rect x="10" y="7" width="1" height="3"/></g>` +
  `<rect x="8" y="8" width="1" height="1" fill="#04F5FF" shape-rendering="crispEdges"/></svg>`;

// The mark spans 24x18 of the 32-unit box (diagonal 30). Maskable icons are cropped to as little
// as a circle of 80% of the size, so scale the mark until its diagonal is ~70% of the icon.
const MASKABLE_SCALE = 0.75;
// iOS applies its own rounded mask; a little extra margin keeps the mark off the corners.
const APPLE_SCALE = 0.9;

const jobs = [
  { out: 'og-image.png', url: pathToFileURL(join(here, 'og-image.html')).href, width: 1200, height: 630, opaque: true },
  ...[16, 32, 48].map((s) => ({
    out: `favicon-${s}.png`,
    html: iconPage(s === 16 ? FAVICON_16_SVG : faviconSvg, s),
    width: s,
    height: s,
    ico: true,
  })),
  { out: 'icon-192.png', html: iconPage(faviconSvg, 192), width: 192, height: 192 },
  { out: 'icon-512.png', html: iconPage(faviconSvg, 512), width: 512, height: 512 },
  { out: 'apple-touch-icon.png', html: iconPage(squareIconSvg(APPLE_SCALE), 180), width: 180, height: 180, opaque: true },
  { out: 'icon-maskable-512.png', html: iconPage(squareIconSvg(MASKABLE_SCALE), 512), width: 512, height: 512, opaque: true },
];

// ---- Headless browser over the Chrome DevTools Protocol -------------------------------------

function findBrowser() {
  const candidates = [
    process.env.BROWSER_PATH,
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/microsoft-edge',
  ].filter(Boolean);
  const found = candidates.find((p) => existsSync(p));
  if (!found) throw new Error('No Chromium-based browser found; set BROWSER_PATH');
  return found;
}

async function launch() {
  const profile = mkdtempSync(join(tmpdir(), 'pitchme-render-'));
  const proc = spawn(
    findBrowser(),
    [
      '--headless=new',
      '--disable-gpu',
      '--hide-scrollbars',
      '--no-first-run',
      '--no-default-browser-check',
      '--allow-file-access-from-files',
      '--font-render-hinting=none',
      '--remote-debugging-port=0',
      `--user-data-dir=${profile}`,
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  const browserWs = await new Promise((res, rej) => {
    let buf = '';
    const timer = setTimeout(() => rej(new Error('Browser did not start')), 20000);
    proc.stderr.on('data', (d) => {
      buf += d;
      const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
      if (m) {
        clearTimeout(timer);
        res(m[1]);
      }
    });
    proc.on('exit', (code) => rej(new Error(`Browser exited early (${code})`)));
  });
  const port = new URL(browserWs).port;
  let page;
  for (let i = 0; i < 40 && !page; i++) {
    const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    page = targets.find((t) => t.type === 'page');
    if (!page) await sleep(250);
  }
  if (!page) throw new Error('No page target');
  const cdp = await connect(page.webSocketDebuggerUrl);
  const exited = new Promise((res) => proc.once('exit', res));
  const close = async () => {
    try {
      const b = await connect(browserWs);
      b.send('Browser.close').catch(() => {});
    } catch {
      /* already gone */
    }
    await Promise.race([exited, sleep(5000)]);
    if (proc.exitCode === null) proc.kill();
    // Helper processes can hold the profile for a moment after the browser exits.
    for (let i = 0; i < 20; i++) {
      try {
        rmSync(profile, { recursive: true, force: true });
        return;
      } catch {
        await sleep(250);
      }
    }
    console.warn(`could not remove the temporary browser profile ${profile}`);
  };
  return { cdp, close };
}

async function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => {
    ws.onopen = res;
    ws.onerror = rej;
  });
  let id = 0;
  const pending = new Map();
  const listeners = new Map();
  ws.onmessage = (m) => {
    const msg = JSON.parse(m.data);
    if (msg.id && pending.has(msg.id)) {
      const { res, rej } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) rej(new Error(`${msg.error.message} (${msg.error.code})`));
      else res(msg.result);
    } else if (msg.method && listeners.has(msg.method)) {
      listeners.get(msg.method).splice(0).forEach((fn) => fn(msg.params));
    }
  };
  return {
    send: (method, params = {}) =>
      new Promise((res, rej) => {
        const i = ++id;
        pending.set(i, { res, rej });
        ws.send(JSON.stringify({ id: i, method, params }));
      }),
    once: (method) =>
      new Promise((res) => {
        if (!listeners.has(method)) listeners.set(method, []);
        listeners.get(method).push(res);
      }),
  };
}

async function capture(cdp, { url, html, width, height, opaque }) {
  await cdp.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  await cdp.send(
    'Emulation.setDefaultBackgroundColorOverride',
    opaque ? {} : { color: { r: 0, g: 0, b: 0, a: 0 } },
  );
  const loaded = cdp.once('Page.loadEventFired');
  await cdp.send('Page.navigate', {
    url: url ?? `data:text/html;base64,${Buffer.from(html).toString('base64')}`,
  });
  await loaded;
  const { result } = await cdp.send('Runtime.evaluate', {
    expression: `document.fonts.ready.then(() => [...document.fonts].map((f) => f.family + ' ' + f.status))`,
    awaitPromise: true,
    returnByValue: true,
  });
  const failed = (result.value ?? []).filter((s) => !s.endsWith(' loaded'));
  if (failed.length) throw new Error(`Fonts did not load: ${failed.join(', ')}`);
  await sleep(150);
  const shot = await cdp.send('Page.captureScreenshot', {
    format: 'png',
    clip: { x: 0, y: 0, width, height, scale: 1 },
    captureBeyondViewport: false,
  });
  return Buffer.from(shot.data, 'base64');
}

// ---- Main --------------------------------------------------------------------------------------

const work = mkdtempSync(join(tmpdir(), 'pitchme-icons-'));
const { cdp, close } = await launch();
try {
  await cdp.send('Page.enable');
  for (const job of jobs) {
    writeFileSync(join(work, job.out), await capture(cdp, job));
    console.log('rendered', job.out);
  }
} finally {
  await close();
}

const manifest = {
  out: publicDir,
  opaque: jobs.filter((j) => j.opaque).map((j) => j.out),
  copy: jobs.filter((j) => !j.ico).map((j) => j.out),
  ico: { file: 'favicon.ico', frames: jobs.filter((j) => j.ico).map((j) => j.out) },
};
writeFileSync(join(work, 'manifest.json'), JSON.stringify(manifest, null, 2));

const python =
  process.env.PYTHON ||
  [join(root, '.venv/Scripts/python.exe'), join(root, '.venv/bin/python')].find((p) => existsSync(p)) ||
  (process.platform === 'win32' ? 'python' : 'python3');
const packed = spawnSync(python, [join(here, 'pack_icons.py'), work], { stdio: 'inherit' });
rmSync(work, { recursive: true, force: true });
if (packed.status !== 0) {
  console.error('pack_icons.py failed (needs Pillow)');
  process.exit(1);
}
