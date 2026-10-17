// A headless Chromium browser (Edge or Chrome) driven over the Chrome DevTools Protocol, shared by
// render.mjs (brand images) and screenshots.mjs (README screenshots). Set BROWSER_PATH to use a
// specific Chromium-based browser.
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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

/** Starts the browser with a throwaway profile; returns a CDP client for its page and close(). */
export async function launch() {
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
