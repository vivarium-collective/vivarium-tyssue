// Generate the dashboard's own investigation report (investigation-<slug>-<date>.html)
// to ~/Downloads, headlessly. Drives a headless Chrome via CDP, loads the running
// dashboard (so it has the real _buildInvestigationReportHtml + helpers), replays
// the exact bundle-fetch chain from _generateInvestigationReport, and captures the
// built HTML — bypassing the client-side download trigger (and Safari's blocked
// Apple-Events JS). Node 25 globals: fetch, WebSocket.
//
// Usage: node scripts/gen_investigation_report.mjs <slug> <dashboardURL> [outDir]
import { spawn } from 'node:child_process';
import { existsSync, statSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

const slug = process.argv[2] || 'tumor-tyssue';
const dashURL = process.argv[3] || 'http://localhost:50456';
const outDir = process.argv[4] || join(homedir(), 'Downloads');
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9222;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const chrome = spawn(CHROME, [
  '--headless=new', `--remote-debugging-port=${PORT}`,
  '--user-data-dir=/tmp/chrome-report-gen', '--no-first-run', '--no-default-browser-check',
  '--disable-gpu', dashURL,
], { stdio: 'ignore' });

async function cdp() {
  // Wait for the devtools endpoint + a page target at the dashboard.
  let target = null;
  for (let i = 0; i < 60; i++) {
    try {
      const list = await fetch(`http://localhost:${PORT}/json`).then((r) => r.json());
      target = list.find((t) => t.type === 'page' && t.url.includes(new URL(dashURL).host));
      if (target?.webSocketDebuggerUrl) break;
    } catch {}
    await sleep(500);
  }
  if (!target) throw new Error('no Chrome page target for the dashboard');

  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map();
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  };
  const send = (method, params = {}) => new Promise((res) => {
    const mid = ++id; pending.set(mid, res); ws.send(JSON.stringify({ id: mid, method, params }));
  });

  await send('Page.enable');
  await send('Runtime.enable');
  // Route downloads to outDir instead of triggering a browser save dialog.
  await send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: outDir, eventsEnabled: true });
  // Give the SPA time to load its scripts.
  await sleep(3500);

  const date = new Date().toISOString().slice(0, 10);
  const out = join(outDir, `investigation-${slug}-${date}.html`);
  try { rmSync(out); } catch {}

  // _buildInvestigationReportHtml is module-private; _generateInvestigationReport
  // is on window and ends in _triggerDownload(<a download>). Set the current iset
  // and fire it — the download is captured to outDir by setDownloadBehavior.
  const trig = await send('Runtime.evaluate', {
    expression: `(window._currentIset=${JSON.stringify(slug)},
       (typeof window._generateInvestigationReport==='function'
         ? (window._generateInvestigationReport(),'started')
         : 'no-fn'))`,
    awaitPromise: false, returnByValue: true,
  });
  if (trig.result?.result?.value !== 'started') {
    throw new Error('_generateInvestigationReport not callable: ' + JSON.stringify(trig.result));
  }

  // Wait for the file to appear and stop growing (.crdownload -> final).
  let lastSize = -1, stable = 0;
  for (let i = 0; i < 120; i++) {
    await sleep(500);
    if (existsSync(out)) {
      const sz = statSync(out).size;
      if (sz === lastSize && sz > 1000) { if (++stable >= 3) break; } else { stable = 0; }
      lastSize = sz;
    }
  }
  if (!existsSync(out) || statSync(out).size < 1000) {
    throw new Error('download did not complete to ' + out);
  }
  console.log(`wrote ${out} (${Math.round(statSync(out).size / 1024)} KB)`);
  ws.close();
}

cdp().then(() => { chrome.kill(); process.exit(0); })
     .catch((e) => { console.error('ERROR:', e.message); chrome.kill(); process.exit(1); });
