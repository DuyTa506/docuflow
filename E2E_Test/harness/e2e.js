// DocuFlow E2E — drives the real Angular UI like a user (login, upload, start
// translate / digest, download PDF+DOCX) for every file under ../inputs, and
// logs every UI/network error. Resumable: progress lives in ../results/state.json.
//
//   node e2e.js            # run / resume
//
// Besides the user-visible downloads it also dumps the raw API payloads
// (OCR text, translation, digest, task history) for quality review.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE = process.env.E2E_BASE || 'http://localhost:8022';
const API = `${BASE}/api/v2`;
const ROOT = path.resolve(__dirname, '..');
const INPUTS = path.join(ROOT, 'inputs');
const OUT = path.join(ROOT, process.env.E2E_OUT || 'results');
const BATCH = process.env.E2E_BATCH === '1'; // pick a whole folder at once
const STATE_FILE = path.join(OUT, 'state.json');
const POLL_MS = 60_000;
const MAX_RETRIES = 1; // a user clicks "Chạy lại" once, then gives up
const LANG_LABEL = { 'tiếng anh': 'Tiếng Anh', 'tiếng nga': 'Tiếng Nga', 'tiếng trung': 'Tiếng Trung' };
const COLS = { ocr: 3, trans: 4, sum: 5 };

fs.mkdirSync(OUT, { recursive: true });
const state = fs.existsSync(STATE_FILE) ? JSON.parse(fs.readFileSync(STATE_FILE, 'utf8')) : { docs: {} };
const saveState = () => fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
const now = () => new Date().toISOString();
const log = (...a) => console.log(now(), ...a);
function logError(kind, detail) {
  fs.appendFileSync(path.join(OUT, 'errors.jsonl'), JSON.stringify({ ts: now(), kind, ...detail }) + '\n');
  log('ERROR', kind, JSON.stringify(detail).slice(0, 300));
}
function logEvent(file, event, extra = {}) {
  fs.appendFileSync(path.join(OUT, 'events.jsonl'), JSON.stringify({ ts: now(), file, event, ...extra }) + '\n');
  log(file, event, JSON.stringify(extra).slice(0, 200));
}
const slug = (f) => f.replace(/\.[^.]+$/, '').replace(/[^\p{L}\p{N}]+/gu, '_').slice(0, 60);

function listInputs() {
  const out = [];
  for (const dir of fs.readdirSync(INPUTS)) {
    for (const f of fs.readdirSync(path.join(INPUTS, dir))) {
      out.push({ file: f, full: path.join(INPUTS, dir, f), lang: LANG_LABEL[dir] || 'Tiếng Anh' });
    }
  }
  return out.sort((a, b) => fs.statSync(a.full).size - fs.statSync(b.full).size); // small first
}

// ── API side-channel (for dumps only; all actions go through the UI) ──
let apiToken = null;
async function api(p, opts = {}) {
  if (!apiToken) {
    const r = await fetch(`${API}/auth/login`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'admin', password: 'admin' }),
    });
    apiToken = (await r.json()).access_token;
  }
  const r = await fetch(`${API}/${p}`, { ...opts, headers: { Authorization: `Bearer ${apiToken}`, ...(opts.headers || {}) } });
  if (r.status === 401) { apiToken = null; return api(p, opts); }
  return r;
}

// ── UI helpers ──
async function login(page) {
  await page.goto(`${BASE}/login`);
  await page.fill('#floatingInput', 'admin');
  await page.fill('input[type=password]', 'admin');
  await page.click('button:has-text("Đăng nhập")');
  await page.waitForURL(/document|analytics|\/$/, { timeout: 30_000 });
  await page.goto(`${BASE}/document`);
  await page.waitForSelector('table tbody');
}

async function ensureOnDocuments(page) {
  if (!page.url().includes('/document')) await login(page);
  if (page.url().includes('/login')) await login(page);
}

async function findRow(page, file) {
  await ensureOnDocuments(page);
  const input = page.locator('input#text');
  await input.fill(file.replace(/\.[^.]+$/, ''));
  await input.press('Enter');
  await page.waitForTimeout(2500);
  const rows = page.locator('table tbody tr').filter({ has: page.locator('td:nth-child(2)', { hasText: file }) });
  const n = await rows.count();
  if (n === 0) return null;
  if (n > 1) {
    // A re-run uploads the same filenames again: act on today's row, not the
    // previous run's document.
    const today = new Date().toLocaleDateString('en-GB');
    for (let i = 0; i < n; i++) {
      const created = (await rows.nth(i).locator('td:nth-child(3)').innerText()).trim();
      if (created === today) return rows.nth(i);
    }
    logError('ui.duplicate_rows', { file, rows: n });
  }
  return rows.first();
}

async function cellState(row, col) {
  const cell = row.locator(`td:nth-child(${COLS[col] + 1})`);
  if (await cell.locator('button[title="Tải xuống"]').count()) return 'COMPLETED';
  const txt = (await cell.innerText()).trim();
  if (txt.includes('Chạy lại')) return 'FAILED';
  if (txt.includes('Đang chạy')) return 'RUNNING';
  if (txt.includes('Đang chờ')) return 'PENDING';
  if (txt.includes('Bắt đầu')) return 'IDLE';
  return `UNKNOWN(${txt})`;
}

async function clickStart(page, row, col) {
  const cell = row.locator(`td:nth-child(${COLS[col] + 1})`);
  const btn = cell.locator('button').first();
  if (await btn.isDisabled()) return false;
  const resp = page.waitForResponse((r) => r.request().method() === 'POST' && /\/(extract|translations|analysis)$/.test(r.url()), { timeout: 60_000 }).catch(() => null);
  await btn.click();
  const r = await resp;
  return r ? { status: r.status(), body: (await r.text().catch(() => '')).slice(0, 500) } : { status: 'no-response' };
}

async function download(page, row, col, format, dest) {
  const cell = row.locator(`td:nth-child(${COLS[col] + 1})`);
  await cell.locator('button[title="Tải xuống"]').click();
  const started = Date.now();
  const dl = page.waitForEvent('download', { timeout: 45 * 60_000 });
  // The FE turns its progress toast into "Tải … thất bại" on an HTTP error —
  // stop waiting as soon as that appears instead of sitting out the timeout.
  const fails = page.locator('.toast, ngb-toast').filter({ hasText: 'thất bại' });
  const before = await fails.count();
  const failToast = (async () => {
    const deadline = Date.now() + 45 * 60_000;
    while (Date.now() < deadline) {
      if ((await fails.count().catch(() => 0)) > before) throw new Error('FE toast: ' + (await fails.last().innerText().catch(() => '')));
      await page.waitForTimeout(2000);
    }
    throw new Error('no download / no toast');
  })();
  failToast.catch(() => {});
  await page.locator('[ngbdropdownmenu].show button, .dropdown-menu.show button').filter({ hasText: format.toUpperCase() }).first().click();
  try {
    const d = await Promise.race([dl, failToast]);
    await d.saveAs(dest);
    return { ok: true, secs: Math.round((Date.now() - started) / 1000), bytes: fs.statSync(dest).size };
  } catch (e) {
    return { ok: false, secs: Math.round((Date.now() - started) / 1000), error: String(e).slice(0, 300) };
  }
}

async function upload(page, item) {
  await ensureOnDocuments(page);
  await page.locator('button.btn-primary:has(i.icon-plus)').click();
  const modal = page.locator('.modal-content');
  await modal.waitFor();
  await modal.locator('input[type=file]').setInputFiles(item.full);
  await modal.locator('select').selectOption({ label: item.lang });
  const respP = page.waitForResponse((r) => r.url().endsWith('/documents/upload'), { timeout: 20 * 60_000 }).catch((e) => e);
  await modal.locator('button:has-text("Lưu")').click();
  const resp = await respP;
  if (resp instanceof Error) return { status: 'timeout', error: String(resp) };
  const body = await resp.text().catch(() => '');
  let json = {}; try { json = JSON.parse(body); } catch {}
  await page.waitForTimeout(1500);
  const toasts = await page.locator('.toast, ngb-toast').allInnerTexts().catch(() => []);
  return { status: resp.status(), body: body.slice(0, 500), document_id: json.document_id, toasts };
}

// One modal, many files — what a user does with a folder, and what stresses
// the upload path and the extraction queue at once.
async function uploadBatch(page, group) {
  await ensureOnDocuments(page);
  await page.locator('button.btn-primary:has(i.icon-plus)').click();
  const modal = page.locator('.modal-content');
  await modal.waitFor();
  // The input is in the DOM before Angular applies [multiple]; setting files
  // too early fails with "Non-multiple file input".
  await page.waitForFunction(() => {
    const el = document.querySelector('.modal-content input[type=file]');
    return !!el && el.multiple;
  }, { timeout: 30_000 });
  await modal.locator('input[type=file]').setInputFiles(group.map((i) => i.full));
  await modal.locator('select').selectOption({ label: group[0].lang });
  const seen = [];
  const onResp = (r) => { if (r.url().endsWith('/documents/upload')) seen.push({ status: r.status(), at: now() }); };
  page.on('response', onResp);
  const t0 = Date.now();
  await modal.locator('button:has-text("Lưu")').click();
  const deadline = Date.now() + 90 * 60_000;
  while (seen.length < group.length && Date.now() < deadline) {
    await page.waitForTimeout(5000);
    const toast = await page.locator('.toast, ngb-toast').allInnerTexts().catch(() => []);
    if (toast.some((t) => /Tải lên .*lỗi|Đã tải lên/.test(t))) break;
  }
  page.off('response', onResp);
  const secs = Math.round((Date.now() - t0) / 1000);
  const toasts = await page.locator('.toast, ngb-toast').allInnerTexts().catch(() => []);
  logEvent(`BATCH ${group[0].lang}`, 'uploaded', { files: group.length, responses: seen.length, secs, toasts });
  // Map file -> document_id from the API: the batch reports no ids itself.
  const list = await (await api('documents?limit=100')).json().catch(() => ({}));
  const byName = new Map();
  for (const doc of list.items || []) byName.set(doc.original_filename || doc.title, doc.id);
  for (const item of group) {
    const d = state.docs[item.file];
    const id = byName.get(item.file);
    d.upload = { batch: true, secs, at: now(), responses: seen.length, ok: !!id };
    if (id) { d.document_id = id; d.started.ocr = now(); }
    else { d.upload_rejected = true; logError('upload.rejected', { file: item.file, batch: true, toasts }); }
    logEvent(item.file, 'uploaded', d.upload);
  }
  saveState();
}

async function dumpApi(doc, dir) {
  const id = doc.document_id;
  const save = async (name, p, opts) => {
    try {
      const r = await api(p, opts);
      const t = await r.text();
      fs.writeFileSync(path.join(dir, name), t);
      if (r.status >= 400) logError('api.dump', { file: doc.file, p, status: r.status, body: t.slice(0, 300) });
    } catch (e) { logError('api.dump', { file: doc.file, p, error: String(e) }); }
  };
  await save('tasks.json', `tasks?document_id=${id}`);
  await save('document.json', `documents/${id}`);
  await save('ocr_text.json', `documents/${id}/text`);
  await save('pipeline_status.json', `documents/${id}/pipeline-status`);
  await save('digest.json', `documents/${id}/digest`, { method: 'POST' });
  const tr = await (await api(`documents/${id}/translations`)).json().catch(() => []);
  if (Array.isArray(tr) && tr[0]) await save('translation.json', `documents/${id}/translations/${tr[0].id}`);
}

async function main() {
  const items = listInputs();
  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
  const page = await ctx.newPage();
  page.setDefaultTimeout(60_000);
  page.on('console', (m) => { if (m.type() === 'error') logError('ui.console', { text: m.text().slice(0, 500), url: page.url() }); });
  page.on('pageerror', (e) => logError('ui.pageerror', { text: String(e).slice(0, 500) }));
  page.on('response', (r) => {
    if (r.status() >= 400 && r.url().includes('/api/')) {
      r.text().then((b) => logError('http', { status: r.status(), method: r.request().method(), url: r.url().replace(BASE, ''), body: b.slice(0, 400) })).catch(() => {});
    }
  });
  page.on('requestfailed', (r) => { if (r.url().includes('/api/') && !r.url().includes('/events')) logError('http.failed', { url: r.url().replace(BASE, ''), err: r.failure()?.errorText }); });
  await login(page);

  // 1. Upload everything — one file at a time, or a whole folder at once.
  const fresh = [];
  for (const item of items) {
    const d = (state.docs[item.file] ||= { file: item.file, lang: item.lang, bytes: fs.statSync(item.full).size, retries: { ocr: 0, trans: 0, sum: 0 }, started: {}, finished: {} });
    if (!d.document_id && !d.upload_rejected) fresh.push(item);
  }
  if (BATCH) {
    for (const lang of [...new Set(fresh.map((i) => i.lang))]) {
      await uploadBatch(page, fresh.filter((i) => i.lang === lang));
    }
  }
  for (const item of items) {
    const d = state.docs[item.file];
    if (d.document_id || d.upload_rejected) continue;
    const t0 = Date.now();
    const r = await upload(page, item);
    d.upload = { ...r, secs: Math.round((Date.now() - t0) / 1000), at: now() };
    if (r.document_id) { d.document_id = r.document_id; d.started.ocr = now(); }
    else { d.upload_rejected = true; logError('upload.rejected', { file: item.file, ...r }); }
    logEvent(item.file, 'uploaded', d.upload);
    saveState();
  }

  // 2. Poll: start translate + digest once OCR is done, retry failures once,
  //    download everything when a document's three stages settle.
  while (true) {
    let open = 0;
    for (const item of items) {
      const d = state.docs[item.file];
      if (!d?.document_id || d.done) continue;
      open++;
      try {
        const row = await findRow(page, item.file);
        if (!row) { logError('ui.row_missing', { file: item.file }); continue; }
        const st = {};
        for (const col of ['ocr', 'trans', 'sum']) st[col] = await cellState(row, col);
        if (JSON.stringify(st) !== JSON.stringify(d.last)) { logEvent(item.file, 'status', st); d.last = st; }
        for (const col of ['ocr', 'trans', 'sum']) {
          if (st[col] === 'COMPLETED' && d.started[col] && !d.finished[col]) d.finished[col] = now();
          if (st[col] === 'FAILED' && !d.finished[col]) {
            if (d.retries[col] < MAX_RETRIES) {
              d.retries[col]++;
              const r = await clickStart(page, row, col);
              logEvent(item.file, `retry.${col}`, r);
            } else { d.finished[col] = now(); d.failed = { ...(d.failed || {}), [col]: true }; logError('stage.failed', { file: item.file, stage: col }); }
          }
        }
        if (st.ocr === 'COMPLETED') {
          for (const col of ['trans', 'sum']) {
            if (st[col] === 'IDLE' && !d.started[col]) {
              const r = await clickStart(page, row, col);
              d.started[col] = now();
              logEvent(item.file, `start.${col}`, r);
              if (typeof r.status === 'number' && r.status >= 400) {
                // Backend refused to start (button stays "Bắt đầu") — nothing to wait for.
                d.finished[col] = now();
                d.failed = { ...(d.failed || {}), [col]: `start ${r.status}: ${r.body}` };
                logError('stage.start_rejected', { file: item.file, stage: col, ...r });
              }
            }
          }
        }
        // OCR failed for good → translate/digest can never start.
        if (d.failed?.ocr) {
          d.finished.trans ||= now(); d.finished.sum ||= now();
          d.failed.trans ||= 'blocked: OCR failed'; d.failed.sum ||= 'blocked: OCR failed';
        }
        saveState();

        const settled = ['ocr', 'trans', 'sum'].every((c) => d.finished[c]);
        if (settled) {
          const dir = path.join(OUT, slug(item.file));
          fs.mkdirSync(dir, { recursive: true });
          d.downloads = {};
          for (const col of ['ocr', 'trans', 'sum']) {
            if (d.failed?.[col] || d.last?.[col] !== 'COMPLETED') continue;
            for (const fmt of ['pdf', 'docx']) {
              const fresh = await findRow(page, item.file);
              const r = await download(page, fresh, col, fmt, path.join(dir, `${col}.${fmt}`));
              d.downloads[`${col}.${fmt}`] = r;
              logEvent(item.file, `download.${col}.${fmt}`, r);
              if (!r.ok) logError('download.failed', { file: item.file, col, fmt, ...r });
            }
          }
          // Eye modal — what the user sees as a quick preview.
          const fresh = await findRow(page, item.file);
          await fresh.locator('button:has(i.icon-eye)').click();
          await page.waitForTimeout(4000);
          await page.screenshot({ path: path.join(dir, 'details_modal.png'), fullPage: true });
          await page.keyboard.press('Escape').catch(() => {});
          await page.locator('.modal .btn-close').first().click({ timeout: 3000 }).catch(() => {});
          await dumpApi(d, dir);
          d.done = now();
          logEvent(item.file, 'done', { failed: d.failed || null });
          saveState();
        }
      } catch (e) {
        logError('harness', { file: item.file, error: String(e).slice(0, 500) });
        await page.screenshot({ path: path.join(OUT, `harness_err_${Date.now()}.png`) }).catch(() => {});
        await page.goto(`${BASE}/document`).catch(() => {});
      }
    }
    if (open === 0) break;
    await page.waitForTimeout(POLL_MS);
  }
  log('ALL DONE');
  await browser.close();
}

main().catch((e) => { logError('fatal', { error: String(e.stack || e) }); process.exit(1); });
