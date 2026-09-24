// Upload the four DjVu scans through the UI, as one batch.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const BASE = 'http://localhost:8022';
const dir = path.resolve(__dirname, '../inputs/tiếng nga');
(async () => {
  const files = fs.readdirSync(dir).filter((f) => f.endsWith('.djvu')).map((f) => path.join(dir, f));
  const b = await chromium.launch({ headless: true, channel: 'chrome' });
  const p = await (await b.newContext()).newPage();
  const toasts = [];
  p.on('response', (r) => { if (r.url().endsWith('/documents/upload')) toasts.push(r.status()); });
  await p.goto(`${BASE}/login`);
  await p.fill('#floatingInput', 'admin');
  await p.fill('input[type=password]', 'admin');
  await p.click('button:has-text("Đăng nhập")');
  await p.waitForURL(/document|analytics|\/$/, { timeout: 30000 });
  await p.goto(`${BASE}/document`);
  await p.waitForSelector('table tbody');
  await p.locator('button.btn-primary:has(i.icon-plus)').click();
  const modal = p.locator('.modal-content');
  await modal.waitFor();
  await p.waitForFunction(() => {
    const el = document.querySelector('.modal-content input[type=file]');
    return !!el && el.multiple;
  }, { timeout: 30000 });
  await modal.locator('input[type=file]').setInputFiles(files);
  console.log('picked:', await modal.innerText().then((t) => t.split('\n').pop()));
  await modal.locator('select').selectOption({ label: 'Tiếng Nga' });
  const t0 = Date.now();
  await modal.locator('button:has-text("Lưu")').click();
  const deadline = Date.now() + 15 * 60_000;
  while (toasts.length < files.length && Date.now() < deadline) await p.waitForTimeout(3000);
  await p.waitForTimeout(2000);
  console.log('responses:', toasts.join(','), '| secs:', Math.round((Date.now() - t0) / 1000));
  console.log('toasts:', (await p.locator('.toast, ngb-toast').allInnerTexts()).join(' || '));
  await b.close();
})();
