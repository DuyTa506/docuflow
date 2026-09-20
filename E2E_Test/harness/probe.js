const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ headless: true, channel: 'chrome' });
  const p = await b.newPage({ viewport: { width: 1600, height: 1000 } });
  p.on('console', m => m.type()==='error' && console.log('CONSOLE', m.text().slice(0,200)));
  await p.goto('http://localhost:8022/login');
  await p.fill('#floatingInput', 'admin');
  await p.fill('input[type=password]', 'admin');
  await p.click('button:has-text("Đăng nhập")');
  await p.waitForTimeout(3000);
  console.log('url', p.url());
  await p.goto('http://localhost:8022/document'); await p.waitForTimeout(3000);
  await p.screenshot({ path: '../results/probe.png' });
  console.log((await p.locator('table').innerText()).slice(0,800));
  await b.close();
})();
