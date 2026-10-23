// The shared frame of the site's pages (site.html, journal.html, lab.html): header with the language buttons,
// footer, and where the data comes from. The same pages run two ways:
// - live, served by humanoid_sim.webapp on this PC: data from /api/lab/…, journal media from the server;
// - static, written by humanoid_sim.site_export for public hosting (<meta name="site-mode" content="static">):
//   data from data/<name>.<lang>.json and media from media/, no live race and no lab controls.
import { lang, t, setLanguage, translatePage, withLang } from './i18n.js';

export const STATIC = document.querySelector('meta[name="site-mode"]')?.content === 'static';
export { lang, t };

// Lab data by name: 'journal', 'lab' (the branch tree), … — the live API or the exported snapshot.
export async function data(name) {
  const url = STATIC ? `data/${name}.${lang}.json` : withLang(name === 'lab' ? '/api/lab' : `/api/lab/${name}`);
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

// A journal picture or clip by file name.
export const media = (name) =>
  STATIC ? `media/${encodeURIComponent(name)}` : `/api/lab/journal/image?name=${encodeURIComponent(name)}`;

// Labels as literal t('…') calls, so the translation scan (humanoid_sim/i18n.py) finds them.
const PAGES = [
  ['site.html', t('Главная')],
  ['journal.html', t('Журнал')],
  ['lab.html', t('Лаборатория')],
];

function header(active) {
  const bar = document.createElement('header');
  bar.className = 'site-header';
  const links = PAGES.map(
      ([file, label]) =>
        `<a href="${STATIC && file === 'site.html' ? 'index.html' : file}"${file === active ? ' aria-current="page"' : ''}>${label}</a>`,
    )
    .join('');
  const buttons = [
    ['en', 'EN'],
    ['zh', '中文'],
    ['ru', 'RU'],
  ]
    .map(([code, name]) => `<button type="button" data-lang="${code}" aria-pressed="${code === lang}">${name}</button>`)
    .join('');
  bar.innerHTML = `<div class="wrap">
<a class="brand" href="${STATIC ? 'index.html' : 'site.html'}"><svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2"><circle cx="14" cy="6" r="3"></circle><path d="M14 9v9M14 18l-5 8M14 18l5 8M8 13h12"></path></svg><b>G1 Lab</b><span>· ${t('Павел Мальцев')}</span></a>
<nav class="site-nav" aria-label="${t('Разделы')}">${links}</nav>
<div class="languages" role="group" aria-label="${t('Язык интерфейса')}">${buttons}</div></div>`;
  for (const button of bar.querySelectorAll('[data-lang]'))
    button.onclick = () => button.dataset.lang !== lang && setLanguage(button.dataset.lang);
  return bar;
}

function footer() {
  const bar = document.createElement('footer');
  bar.className = 'site-footer';
  bar.innerHTML = `<div class="wrap"><span>G1 Lab · MuJoCo, MuJoCo Warp, PyTorch · unitree_mujoco, g1_ctrl (unitree_rl_mjlab)</span><span>${t('Все числа на сайте — из записей лаборатории')}</span></div>`;
  return bar;
}

// Frames the page: header before <main>, footer after it, then translates the page's static Russian.
export function frame(active) {
  const main = document.querySelector('main');
  document.body.insertBefore(header(active), main);
  document.body.appendChild(footer());
  translatePage();
}
