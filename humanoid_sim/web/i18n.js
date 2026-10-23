// UI language. Russian is the source text: pages and scripts are written in Russian, and each string is
// looked up as it is in /i18n/<lang>.json (en, zh), so a missing translation shows the Russian. The choice
// is remembered per browser (localStorage); without one the browser's own language decides (Chinese, Russian,
// otherwise English: the site is also read by people outside the lab). It is sent to the server as ?lang=,
// which translates its own messages, labels, the report and the journal (humanoid_sim/i18n.py).
export const LANGUAGES = { ru: 'Русский', en: 'English', zh: '中文' };
const LOCALES = { ru: 'ru-RU', en: 'en-US', zh: 'zh-CN' };
const TAGS = { ru: 'ru', en: 'en', zh: 'zh-Hans' };
const KEY = 'lab-language';
const fromBrowser = () => {
  const tag = (navigator.language || '').toLowerCase();
  return tag.startsWith('zh') ? 'zh' : tag.startsWith('ru') ? 'ru' : 'en';
};
export const lang = (() => {
  try {
    const value = localStorage.getItem(KEY);
    return value in LANGUAGES ? value : fromBrowser();
  } catch {
    return fromBrowser();
  }
})();
// Switching the language reloads the page, so every script starts in the new language.
export function setLanguage(value) {
  try {
    localStorage.setItem(KEY, value);
  } catch {}
  location.reload();
}
export const locale = LOCALES[lang];
let table = {};
if (lang !== 'ru')
  try {
    // Beside this script, so the static site (humanoid_sim.site_export) also works from a subfolder.
    table = await (await fetch(new URL(`i18n/${lang}.json`, import.meta.url), { cache: 'no-store' })).json();
  } catch {
    table = {};
  }
const fill = (text, params) =>
  params ? text.replace(/\{(\w+)\}/g, (all, key) => (key in params ? String(params[key]) : all)) : text;
// t('Осталось {n} с', { n: 5 }): the translation of the Russian text with its {placeholders} filled in.
export const t = (text, params) => fill(table[text] ?? text, params);
// tn(3, 'шаг|шага|шагов'): the plural form for the count (Russian one|few|many, English one|other,
// Chinese a single form).
export function tn(count, forms) {
  const list = t(forms).split('|'),
    category = new Intl.PluralRules(locale).select(count);
  return list[Math.min({ one: 0, few: 1, many: 2 }[category] ?? 1, list.length - 1)];
}
// A server URL with the language for the server's own texts; Russian needs none.
export const withLang = (url) =>
  lang === 'ru' ? url : url + (url.includes('?') ? '&' : '?') + 'lang=' + lang;
// Translates the page's static Russian: text nodes and the attributes people read, matched as whole strings
// with the whitespace of the HTML source squashed. Scripts translate what they build themselves with t().
const ATTRIBUTES = ['title', 'aria-label', 'alt', 'placeholder', 'label'];
const squash = (text) => text.replace(/\s+/g, ' ').trim();
export function translatePage(root = document.body) {
  document.documentElement.lang = TAGS[lang];
  if (lang === 'ru') return;
  document.title = t(squash(document.title));
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const key = squash(node.nodeValue);
    if (key && key in table && !['SCRIPT', 'STYLE'].includes(node.parentElement?.tagName))
      node.nodeValue = node.nodeValue.replace(
        /^(\s*)[\s\S]*?(\s*)$/,
        (all, lead, trail) => lead + table[key] + trail,
      );
  }
  for (const element of root.querySelectorAll('*'))
    for (const name of ATTRIBUTES) {
      const value = element.getAttribute(name);
      if (value && squash(value) in table) element.setAttribute(name, table[squash(value)]);
    }
  for (const input of root.querySelectorAll('input[value]:not([type])'))
    if (squash(input.value) in table) input.value = table[squash(input.value)];
}
