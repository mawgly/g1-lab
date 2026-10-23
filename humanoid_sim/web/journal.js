// The lab journal (journal.html): entries, the lab council, the roadmap stages with their stories and the latest
// goal check, one view at a time (tabs, remembered in the URL hash). The research supervisor and the chronicler
// write <lab>/journal/journal.json (humanoid_sim/journal.py); this page only reads it, already in the page language:
// live from the lab server (again every 30 s while the page is shown) or from the static export (site.js data()).
// Only the page's own labels go through t().
import { STATIC, t, data, media, frame } from './site.js';
import { tn, locale, withLang } from './i18n.js';

frame('journal.html');
const $ = (id) => document.getElementById(id);
const main = document.querySelector('main');
const esc = (v) =>
  String(v ?? '').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );
const fmt = (v, digits = 0) =>
  v == null ? '—' : Number(v).toLocaleString(locale, { maximumFractionDigits: digits });
const when = (at) =>
  new Date(at * 1000).toLocaleString(locale, {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
const time = (at) => `<time datetime="${new Date(at * 1000).toISOString()}">${esc(when(at))}</time>`;
const paragraphs = (text) =>
  String(text || '')
    .split(/\n\s*\n/)
    .filter((p) => p.trim())
    .map((p) => `<p>${esc(p.trim())}</p>`)
    .join('');
// A sentence ends at . ! ? before a space and a capital (not «No. 6», «0.5»), or at a Chinese full stop.
const firstSentence = (text, limit = 160) => {
  const sentence = String(text || '').split(/(?<=[.!?])\s+(?=[^\p{Ll}\d\s])|(?<=[。！？])/u)[0];
  return sentence.length > limit ? sentence.slice(0, limit - 1) + '…' : sentence;
};

// Who speaks in the council (humanoid_sim.journal.ROLES), roadmap states and path verdicts.
const ROLES = {
  lead: t('Руководитель'),
  critic: t('Критик'),
  scout: t('Разведчик'),
  analyst: t('Аналитик'),
  watch: t('Дежурный'),
  chronicler: t('Летописец'),
  engineer: t('Инженер'),
};
const STATES = { done: t('сделано'), active: t('идёт'), next: t('впереди') };
const VERDICTS = { best: t('★ лучшая'), control: t('контроль'), dead_end: t('тупик'), active: t('идёт') };
const MARKS = { ok: '✓', fail: '✗', unknown: '·' };
const MARK_WORDS = { ok: t('выполнено'), fail: t('не выполнено'), unknown: t('не определено') };
const markOf = (item) => (item.ok ? 'ok' : item.ok === false ? 'fail' : 'unknown');

const VIEWS = ['entries', 'council', 'stages', 'check'];
const RECENT_ENTRIES = 12;
let journal = null,
  journalText = '',
  loadedAt = 0,
  lab = null, // /api/lab, live only: which branch is training now
  allEntries = false,
  races = []; // race specs behind the live 3D links of the view shown
// The reader's own opening and closing of folded blocks, kept across the live reloads.
const folds = new Map();

function status(message, error = false) {
  $('journal-status').textContent = message;
  $('journal-status').classList.toggle('error', error);
}

// #entries[/<entry id>], #council[/<iteration id>], #stages[/<stage n>[/<step n>]], #check.
function route() {
  let parts = [];
  try {
    parts = decodeURIComponent(location.hash.slice(1)).split('/');
  } catch {}
  const [view, ...rest] = parts;
  return { view: VIEWS.includes(view) ? view : 'entries', rest };
}

// --- Pieces shared by the views ---------------------------------------------------------------------------------

const fold = (key, summary, body, open = false, cls = '') =>
  `<details class="j-fold ${cls}" data-fold="${esc(key)}"${(folds.get(key) ?? open) ? ' open' : ''}><summary>${summary}</summary><div class="j-body">${body}</div></details>`;

// A pre-rendered clip (humanoid_sim.clip): its poster until pressed, then the looping animation; pressing again
// stops on the poster. The clip file is fetched only when played. A still picture in the clip slot (.png/.jpg)
// is shown as it is.
function clipMarkup(item, title) {
  if (!item.clip) return '';
  if (/\.(png|jpe?g)$/i.test(item.clip) || !item.poster)
    return `<a class="j-still" href="${media(item.clip)}" target="_blank" rel="noopener" title="${t('Открыть картинку целиком')}"><img src="${media(item.poster || item.clip)}" alt="${esc(title)}" loading="lazy" /></a>`;
  return `<figure class="j-clip"><button type="button" class="j-play" aria-pressed="false" title="${t('Нажмите, чтобы смотреть или остановить')}" data-clip="${media(item.clip)}" data-poster="${media(item.poster)}"><img src="${media(item.poster)}" alt="${esc(title)}" loading="lazy" /><span class="j-play-label">${t('▶ Смотреть')}</span></button></figure>`;
}
const imageMarkup = (item) =>
  item.image
    ? `<a class="j-still" href="${media(item.image)}" target="_blank" rel="noopener" title="${t('Открыть картинку целиком')}"><img src="${media(item.image)}" alt="${esc(item.title)}" loading="lazy" /></a>`
    : '';
// An interactive figure (humanoid_sim/figures): its own page in a frame. The lab server sends it with
// "Content-Security-Policy: sandbox allow-scripts", so its script cannot reach the lab; a static host sends no
// such header, so there the frame itself is sandboxed.
function figureMarkup(item, title) {
  if (!item.figure) return '';
  const src = STATIC ? media(item.figure) : '/api/lab/journal/figure?name=' + encodeURIComponent(item.figure);
  return `<div class="j-figure"><iframe src="${src}" title="${t('{title}: схема', { title: esc(title) })}" loading="lazy"${STATIC ? ' sandbox="allow-scripts"' : ''}></iframe><a href="${src}" target="_blank" rel="noopener">${t('Открыть схему отдельно')}</a></div>`;
}
// A live 3D race of the entry's or step's lanes (same seed and command) with its scenario: only on the lab's own
// server. The link carries the lanes (race.js presets them from ?lanes=); the scenario is sent when it is opened.
function raceLink(race, label) {
  if (STATIC || !race?.lanes?.length) return '';
  races.push(race);
  const query = new URLSearchParams({
    robot: race.robot,
    task: race.task,
    lanes: race.lanes.map((lane) => `${lane.experiment}:${lane.stage}`).join(','),
  });
  return `<a class="button ghost j-small" href="race.html?${esc(query)}" target="_blank" rel="noopener" data-race="${races.length - 1}">${label}</a>`;
}
async function control(race, body) {
  const query = new URLSearchParams({ robot: race.robot, task: race.task }).toString();
  const response = await fetch(withLang('/api/race/control?' + query), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error((await response.json().catch(() => ({}))).error || response.status);
}
// The same commands as the lab's old journal tab: the lanes, then the scenario.
async function prepareRace(race) {
  try {
    await control(race, { command: 'lanes', lanes: race.lanes });
    if (race.scenario) await control(race, { command: 'scenario', value: race.scenario });
  } catch (error) {
    status(t('Не удалось открыть забег: {error}', { error: error.message }), true);
  }
}

// What is training now and whether the latest check is about it (live only; the lab tab showed the same).
const APPROACHES = {
  careful: t('Осторожное обучение'),
  explore: t('Больше поиска'),
  baseline: t('Исходные настройки'),
};
const displayName = (e) =>
  String(e.name || e.id)
    .replace(/^(baseline|careful|explore) · seed /, (all, preset) => `${APPROACHES[preset]} · ${t('вариант')} `)
    .replace(' · seed ', ` · ${t('вариант')} `);
function activeRun() {
  const experiments = lab?.experiments || [];
  return (
    experiments.find((e) => (lab.active_trainers || []).includes(e.id)) ||
    experiments.find((e) => ['training', 'starting', 'saving'].includes(e.status?.state))
  );
}
function nowMarkup() {
  if (STATIC || !lab) return '';
  const e = activeRun(),
    c = journal.check;
  const age = c ? Date.now() / 1000 - c.at : 0;
  const stale =
    e && c && c.branch !== e.id
      ? t('Это проверка прошлой ветки: идущая ещё не проверялась (проверка — каждые полчаса).')
      : e && c && age > 50 * 60
        ? t('Проверке уже {n} мин, а новая бывает каждые полчаса.', { n: fmt(age / 60) })
        : '';
  return (
    (e
      ? `<p>${t('Учится:')} <strong>${esc(displayName(e))}</strong> · ${t('{n} млн шагов', { n: fmt((e.status?.steps || 0) / 1e6, 1) })}</p>`
      : `<p>${t('Сейчас обучение не идёт.')}</p>`) + (stale ? `<p class="j-stale">${stale}</p>` : '')
  );
}
function renderNow() {
  const markup = journal ? nowMarkup() : '';
  document.querySelectorAll('[data-now]').forEach((box) => {
    box.innerHTML = markup;
    box.hidden = !markup;
  });
}
const ticks = (check) =>
  `<span class="j-ticks" aria-hidden="true">${check.items.map((i) => `<span class="${markOf(i)}">${MARKS[markOf(i)]}</span>`).join('')}</span>`;
const passed = (check) =>
  t('выполнено {done} из {total}', { done: check.items.filter((i) => i.ok).length, total: check.items.length });
const iterations = () => [...(journal.iterations || [])].sort((a, b) => b.at - a.at);

// --- Entries ----------------------------------------------------------------------------------------------------

function asideMarkup() {
  const c = journal.check,
    latest = iterations()[0];
  const check = c
    ? `<section class="card" aria-labelledby="aside-check"><p class="eyebrow">${t('Последняя проверка')}</p><p class="j-meta">${time(c.at)}</p><h2 class="j-title" id="aside-check"><a href="#check">${esc(c.title)}</a></h2><p>${ticks(c)} ${passed(c)}</p><div class="j-now" data-now hidden></div></section>`
    : `<section class="card"><p class="eyebrow">${t('Последняя проверка')}</p><p>${t('Проверок пока не было.')}</p><div class="j-now" data-now hidden></div></section>`;
  const council = latest
    ? `<section class="card" aria-labelledby="aside-council"><p class="eyebrow">${t('Совет лаборатории')}</p><p class="j-meta">${time(latest.at)}</p><h2 class="j-title" id="aside-council"><a href="#council/${esc(latest.id)}">${esc(latest.title)}</a></h2><p><b>${latest.decision ? t('Решение:') : t('Обсуждается:')}</b> ${esc(firstSentence(latest.decision || t('решения ещё нет'), 240))}</p></section>`
    : '';
  const plan = (journal.roadmap || []).length
    ? `<section class="card" aria-labelledby="aside-plan"><h2 class="eyebrow" id="aside-plan">${t('План')}</h2><ol class="j-plan">${journal.roadmap
        .map(
          (s, i) =>
            `<li><a href="#stages/${i + 1}">${i + 1}. ${esc(s.title)}</a><span class="j-tag ${esc(s.status)}">${STATES[s.status] || esc(s.status)}</span></li>`,
        )
        .join('')}</ol></section>`
    : '';
  return check + council + plan;
}
const entryMarkup = (x, focus) =>
  `<article class="card j-entry${x.id === focus ? ' j-focus' : ''}" id="entry-${esc(x.id)}"><p class="j-meta"><a href="#entries/${esc(x.id)}">${time(x.at)}</a></p><h2 class="j-title">${esc(x.title)}</h2><div class="j-text">${paragraphs(x.text)}</div>${clipMarkup(x, x.title)}${imageMarkup(x)}${figureMarkup(x, x.title)}${raceLink(x.race, t('Открыть в 3D'))}</article>`;
function renderEntries([focus]) {
  const entries = [...journal.entries].sort((a, b) => b.at - a.at);
  if (focus && entries.findIndex((x) => x.id === focus) >= RECENT_ENTRIES) allEntries = true;
  const shown = allEntries ? entries : entries.slice(0, RECENT_ENTRIES);
  const more =
    entries.length > RECENT_ENTRIES
      ? `<button type="button" class="button ghost j-more" data-action="more-entries">${allEntries ? t('Скрыть старые записи') : t('Ещё {n} старых записей', { n: entries.length - RECENT_ENTRIES })}</button>`
      : '';
  const list = shown.map((x) => entryMarkup(x, focus)).join('') || `<p class="muted">${t('Записей пока нет.')}</p>`;
  return `<div class="j-layout"><div class="j-aside">${asideMarkup()}</div><div class="j-list">${list}${more}</div></div>`;
}

// --- Lab council ------------------------------------------------------------------------------------------------

// Newest first and open; older iterations fold to the date, the title and the decision's first sentence.
function renderCouncil([focus]) {
  const items = iterations();
  if (!items.length) return `<p class="muted">${t('Итераций пока не было.')}</p>`;
  if (focus) folds.set('i:' + focus, true);
  const undecided = t('решения ещё нет');
  const said = (m) =>
    `<li><span class="j-role ${esc(m.role)}">${ROLES[m.role] || esc(m.role)}</span><p>${esc(m.text)}</p></li>`;
  const card = (it, i) =>
    `<article class="card j-iteration" id="iteration-${esc(it.id)}">${fold(
      'i:' + it.id,
      `<span class="j-meta">${time(it.at)}</span><h2 class="j-title">${esc(it.title)}</h2><span class="j-note">${esc(firstSentence(it.decision || undecided))}</span>`,
      `<p><b>${t('Вопрос:')}</b> ${esc(it.question)}</p><ul class="j-said">${it.messages.map(said).join('')}</ul>` +
        `<div class="j-decision"><b>${it.decision ? t('Решение:') : t('Обсуждается:')}</b> ${esc(it.decision || undecided)}</div>`,
      i === 0,
    )}</article>`;
  return `<div class="j-council">${items.map(card).join('')}</div>`;
}

// --- Stages -----------------------------------------------------------------------------------------------------

const stateTag = (s) => `<span class="j-tag ${esc(s.status)}">${STATES[s.status] || esc(s.status)}</span>`;
function renderStages([number, step]) {
  const stages = journal.roadmap || [];
  const index = Number(number) - 1;
  if (stages[index]) return storyMarkup(index, Number(step) || null);
  if (!stages.length) return `<p class="muted">${t('План ещё не записан.')}</p>`;
  return `<ol class="j-stages">${stages
    .map(
      (s, i) =>
        `<li class="card j-stage ${esc(s.status)}${s.status === 'active' ? ' accent' : ''}"><p class="eyebrow">${t('Этап {n}', { n: i + 1 })} · ${stateTag(s)}</p><h2 class="j-title">${esc(s.title)}</h2><div class="j-text">${paragraphs(s.detail)}</div><a class="button ghost j-small" href="#stages/${i + 1}">${t('Подробно об этапе')}</a></li>`,
    )
    .join('')}</ol>`;
}
// One stage told in detail: its sections, the path through branches (a strip of verdicts, then the steps with what
// was changed, what came out and the clip; the newest step open) and the before/after races.
function storyMarkup(index, focusStep) {
  const stage = journal.roadmap[index],
    story = stage.story,
    n = index + 1;
  const path = story?.path || [],
    comparisons = story?.races || [];
  if (focusStep) folds.set(`p:${index}:${focusStep - 1}`, true);
  const toc = [];
  const sections = (story?.sections || [])
    .map((s, i) => {
      toc.push([`story-section-${i}`, esc(s.title)]);
      return `<section class="card j-section" id="story-section-${i}"><h3>${esc(s.title)}</h3><div class="j-text">${paragraphs(s.text)}</div>${figureMarkup(s, s.title)}</section>`;
    })
    .join('');
  let steps = '';
  if (path.length) {
    toc.push(['story-path', `${t('Как проходило обучение')} · ${path.length}`]);
    const counts = {};
    for (const p of path) if (p.verdict in VERDICTS) counts[p.verdict] = (counts[p.verdict] || 0) + 1;
    const verdict = (p) => `v-${esc(p.verdict || 'none')}`;
    // The strip repeats the list below for the eye; keyboard and screen readers use the list.
    const strip = `<div class="j-strip" aria-hidden="true">${path
      .map(
        (p, i) =>
          `<button type="button" tabindex="-1" class="${verdict(p)}" data-step="${i + 1}" title="${esc(`${i + 1}. ${p.title}`)}"></button>`,
      )
      .join('')}</div>`;
    const legend = `<p class="j-legend">${Object.keys(VERDICTS)
      .filter((v) => counts[v])
      .map((v) => `<span class="v-${v}"><i></i>${VERDICTS[v]} · ${counts[v]}</span>`)
      .join('')}</p>`;
    const list = path
      .map(
        (p, i) =>
          `<li id="step-${i + 1}" class="${verdict(p)}">${fold(
            `p:${index}:${i}`,
            `<span class="j-step-n">${i + 1}</span><h4 class="j-step-title">${esc(p.title)}</h4>${p.verdict ? `<span class="j-verdict">${VERDICTS[p.verdict] || esc(p.verdict)}</span>` : ''}${p.clip ? `<span class="j-has-clip" role="img" aria-label="${t('Есть ролик')}" title="${t('Есть ролик')}">▶</span>` : ''}`,
            `<p><b>${t('Что меняли:')}</b> ${esc(p.change)}</p><p><b>${t('Что вышло:')}</b> ${esc(p.result)}</p>${clipMarkup(p, p.title)}${raceLink(p.race, t('Открыть в 3D (было → стало)'))}`,
            i === path.length - 1,
            'j-step',
          )}</li>`,
      )
      .join('');
    steps = `<section class="card j-section" id="story-path"><h3>${t('Как проходило обучение')} · ${path.length} ${tn(path.length, 'шаг|шага|шагов')}</h3>${strip}${legend}<p class="j-hint">${t('Шаги по порядку; нажмите на шаг, чтобы открыть «что меняли / что вышло» и ролик.')} <button type="button" class="j-link" data-step="${path.length}">${t('К последнему шагу ↓')}</button></p><ol class="j-steps">${list}</ol></section>`;
  }
  let compare = '';
  if (comparisons.length) {
    toc.push(['story-races', `${t('Сравнить: было → стало')} · ${comparisons.length}`]);
    compare = `<section class="card j-section" id="story-races"><h3>${t('Сравнить: было → стало')} · ${comparisons.length}</h3><div class="j-races">${comparisons
      .map(
        (r) =>
          `<article class="card j-race-card"><h4 class="j-step-title">${esc(r.title)}</h4>${r.text ? `<div class="j-text">${paragraphs(r.text)}</div>` : ''}${clipMarkup(r, r.title)}${raceLink(r, t('Открыть в 3D'))}</article>`,
      )
      .join('')}</div></section>`;
  }
  const contents = toc.length
    ? `<nav class="j-toc" aria-label="${t('Оглавление')}">${toc.map(([id, label]) => `<button type="button" data-jump="${id}">${label}</button>`).join('')}</nav>`
    : '';
  const body = story
    ? sections + steps + compare
    : `<p class="muted">${t('Подробный рассказ появится, когда этап начнётся.')}</p>`;
  return `<p class="j-back"><a href="#stages">← ${t('Этапы')}</a></p><article class="j-story"><header class="card j-story-head${stage.status === 'active' ? ' accent' : ''}"><p class="eyebrow">${t('Этап {n}', { n })} · ${stateTag(stage)}</p><h2>${esc(stage.title)}</h2><div class="j-text">${paragraphs(stage.detail)}</div>${contents}</header>${body}</article>`;
}

// --- The latest goal check --------------------------------------------------------------------------------------

function renderCheck() {
  const c = journal.check;
  if (!c) return `<p class="muted">${t('Проверок пока не было.')}</p><div class="j-now" data-now hidden></div>`;
  const rows = c.items
    .map((item) => {
      const mark = markOf(item);
      return `<tr class="${mark}"><td><span class="j-mark" aria-hidden="true">${MARKS[mark]}</span><span class="j-sr">${MARK_WORDS[mark]}: </span>${esc(item.label)}</td><td class="num" data-label="${t('Значение')}">${esc(item.value)}</td><td data-label="${t('Цель')}">${esc(item.target)}</td></tr>`;
    })
    .join('');
  return `<div class="card j-check"><p class="eyebrow">${t('Последняя проверка')}</p><p class="j-meta">${time(c.at)} · ${esc(c.branch)} · ${t('{n} млн шагов', { n: fmt(c.stage / 1e6, 1) })}</p><h2 class="j-title">${esc(c.title)}</h2><p class="j-score">${ticks(c)} ${passed(c)}</p><div class="j-now" data-now hidden></div><div class="scroll-x"><table class="data j-checks"><thead><tr><th scope="col">${t('Условие')}</th><th scope="col">${t('Значение')}</th><th scope="col">${t('Цель')}</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
}

// --- Showing a view ---------------------------------------------------------------------------------------------

const RENDER = { entries: renderEntries, council: renderCouncil, stages: renderStages, check: renderCheck };
const tabs = [...document.querySelectorAll('[role="tab"]')];

// A figure page reports its height (humanoid_sim/figures); only a number is taken from it. One that loaded while
// hidden measured nothing, so a shown view asks its figures again.
addEventListener('message', (event) => {
  const height = event.data && event.data.anatomyHeight;
  if (!Number.isFinite(height)) return;
  for (const frame of document.querySelectorAll('.j-figure iframe'))
    if (frame.contentWindow === event.source)
      frame.style.height = Math.min(1600, Math.max(200, Math.ceil(height))) + 'px';
});
const measureFigures = (root) =>
  root
    .querySelectorAll('.j-figure iframe')
    .forEach((frame) => frame.contentWindow?.postMessage({ anatomyMeasure: true }, '*'));

// Scrolls an element to just below the sticky site header (three rows tall on a phone). Smooth for jumps within
// a view; otherwise at once, and for a few seconds put back in place while lazy pictures above it load (their size
// is not known before) and push it down, unless the reader has scrolled away meanwhile.
function jumpTo(element, smooth = false) {
  if (!element) return;
  const place = () => {
    const covered = Math.max(0, document.querySelector('.site-header')?.getBoundingClientRect().bottom || 0);
    const top = element.getBoundingClientRect().top + scrollY - covered - 12;
    scrollTo({ top, behavior: smooth ? 'smooth' : 'instant' });
    return scrollY;
  };
  let placed = place();
  if (smooth) return;
  const until = Date.now() + 4000;
  for (const image of main.querySelectorAll('img'))
    if (!image.complete)
      image.addEventListener(
        'load',
        () => Date.now() < until && Math.abs(scrollY - placed) < 4 && (placed = place()),
        { once: true },
      );
}

// Draws the view the hash names; navigated: the reader went there (scroll to the entry, iteration or step it
// names, or back up to the tabs), not a live reload of the same view.
function show(navigated) {
  const { view, rest } = route();
  for (const tab of tabs) {
    const selected = tab.dataset.view === view;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  for (const name of VIEWS) {
    $('view-' + name).hidden = name !== view;
    if (name !== view) $('view-' + name).replaceChildren(); // stops its clips; drawn again when shown
  }
  if (!journal) return;
  const panel = $('view-' + view);
  races = [];
  panel.innerHTML = RENDER[view](rest);
  renderNow();
  measureFigures(panel);
  if (!navigated) return;
  const target =
    view === 'entries' && rest[0]
      ? $('entry-' + rest[0])
      : view === 'council' && rest[0]
        ? $('iteration-' + rest[0])
        : view === 'stages' && rest[1]
          ? $('step-' + rest[1])
          : null;
  if (target) jumpTo(target);
  else if ($('journal-tabs').getBoundingClientRect().top < 0) jumpTo($('journal-tabs'));
}
function selectTab(view, focus = false) {
  history.replaceState(null, '', '#' + view);
  show(true);
  if (focus) $('tab-' + view).focus();
}
tabs.forEach((tab, i) => {
  tab.onclick = () => selectTab(tab.dataset.view);
  tab.onkeydown = (event) => {
    const next = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    selectTab(tabs[(next + tabs.length) % tabs.length].dataset.view, true);
  };
});
addEventListener('hashchange', () => show(true));

// Opens one step of the story shown, from the strip or «to the last step».
function openStep(number) {
  const item = $('step-' + number),
    details = item?.querySelector('details');
  if (!details) return;
  details.open = true;
  folds.set(details.dataset.fold, true);
  history.replaceState(null, '', `#stages/${route().rest[0]}/${number}`);
  jumpTo(item, true);
  details.querySelector('summary').focus({ preventScroll: true });
}
main.addEventListener('click', (event) => {
  const summary = event.target.closest('details[data-fold] > summary');
  if (summary) folds.set(summary.parentElement.dataset.fold, !summary.parentElement.open);
  const play = event.target.closest('.j-play');
  if (play) {
    const playing = play.getAttribute('aria-pressed') !== 'true';
    play.setAttribute('aria-pressed', String(playing));
    play.querySelector('img').src = playing ? play.dataset.clip : play.dataset.poster;
  }
  const step = event.target.closest('[data-step]');
  if (step) openStep(Number(step.dataset.step));
  const jump = event.target.closest('[data-jump]');
  if (jump) jumpTo($(jump.dataset.jump), true);
  if (event.target.closest('[data-action="more-entries"]')) {
    allEntries = !allEntries;
    show(false);
    if (!allEntries) main.querySelector('.j-more')?.scrollIntoView({ block: 'center', behavior: 'instant' });
  }
  const race = event.target.closest('a[data-race]');
  if (race) prepareRace(races[Number(race.dataset.race)]);
});
// A race link opened with the middle button gets its lanes and scenario too.
main.addEventListener('auxclick', (event) => {
  const race = event.button === 1 && event.target.closest('a[data-race]');
  if (race) prepareRace(races[Number(race.dataset.race)]);
});

// --- Loading ----------------------------------------------------------------------------------------------------

function counts() {
  const c = journal.check;
  const values = {
    entries: journal.entries.length,
    council: (journal.iterations || []).length,
    stages: journal.roadmap.length,
    check: c ? `${c.items.filter((i) => i.ok).length}/${c.items.length}` : '',
  };
  document.querySelectorAll('[data-count]').forEach((box) => (box.textContent = values[box.dataset.count] || ''));
}
async function load() {
  loadedAt = Date.now();
  try {
    const [next, snapshot] = await Promise.all([
      data('journal'),
      STATIC ? null : data('lab').catch(() => null),
    ]);
    lab = snapshot;
    const text = JSON.stringify(next);
    status('');
    if (text === journalText) return renderNow();
    const first = !journal;
    journalText = text;
    journal = { roadmap: [], check: null, entries: [], iterations: [], ...next };
    counts();
    show(first);
  } catch (error) {
    status(t('Не удалось загрузить журнал: {error}', { error: error.message }), true);
  }
}
// A link to one entry, iteration or step scrolls there itself, so the browser must not restore an old position
// over it after a reload.
if (route().rest.length && route().view !== 'check') history.scrollRestoration = 'manual';
show(false);
await load();
// Live: the journal changes at the lab's decisions and the training goes on; refresh while the page is shown.
if (!STATIC) {
  setInterval(() => document.visibilityState === 'visible' && load(), 30000);
  document.addEventListener(
    'visibilitychange',
    () => document.visibilityState === 'visible' && Date.now() - loadedAt > 30000 && load(),
  );
}
