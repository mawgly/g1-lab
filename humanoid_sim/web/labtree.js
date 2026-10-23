// «Лаборатория» (lab.html): the research tree, read-only. One robot's branches by their parent links (source_id),
// with the verdicts and notes of notes.json as the /api/lab snapshot gives them, stage counts and steps, and the
// running branch's progress. Live (humanoid_sim.webapp) the page refreshes every 5 s and links each branch to its
// report and to the live 3D race; the static copy (site_export, data/lab.<lang>.json) is read once and has neither.
// Nothing here trains: the lab's agents and scripts call /api/lab/… themselves.
import { t, tn, locale, withLang } from './i18n.js';
import { STATIC, data, frame } from './site.js';

frame('lab.html');
const $ = (id) => document.getElementById(id);
const fmt = (v, digits = 0) =>
  v == null ? '—' : Number(v).toLocaleString(locale, { maximumFractionDigits: digits });
const esc = (v) =>
  String(v ?? '').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );
const when = (at) =>
  new Date(at * 1000).toLocaleString(locale, { day: 'numeric', month: 'long', hour: '2-digit', minute: '2-digit' });
const states = {
  training: t('Обучается'),
  saving: t('Сохранение'),
  starting: t('Запускается'),
};
const verdicts = {
  best: [t('★ лучшая'), 'best'],
  active: [t('в работе'), 'active'],
  control: [t('контроль'), 'control'],
  dead_end: [t('тупик'), 'dead'],
};
const approaches = {
  careful: t('Осторожное обучение'),
  explore: t('Больше поиска'),
  baseline: t('Исходные настройки'),
};
// Series runs are named «careful · seed 42»: the preset and the seed in words.
const displayName = (e) =>
  String(e.name ?? e.id)
    .replace(/^(baseline|careful|explore) · seed /, (all, preset) => `${approaches[preset]} · ${t('вариант')} `)
    .replace(' · seed ', ` · ${t('вариант')} `);

const params = new URLSearchParams(location.search);
const recall = (key, fallback) => {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
};
const remember = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {}
};
let lab = null, // the /api/lab snapshot (or its static copy)
  robots = [], // /api/lab/robots: labels, and which tasks have the race view
  robot = null,
  serverStarted = null,
  found = null; // the branch the address points at (#branch-<id>)
// A line (a root and its descendants) opens and closes on the reader's own clicks, remembered per browser;
// otherwise the line being trained is open (or the newest one when nothing trains).
const lineOpen = recall('lab-tree-lines', {});
const notesOpen = new Set();

const robotOf = (e) => e.config?.robot_id || 'humanoid';
const taskOf = (e) => e.config?.task_id || 'humanoid_v5';
const running = (e) =>
  (lab.active_trainers || []).includes(e.id) || ['training', 'saving', 'starting'].includes(e.status?.state);
const robotLabel = (id) => robots.find((r) => r.robot_id === id)?.label || id;
const catalogTask = (e) =>
  robots.find((r) => r.robot_id === robotOf(e))?.tasks?.find((task) => task.task_id === taskOf(e));

// Robots that have branches, in catalog order.
function robotIds() {
  const order = robots.map((r) => r.robot_id),
    rank = (id) => (order.includes(id) ? order.indexOf(id) : order.length);
  return [...new Set(lab.experiments.map(robotOf))].sort((a, b) => rank(a) - rank(b));
}
// The address's robot, else the one training now, else the one last looked at, else the newest branch's.
function chooseRobot() {
  const ids = robotIds(),
    active = lab.experiments.find(running),
    newest = [...lab.experiments].sort((a, b) => (b.status?.updated_at || 0) - (a.status?.updated_at || 0))[0];
  let saved = null;
  try {
    saved = localStorage.getItem('lab-view-robot');
  } catch {}
  const candidates = [params.get('robot'), active && robotOf(active), saved, newest && robotOf(newest)];
  return candidates.find((id) => ids.includes(id)) || ids[0] || null;
}
function setRobot(id) {
  robot = id;
  try {
    localStorage.setItem('lab-view-robot', id);
  } catch {}
  params.set('robot', id);
  history.replaceState(null, '', `${location.pathname}?${params}${location.hash}`);
  renderRobots();
  renderTree();
}

// The tree of one robot's branches: children by parent, roots newest line first (by the latest update inside).
function grow(branches) {
  const ids = new Set(branches.map((e) => e.id)),
    children = new Map(),
    latest = new Map();
  for (const e of branches) {
    const parent = ids.has(e.config?.source_id) ? e.config.source_id : null;
    children.set(parent, [...(children.get(parent) || []), e]);
  }
  const visit = (e) => {
    const value = Math.max(e.status?.updated_at || e.created_at || 0, ...(children.get(e.id) || []).map(visit));
    latest.set(e.id, value);
    return value;
  };
  const roots = children.get(null) || [];
  roots.forEach(visit);
  roots.sort((a, b) => latest.get(b.id) - latest.get(a.id));
  const below = (e) => (children.get(e.id) || []).flatMap((child) => [child, ...below(child)]);
  return { ids, children, latest, roots, below };
}
const rootOf = (e, branches) => {
  const byId = new Map(branches.map((x) => [x.id, x]));
  while (byId.has(e.config?.source_id)) e = byId.get(e.config.source_id);
  return e;
};

// «Открыть в 3D» shows this branch alone: one lane, its newest stage, following new checkpoints (?lanes= replaces
// whatever was racing — the user asked not to see other stages there). The arm has no race and opens in the
// single player.
function viewLink(e) {
  if (catalogTask(e)?.race ?? true) {
    const query = new URLSearchParams({ robot: robotOf(e), task: taskOf(e), lanes: `${e.id}:latest` });
    return `race.html?${query}`;
  }
  return `index.html?experiment=${encodeURIComponent(e.id)}${running(e) ? '&live=1' : ''}`;
}
// A note shows its first sentence; the rest opens under it.
function noteMarkup(e, text) {
  text = String(text || '').trim();
  if (!text) return '';
  const sentence = text.split(/(?<=[.!?])\s/)[0],
    long = sentence.length > 180,
    summary = long ? sentence.slice(0, 179) + '…' : sentence,
    rest = long ? text : text.slice(sentence.length).trim();
  if (!rest) return `<p class="branch-note">${esc(text)}</p>`;
  const paragraphs = rest
    .split(/\n\s*\n/)
    .map((p) => `<p>${esc(p)}</p>`)
    .join('');
  return `<details class="branch-note" data-note="${esc(e.id)}"${notesOpen.has(e.id) ? ' open' : ''}><summary data-key="note:${esc(e.id)}">${esc(summary)}</summary>${paragraphs}</details>`;
}
function card(e, tree) {
  const note = lab.notes?.[e.id] || {},
    [verdict, kind] = verdicts[note.verdict] || ['', ''],
    live = running(e),
    stages = e.stage_count || 0,
    // A running branch shows its last saved stage here; the live step count is in «Сейчас».
    steps = live ? (e.last_stage ?? e.status?.steps ?? 0) : e.status?.steps || 0,
    warm = e.warm_start;
  const meta = [t('{n} млн шагов', { n: fmt(steps / 1e6, 1) })];
  if (tree.ids.has(e.config?.source_id) && e.config.source_stage != null)
    meta.push(t('от {n} млн', { n: fmt(e.config.source_stage / 1e6, 1) }));
  else if (warm?.source_id) {
    const parent = lab.experiments.find((x) => x.id === warm.source_id),
      name = esc(parent ? displayName(parent) : warm.source_id);
    meta.push(
      t('перенос с «{name}», {n} млн', {
        name: tree.ids.has(warm.source_id)
          ? `<a href="#branch-${esc(warm.source_id)}" data-key="warm:${esc(e.id)}">${name}</a>`
          : name,
        n: fmt((warm.source_stage || 0) / 1e6, 1),
      }),
    );
  }
  meta.push(stages ? `${fmt(stages)} ${tn(stages, 'этап|этапа|этапов')}` : t('нет этапов'));
  meta.push(esc(e.id));
  const links = [];
  if (!STATIC && e.has_report)
    links.push(
      `<a href="${esc(withLang('report.html?experiment=' + encodeURIComponent(e.id)))}" data-key="report:${esc(e.id)}">${t('Отчёт')}</a>`,
    );
  if (!STATIC && stages)
    links.push(
      `<a href="${esc(viewLink(e))}" data-key="view:${esc(e.id)}" title="${esc(t('Только эта ветка: её последний этап, следить за новыми'))}">${t('Открыть в 3D')}</a>`,
    );
  return `<article class="branch ${kind}${live ? ' running' : ''}" tabindex="-1" aria-labelledby="name-${esc(e.id)}">
<div class="branch-head"><span class="branch-name" id="name-${esc(e.id)}">${esc(displayName(e))}</span>${verdict ? `<span class="tag ${kind}">${verdict}</span>` : ''}${live ? `<span class="tag running">${states[e.status?.state] || t('Обучается')}</span>` : ''}</div>
<p class="branch-meta">${meta.map((item) => `<span>${item}</span>`).join('')}</p>
${noteMarkup(e, note.note)}${links.length ? `<p class="branch-links">${links.join('')}</p>` : ''}</article>`;
}
// A line runs down from its root; at a fork it continues with the child where the newest work is, and the
// other children hang off the fork as indented side lines.
function chain(e, tree) {
  let markup = '';
  for (let node = e; node; ) {
    const kids = tree.children.get(node.id) || [],
      next = kids.reduce((a, b) => (!a || tree.latest.get(b.id) > tree.latest.get(a.id) ? b : a), null),
      kind = (verdicts[lab.notes?.[node.id]?.verdict] || [])[1] || '';
    markup += `<li id="branch-${esc(node.id)}" class="${kind}${running(node) ? ' running' : ''}${node.id === found ? ' found' : ''}">${card(node, tree)}${kids
      .filter((kid) => kid !== next)
      .map((side) => `<ol class="chain side">${chain(side, tree)}</ol>`)
      .join('')}</li>`;
    node = next;
  }
  return markup;
}
function lineMarkup(root, tree, fallbackOpen) {
  const members = [root, ...tree.below(root)],
    live = members.some(running),
    open = lineOpen[root.id] ?? (live || fallbackOpen || members.some((e) => e.id === found)),
    count = members.length;
  return `<details class="line" data-line="${esc(root.id)}"${open ? ' open' : ''}><summary data-key="line:${esc(root.id)}"><h3>${esc(displayName(root))}</h3><span class="line-meta"><span>${fmt(count)} ${tn(count, 'ветка|ветки|веток')}</span>${live ? `<span class="tag running">${t('Обучается')}</span>` : ''}</span></summary><div class="line-body"><ol class="chain">${chain(root, tree)}</ol></div></details>`;
}

// Lines are replaced only when their markup changed, and the focused link or summary keeps the focus.
let shown = new Map();
function renderTree() {
  const box = $('tree'),
    branches = lab.experiments.filter((e) => robotOf(e) === robot);
  if (!branches.length) {
    box.innerHTML = `<p class="muted">${t('Веток пока нет.')}</p>`;
    shown = new Map();
    return;
  }
  const tree = grow(branches),
    anyRunning = branches.some(running),
    lines = tree.roots.map((root, i) => [root.id, lineMarkup(root, tree, !anyRunning && i === 0)]);
  const focused = box.contains(document.activeElement)
    ? document.activeElement.closest('[data-key]')?.dataset.key
    : null;
  if (lines.map(([id]) => id).join() !== [...shown.keys()].join())
    box.innerHTML = lines.map(([, markup]) => markup).join('');
  else
    for (const [id, markup] of lines)
      if (shown.get(id) !== markup) box.querySelector(`[data-line="${CSS.escape(id)}"]`).outerHTML = markup;
  shown = new Map(lines);
  if (focused) box.querySelector(`[data-key="${CSS.escape(focused)}"]`)?.focus();
}
$('tree').addEventListener('click', (event) => {
  const summary = event.target.closest('summary');
  if (!summary) return;
  const details = summary.parentElement,
    opening = !details.open;
  if (details.dataset.line) {
    lineOpen[details.dataset.line] = opening;
    remember('lab-tree-lines', lineOpen);
  } else if (details.dataset.note) opening ? notesOpen.add(details.dataset.note) : notesOpen.delete(details.dataset.note);
});
function foldAll(open) {
  if (!lab) return;
  for (const root of grow(lab.experiments.filter((e) => robotOf(e) === robot)).roots) lineOpen[root.id] = open;
  remember('lab-tree-lines', lineOpen);
  renderTree();
}
$('open-all').onclick = () => foldAll(true);
$('close-all').onclick = () => foldAll(false);

function renderRobots() {
  const box = $('robots'),
    ids = robotIds(),
    key = ids.map((id) => `${id}=${robotLabel(id)}`).join('|');
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    box.innerHTML = ids
      .map(
        (id) =>
          `<button type="button" class="chip" data-robot="${esc(id)}" aria-pressed="false">${esc(robotLabel(id))}</button>`,
      )
      .join('');
  }
  for (const button of box.querySelectorAll('[data-robot]'))
    button.setAttribute('aria-pressed', String(button.dataset.robot === robot));
}
$('robots').onclick = (event) => {
  const button = event.target.closest('[data-robot]');
  if (button && button.dataset.robot !== robot) setRobot(button.dataset.robot);
};

// «Сейчас»: the running branch's progress, as the trainer's status.json has it. Built once per running set and
// then updated in place, so a focused link survives the refreshes.
function progress(e) {
  const s = e.status || {},
    c = e.config || {},
    budget = Number(c.steps || 0),
    // The trainer's target is its starting step + budget; a warm-started root starts far from 0 with no source.
    start = s.target_steps != null ? s.target_steps - budget : Number(c.source_stage || 0),
    done = Math.max(0, Math.min(budget, (s.steps || start) - start)),
    rate = s.steps_per_second || 0,
    // A timed run ends at whichever comes first: the step budget or the timer.
    timer = s.remaining_seconds != null ? s.remaining_seconds / 60 : Infinity,
    minutes = rate ? Math.ceil(Math.min(Math.max(0, budget - done) / rate / 60, timer)) : null,
    stale = !STATIC && Date.now() / 1000 - (s.updated_at || 0) > 20,
    toStage =
      Math.min(s.target_steps || Infinity, (e.last_stage ?? c.source_stage ?? 0) + (c.every || Infinity)) -
      (s.steps || 0),
    seconds = rate && Number.isFinite(toStage) ? Math.ceil(Math.max(0, toStage) / rate) : null;
  return {
    pct: budget ? (100 * done) / budget : 0,
    text: {
      state: stale ? t('Ждём обновления') : states[s.state] || t('Обучается'),
      budget: t('+{done} / {budget} млн шагов', { done: fmt(done / 1e6, 2), budget: fmt(budget / 1e6, 0) }),
      rate: t('{n} шагов/с', { n: fmt(rate) }),
      left: minutes == null ? t('Оцениваем скорость') : t('примерно {n} мин до конца запуска', { n: fmt(minutes) }),
      next:
        s.state === 'saving' || seconds === 0
          ? t('Сохраняем новый этап')
          : seconds == null
            ? t('Ждём новый этап')
            : t('Следующий этап примерно через {n} с', { n: fmt(seconds) }),
      steps: t('{n} млн шагов', { n: fmt((s.steps || 0) / 1e6, 1) }),
    },
  };
}
function renderNow() {
  const box = $('now'),
    active = lab.experiments.filter(running),
    key = active.map((e) => `${e.id}:${e.name}`).join('|');
  if (box.dataset.key !== key) {
    box.dataset.key = key;
    // The static copy leaves out the time left and the next stage: they are stale as soon as it is exported.
    const fields = (STATIC ? ['budget', 'rate'] : ['budget', 'rate', 'left', 'next'])
      .map((field) => `<span data-field="${field}"></span>`)
      .join('');
    box.innerHTML =
      (active.length
        ? active
            .map(
              (e) =>
                `<article class="card accent now-card" data-run="${esc(e.id)}"><div class="branch-head"><span class="tag running" data-field="state"></span><span class="muted mono" data-field="steps"></span></div><h3><a href="#branch-${esc(e.id)}">${esc(displayName(e))}</a></h3><progress max="100" aria-label="${esc(t('Прогресс текущего обучения'))}"></progress><div class="now-stats">${fields}</div></article>`,
            )
            .join('')
        : `<p>${t('Сейчас обучение не идёт.')}</p>`) + '<p class="muted" data-field="stamp"></p>';
  }
  for (const e of active) {
    const facts = progress(e),
      item = box.querySelector(`[data-run="${CSS.escape(e.id)}"]`);
    item.querySelector('progress').value = facts.pct;
    for (const [field, text] of Object.entries(facts.text)) {
      const slot = item.querySelector(`[data-field="${field}"]`);
      if (slot) slot.textContent = text;
    }
  }
  // The static copy is as of its export; live, as of the newest status a trainer wrote.
  const latest = lab.exported_at || Math.max(0, ...lab.experiments.map((e) => e.status?.updated_at || 0));
  box.querySelector('[data-field="stamp"]').textContent = latest ? t('Данные на {time}', { time: when(latest) }) : '';
}

// #branch-<id> (the «Сейчас» card, a carried-over root's parent, a link from the journal): shows that branch's
// robot, opens its line and moves the focus there.
function reveal() {
  if (!lab || !location.hash.startsWith('#branch-')) return;
  const id = decodeURIComponent(location.hash.slice('#branch-'.length)),
    e = lab.experiments.find((x) => x.id === id);
  if (!e) return;
  found = id;
  lineOpen[rootOf(e, lab.experiments.filter((x) => robotOf(x) === robotOf(e))).id] = true;
  if (robotOf(e) !== robot) setRobot(robotOf(e));
  else renderTree();
  const target = document.getElementById('branch-' + id);
  target?.scrollIntoView({ block: 'start' });
  target?.querySelector('.branch')?.focus({ preventScroll: true });
}
addEventListener('hashchange', reveal);

// A restarted server (new code) makes the open page stale: it reloads itself when the start time changes.
async function refresh() {
  try {
    const next = await data('lab');
    if (!STATIC && serverStarted && next.server_started !== serverStarted) return location.reload();
    serverStarted = next.server_started;
    if (!robots.length && Array.isArray(next.robots)) robots = next.robots;
    lab = next;
    $('error').hidden = true;
    if (!robotIds().includes(robot)) robot = chooseRobot();
    renderRobots();
    renderNow();
    renderTree();
  } catch (error) {
    $('error').textContent = t('Не удалось загрузить данные лаборатории: {error}', { error: error.message });
    $('error').hidden = false;
    if (!lab) $('now').replaceChildren();
  }
}

$('static-note').hidden = !STATIC;
// The catalog (robot names, which tasks race) comes alongside: the server builds it on its first request. The
// static copy has it in its snapshot (`robots`) if the export put it there; otherwise robot ids stand for names.
const catalog = STATIC
  ? Promise.resolve()
  : data('robots').then(
      (value) => {
        if (Array.isArray(value)) robots = value;
      },
      () => {},
    );
await refresh();
await catalog;
if (lab) {
  renderRobots();
  renderTree();
}
reveal();
if (!STATIC) setInterval(refresh, 5000);
