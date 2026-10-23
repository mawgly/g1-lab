import { THREE, createViewer, addGeometry, applyGeoms } from './viewer3d.js';
import { t, locale, withLang, translatePage } from './i18n.js';

const $ = (id) => document.getElementById(id);
translatePage();
const viewParams = new URLSearchParams(location.search);
const apiQuery = new URLSearchParams({
  experiment: viewParams.get('experiment') || 'baseline',
  channel: viewParams.get('channel') || 'main',
}).toString();
const number = (value, digits = 0) =>
  Number(value).toLocaleString(locale, { maximumFractionDigits: digits, minimumFractionDigits: digits });
const compact = (value) =>
  value >= 1e6
    ? t('{n} млн', { n: number(value / 1e6, 2) })
    : value >= 1000
      ? t('{n} тыс.', { n: number(value / 1000, 0) })
      : number(value);
let catalog = { stages: [], training: {} },
  frame = null,
  metric = 'reward',
  displayedStage = null;
let dragging = false,
  pendingStage = null;

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = false;
}
async function api(path, body) {
  const response = await fetch(
    withLang(path + (path.includes('?') ? '&' : '?') + apiQuery),
    body === undefined
      ? { cache: 'no-store' }
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        },
  );
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || t('Ошибка {status}', { status: response.status }));
  return result;
}
async function command(body) {
  if (body.command === 'latest') {
    pendingStage = null;
    dragging = false;
  }
  try {
    await api('/api/control', body);
    $('error').hidden = true;
  } catch (error) {
    pendingStage = null;
    dragging = false;
    showError(error.message);
  }
}
function selectStage(step) {
  pendingStage = step;
  command({ command: 'stage', step });
}
function moveStage(delta) {
  if (!frame || !catalog.stages.length) return;
  const index = catalog.stages.findIndex((stage) => stage.step === frame.stage);
  selectStage(catalog.stages[Math.max(0, Math.min(catalog.stages.length - 1, index + delta))].step);
}

const viewport = $('viewport');
let scene, camera, orbit, meshes = [];
let previousTorso = null;
function resetCamera() {
  const torso = frame?.torso || [0, 0, 1];
  const view = frame?.view || {},
    offset = view.camera_offset || [3.1, -4, 1.6];
  const height = view.target_height ?? 0.9;
  orbit.target.set(torso[0], torso[1], height);
  camera.position.set(torso[0] + offset[0], torso[1] + offset[1], height + offset[2]);
  camera.up.set(0, 0, 1);
  orbit.update();
}
async function createScene() {
  // The sun follows the robot so its shadow stays inside the shadow camera.
  ({ scene, camera, orbit } = createViewer(viewport, (sun) => {
    if (!frame) return;
    sun.position.set(frame.torso[0] + 3, frame.torso[1] - 4, 8);
    sun.target.position.set(frame.torso[0], frame.torso[1], 0);
  }));
  meshes = addGeometry(scene, await api('/api/model'));
  resetCamera();
}

function applyFrame(next) {
  frame = next;
  if (frame.error) showError(t('Симуляция остановлена: {error}', { error: frame.error }));
  applyGeoms(meshes, frame.geoms);
  if (previousTorso) {
    const delta = new THREE.Vector3(frame.torso[0] - previousTorso[0], frame.torso[1] - previousTorso[1], 0);
    camera.position.add(delta);
    orbit.target.add(delta);
  } else resetCamera();
  previousTorso = frame.torso;
  $('loading').hidden = true;
  $('tour-start').textContent = frame?.tour ? t('↺ С начала обучения') : t('▶ Все этапы подряд');
  $('tour-status').textContent = frame?.tour
    ? t('Осталось {n} с', { n: number(frame.stage_remaining, 1) }) +
      (frame.ended ? ' · ' + t('попытка завершена') : '')
    : '';
  $('episode-time').textContent = number(frame.episode_seconds, 1);
  const value = frame[frame.view?.metric_key || 'forward_m'];
  $('episode-distance').textContent =
    (frame.view?.metric_label || '') + (value == null ? '—' : number(value, 2));
  // Speed is shown for robots the camera follows (floating base); averaged over the last 2 s.
  $('speed-box').hidden = frame.height_m == null;
  $('episode-speed').textContent = frame.speed_mps == null ? '—' : number(frame.speed_mps, 2);
  $('speed-command').textContent =
    frame.command_mps == null ? '' : ' · ' + t('команда {n}', { n: number(frame.command_mps, 1) });
  $('attempt-label').textContent = frame.ended
    ? t('Попытка завершена')
    : t('Попытка {n}', { n: frame.episode });
  $('pause-icon').textContent = frame.paused ? '▶' : 'Ⅱ';
  $('pause-label').textContent = frame.paused ? t('Играть') : t('Пауза');
  $('pause').setAttribute('aria-label', frame.paused ? t('Продолжить просмотр') : t('Пауза'));
  $('follow').checked = frame.follow;
  $('view-mode').textContent = frame.paused
    ? t('На паузе')
    : frame.follow
      ? t('Последний этап')
      : t('Выбранный этап');
  document
    .querySelectorAll('[data-speed]')
    .forEach((button) => button.classList.toggle('selected', Number(button.dataset.speed) === frame.speed));
  if (pendingStage === frame.stage) {
    pendingStage = null;
    dragging = false;
  }
  if (displayedStage !== frame.stage) {
    displayedStage = frame.stage;
    updateStage();
    drawChart();
  }
}
function updateStage() {
  const stages = catalog.stages;
  $('stage-step').textContent = frame?.stage == null ? t('Нет обученной политики') : number(frame.stage);
  $('stage-caption').textContent =
    frame?.stage === 0 ? t('исходная модель · без обучения') : t('шагов обучения');
  const index = stages.findIndex((stage) => stage.step === frame?.stage);
  $('timeline').max = Math.max(0, stages.length - 1);
  if (!dragging && pendingStage === null) $('timeline').value = Math.max(0, index);
  $('timeline').style.setProperty(
    '--value',
    `${(100 * Math.max(0, index)) / Math.max(1, stages.length - 1)}%`,
  );
  $('timeline').disabled = stages.length < 2;
  $('previous').disabled = index <= 0;
  $('next').disabled = index >= stages.length - 1;
  $('first-stage').disabled = !stages.length;
  $('latest-stage').disabled = !stages.length;
  $('stage-count').textContent = t('{n} этапов', { n: stages.length });
  $('timeline-position').textContent = index >= 0 ? `${index + 1} / ${stages.length}` : '—';
  $('timeline-end').textContent = stages.length ? compact(stages.at(-1).step) : '—';
  const current = stages[index],
    baseline = stages[0];
  $('metric-reward').textContent = current?.reward != null ? number(current.reward) : '—';
  $('metric-seconds').textContent =
    current?.seconds != null ? t('{n} с', { n: number(current.seconds, 1) }) : '—';
  $('metric-distance').textContent =
    current?.distance != null ? t('{n} м', { n: number(current.distance, 2) }) : '—';
  $('insight').textContent =
    current?.seconds != null && baseline?.seconds > 0
      ? current.step === 0
        ? t('Начальная точка эксперимента. Эта модель ещё не проходила обучение.')
        : t('Средняя попытка: {now} с. До обучения: {before} с. Оцени также направление и сами движения.', {
            now: number(current.seconds, 1),
            before: number(baseline.seconds, 1),
          })
      : t('Этот этап ещё не проверен. Показатели появятся после оценки модели.');
}
function drawChart() {
  const svg = $('chart'),
    rows = catalog.stages.filter((stage) => Number.isFinite(stage[metric]));
  $('chart-empty').hidden = rows.length > 0;
  if (!rows.length) {
    svg.innerHTML = '';
    return;
  }
  const width = 800,
    height = 180,
    left = 48,
    right = 14,
    top = 13,
    bottom = 28;
  const min = Math.min(0, ...rows.map((row) => row[metric])),
    max = Math.max(1, ...rows.map((row) => row[metric]));
  const span = max - min,
    maxStep = Math.max(1, catalog.stages.at(-1).step);
  const x = (step) => left + (step / maxStep) * (width - left - right);
  const y = (val) => top + ((max - val) / span) * (height - top - bottom);
  let content =
    '<defs><linearGradient id="area" x1="0" x2="0" y1="0" y2="1"><stop stop-color="#b8e9bb" stop-opacity=".14"/><stop offset="1" stop-color="#b8e9bb" stop-opacity="0"/></linearGradient></defs>';
  for (let i = 0; i <= 3; i++) {
    const val = min + (i / 3) * span,
      yy = y(val);
    content += `<line x1="${left}" y1="${yy}" x2="${width - right}" y2="${yy}" stroke="#344047" stroke-dasharray="3 5" opacity=".7"/><text x="${left - 9}" y="${yy + 3}" text-anchor="end" fill="#7e939c" font-size="9">${number(val, metric === 'reward' ? 0 : 1)}</text>`;
  }
  const path = rows.map((row, i) => `${i ? 'L' : 'M'}${x(row.step)},${y(row[metric])}`).join(' ');
  content += `<path d="${path} L${x(rows.at(-1).step)},${height - bottom} L${x(rows[0].step)},${height - bottom} Z" fill="url(#area)"/><path d="${path}" stroke="#b8e9bb" stroke-width="2" fill="none" stroke-linejoin="round"/>`;
  for (const row of rows) {
    const selected = row.step === frame?.stage;
    if (selected)
      content += `<line x1="${x(row.step)}" y1="${top}" x2="${x(row.step)}" y2="${height - bottom}" stroke="#b8e9bb" opacity=".3"/>`;
    content += `<g class="chart-point" data-step="${row.step}" tabindex="0" role="button" aria-label="${t('Этап {n}', { n: row.step })}"><circle cx="${x(row.step)}" cy="${y(row[metric])}" r="9" fill="transparent"/><circle cx="${x(row.step)}" cy="${y(row[metric])}" r="${selected ? 4 : 2.5}" fill="${selected ? '#dcf6d6' : '#9fcba7'}" stroke="#1b2624" stroke-width="1.5"/><title>${t('{n} шагов', { n: number(row.step) })} · ${number(row[metric], 2)}</title></g>`;
  }
  for (const ratio of [0, 0.25, 0.5, 0.75, 1])
    content += `<text x="${x(ratio * maxStep)}" y="${height - 6}" fill="#728993" font-size="9" text-anchor="middle">${compact(ratio * maxStep)}</text>`;
  svg.innerHTML = content;
}
async function refreshCatalog() {
  try {
    catalog = await api('/api/state');
    // Training is controlled from the lab page; the player only reports a trainer error.
    if (catalog.training.error) showError(t('Ошибка обучения: {error}', { error: catalog.training.error }));
    updateStage();
    drawChart();
  } catch (error) {
    showError(t('Нет связи с локальным сервером. Открой humanoid_browser.cmd.') + ' ' + error.message);
  }
}

$('pause').onclick = () => frame && command({ command: 'pause', value: !frame.paused });
$('reset').onclick = () => command({ command: 'reset' });
$('follow').onchange = (event) =>
  event.target.checked ? command({ command: 'latest' }) : frame?.stage != null && selectStage(frame.stage);
$('first-stage').onclick = () => catalog.stages.length && selectStage(catalog.stages[0].step);
$('latest-stage').onclick = () => command({ command: 'latest' });
$('previous').onclick = () => moveStage(-1);
$('next').onclick = () => moveStage(1);
$('timeline').oninput = () => {
  dragging = true;
  $('timeline-position').textContent = `${Number($('timeline').value) + 1} / ${catalog.stages.length}`;
};
$('timeline').onchange = () => selectStage(catalog.stages[Number($('timeline').value)].step);
$('camera-reset').onclick = resetCamera;
$('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.querySelector('.main-column').requestFullscreen();
  } catch (error) {
    showError(error.message);
  }
};
document
  .querySelectorAll('[data-speed]')
  .forEach(
    (button) => (button.onclick = () => command({ command: 'speed', value: Number(button.dataset.speed) })),
  );
document.querySelectorAll('[data-metric]').forEach(
  (button) =>
    (button.onclick = () => {
      metric = button.dataset.metric;
      document
        .querySelectorAll('[data-metric]')
        .forEach((other) => other.classList.toggle('selected', other === button));
      drawChart();
    }),
);
$('chart').onclick = (event) => {
  const point = event.target.closest('[data-step]');
  if (point) selectStage(Number(point.dataset.step));
};
$('chart').onkeydown = (event) => {
  if (event.key === 'Enter') {
    const point = event.target.closest('[data-step]');
    if (point) selectStage(Number(point.dataset.step));
  }
};
document.addEventListener('keydown', (event) => {
  if (['INPUT', 'BUTTON', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName) || !frame) return;
  if (event.code === 'Space') {
    event.preventDefault();
    $('pause').click();
  } else if (event.code === 'ArrowLeft') {
    event.preventDefault();
    moveStage(-1);
  } else if (event.code === 'ArrowRight') {
    event.preventDefault();
    moveStage(1);
  }
});
$('tour-start').onclick = () => {
  if ($('stage-duration').reportValidity()) {
    pendingStage = null;
    command({
      command: 'tour',
      seconds: Number($('stage-duration').value),
      mode: $('tour-mode').value,
      full_fall: $('full-fall').checked,
    });
  }
};
try {
  await createScene();
  await refreshCatalog();
  if (viewParams.has('live')) {
    await command({ command: 'latest' });
    await command({ command: 'pause', value: false });
  }
  if (viewParams.has('stage')) await command({ command: 'stage', step: Number(viewParams.get('stage')) });
  const stream = new EventSource(withLang('/api/stream?' + apiQuery));
  stream.onopen = () => {
    $('connection-text').textContent = t('Локальное подключение');
    $('connection-dot').classList.remove('off');
  };
  stream.onmessage = (event) => applyFrame(JSON.parse(event.data));
  stream.onerror = () => {
    $('connection-text').textContent = t('Переподключение…');
    $('connection-dot').classList.add('off');
  };
  setInterval(refreshCatalog, 2500);
} catch (error) {
  $('loading').hidden = true;
  showError(
    t('Не удалось открыть 3D-просмотр: {error}. Проверь, что WebGL включён в браузере.', {
      error: error.message,
    }),
  );
}
