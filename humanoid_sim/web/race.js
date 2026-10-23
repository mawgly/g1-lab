// Stage race: each lane replays one saved stage with the same seed and command; physics runs on the server.
import { THREE, createViewer, addGeometry } from './viewer3d.js';
import { t, locale, withLang, translatePage } from './i18n.js';

const $ = (id) => document.getElementById(id);
translatePage();
const params = new URLSearchParams(location.search);
// One race per robot (and task); lanes may come from any branch of that robot.
const query = new URLSearchParams(
  [...params].filter(([key]) => ['robot', 'task', 'focus'].includes(key)),
).toString();
const MAX_LANES = 10; // matches Race.MAX_LANES on the server
// Back to the lab's branch tree of the same robot.
if (params.get('robot')) $('back').href = 'lab.html?' + new URLSearchParams({ robot: params.get('robot') });
const COLORS = [
  '#9fe0a8',
  '#86c5e8',
  '#efb97d',
  '#d9a3ec',
  '#f08f86',
  '#e6dc78',
  '#7fd8cf',
  '#aab4f5',
  '#d1a77c',
  '#eef2ee',
];
const LANE_GAP = 2; // metres between lanes; only the drawing is shifted, each robot has its own floor
const number = (value, digits = 0) =>
  value == null
    ? '—'
    : Number(value).toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits });
const steps = (value) =>
  value >= 1e6
    ? t('{n} млн', { n: number(value / 1e6, 1) })
    : value >= 1e3
      ? t('{n} тыс.', { n: number(value / 1e3) })
      : number(value);

let scene, camera, orbit, geometry;
const geometryCache = new Map(); // all lanes show the same robot model
let frame = null,
  lanes = [],
  decorations = [],
  followedX = 0,
  tableKey = '';

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}
async function api(path, body) {
  const response = await fetch(
    withLang(path + '?' + query),
    body === undefined
      ? { cache: 'no-store' }
      : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) },
  );
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || response.status);
  return value;
}
function control(body) {
  api('/api/race/control', body)
    .then(() => showError(''))
    .catch((error) => showError(error.message));
}
const laneOffset = (index, count) => (index - (count - 1) / 2) * LANE_GAP;

// Distance marks beside the lanes, every 5 m (brighter every 10 m), created around the camera.
const marks = new Map();
function markMesh(distance) {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 96;
  const context = canvas.getContext('2d');
  context.fillStyle = distance % 10 ? 'rgba(232, 241, 238, 0.6)' : 'rgba(244, 248, 246, 0.95)';
  context.font = `600 ${distance % 10 ? 46 : 64}px system-ui, sans-serif`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(t('{n} м', { n: distance }), 128, 48);
  // A sprite always faces the camera, so the mark stays readable however the view is orbited.
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, depthWrite: false }),
  );
  sprite.scale.set(1.2, 0.45, 1);
  return sprite;
}
function dropMark(distance) {
  const sprite = marks.get(distance);
  scene.remove(sprite);
  sprite.material.map.dispose();
  sprite.material.dispose();
  marks.delete(distance);
}
function clearMarks() {
  for (const distance of [...marks.keys()]) dropMark(distance);
}
function updateMarks() {
  const from = Math.max(5, Math.floor((followedX - 20) / 5) * 5),
    to = Math.ceil((followedX + 45) / 5) * 5,
    side = -(lanes.length * LANE_GAP) / 2 - 0.6;
  for (const distance of [...marks.keys()]) if (distance < from || distance > to) dropMark(distance);
  for (let distance = from; distance <= to; distance += 5)
    if (!marks.has(distance)) {
      const mesh = markMesh(distance);
      mesh.position.set(distance, side, 0.25);
      scene.add(mesh);
      marks.set(distance, mesh);
    }
}

function buildLanes(count) {
  clearMarks(); // their side position depends on the number of lanes
  for (const object of [...lanes.flatMap((lane) => lane.meshes), ...decorations])
    if (object) scene.remove(object);
  decorations = [];
  lanes = [];
  for (let i = 0; i < count; i++) {
    lanes.push({
      meshes: addGeometry(scene, geometry, { floor: i === 0, tint: COLORS[i], cache: geometryCache }),
    });
    const stripe = new THREE.Mesh(
      new THREE.PlaneGeometry(2000, 0.06),
      new THREE.MeshBasicMaterial({ color: COLORS[i], transparent: true, opacity: 0.75, depthWrite: false }),
    );
    stripe.position.set(995, laneOffset(i, count), 0.009);
    decorations.push(stripe);
  }
  const start = new THREE.Mesh(
    new THREE.PlaneGeometry(0.06, count * LANE_GAP),
    new THREE.MeshBasicMaterial({ color: '#f4f7f2', transparent: true, opacity: 0.8, depthWrite: false }),
  );
  start.position.set(0, 0, 0.01);
  decorations.push(start);
  scene.add(...decorations);
  resetCamera();
}

// Behind, to the side and above: lanes separate across the screen instead of hiding each other.
// The distance grows with the number of lanes so all of them stay in view.
function resetCamera() {
  const width = lanes.length * LANE_GAP;
  followedX = leaderX();
  orbit.target.set(followedX + 1.5, 0, 0.5);
  camera.position.set(followedX - 3.2 - width / 2, -(2.2 + width / 2), 2.4 + width / 3);
  orbit.update();
  scene.fog.near = 10 + width;
  scene.fog.far = 34 + 2 * width;
}

// Evenly spaced stages of the last lane's branch; the last of them follows the newest stage.
const stagesOf = (experiment) => frame.experiments.find((e) => e.id === experiment)?.stages || [];
function spread(count) {
  const experiment = frame.lanes.at(-1).experiment,
    all = stagesOf(experiment);
  const fixed = Array.from(
    { length: count - 1 },
    (_, i) => all[Math.round((i * (all.length - 1)) / (count - 1))],
  );
  return [...fixed, 'latest'].map((stage) => ({ experiment, stage }));
}
const laneSpec = (lane) => ({ experiment: lane.experiment, stage: lane.follow ? 'latest' : lane.stage });
const short = (name) => (name.length > 26 ? name.slice(0, 25) + '…' : name);
function leaderX() {
  return frame?.lanes.length ? Math.max(...frame.lanes.map((lane) => lane.torso[0])) : 0;
}

// Smooth playback. Frames come at up to the 50 Hz physics rate but unevenly (browser and Windows timers),
// so drawing each one on arrival made fast running stutter. Poses are kept with their simulation time and
// drawn interpolated on a simulation clock running LAG behind the newest frame.
const LAG = 0.06; // s of simulation time
let poses = [], // [{ seconds, lanes: [{ geoms, torso }] }], oldest first
  clock = null; // simulation time = offset + wall seconds × speed
const quatA = new THREE.Quaternion(),
  quatB = new THREE.Quaternion();
const sameLanes = (a, b) =>
  a.lanes.length === b.lanes.length &&
  a.lanes.every((lane, i) => lane.experiment === b.lanes[i].experiment && lane.stage === b.lanes[i].stage);

function bufferPose(next, previous) {
  const continuous =
    previous &&
    next.round === previous.round &&
    sameLanes(next, previous) &&
    next.seconds >= (poses.at(-1)?.seconds ?? 0);
  if (!continuous) poses = [];
  const pose = {
    seconds: next.seconds,
    lanes: next.lanes.map((lane) => ({ geoms: lane.geoms, torso: lane.torso })),
  };
  if (poses.length && poses.at(-1).seconds === pose.seconds) poses[poses.length - 1] = pose;
  else poses.push(pose);
  if (poses.length > 8) poses.shift();
  const wall = performance.now() / 1000,
    speed = next.paused ? 0 : next.speed,
    offset = next.seconds - LAG - wall * speed;
  if (!continuous || !clock || clock.speed !== speed || Math.abs(offset - clock.offset) > 0.25)
    clock = { offset, speed };
  else clock.offset += (offset - clock.offset) * 0.1; // follow arrival jitter slowly
}

function blendGeoms(meshes, from, to, t, offsetY) {
  for (let i = 0; i < meshes.length; i++) {
    const a = from[i],
      b = to[i];
    if (!a || !b || !meshes[i] || meshes[i].userData.floor) continue; // the viewer moves the floor
    meshes[i].position.set(
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t + offsetY,
      a[2] + (b[2] - a[2]) * t,
    );
    meshes[i].quaternion.slerpQuaternions(quatA.fromArray(a, 3), quatB.fromArray(b, 3), t);
  }
}

// Draws the interpolated poses; returns the leading torso x for the camera (null before the first frame).
function drawPoses() {
  if (!poses.length || poses[0].lanes.length !== lanes.length) return null;
  const now = clock.offset + (performance.now() / 1000) * clock.speed;
  const next = poses.findIndex((pose) => pose.seconds > now);
  const a = next > 0 ? poses[next - 1] : next === 0 ? poses[0] : poses.at(-1),
    b = next > 0 ? poses[next] : a,
    t = b === a ? 0 : (now - a.seconds) / (b.seconds - a.seconds);
  a.lanes.forEach((lane, i) =>
    blendGeoms(lanes[i].meshes, lane.geoms, b.lanes[i].geoms, t, laneOffset(i, lanes.length)),
  );
  return Math.max(...a.lanes.map((lane, i) => lane.torso[0] + (b.lanes[i].torso[0] - lane.torso[0]) * t));
}

// Reward terms: fixed colours for known terms so a colour always means the same thing.
const REWARD_COLORS = {
  velocity: '#5fd18a',
  yaw: '#46b3e6',
  alive: '#a9e36b',
  gait: '#f2c94c',
  swing: '#ef8f5a',
  tilt: '#e06c75',
  torso: '#c678dd',
  base_motion: '#d19a66',
  still: '#56b6c2',
  stand_pose: '#4fa3a8',
  slip: '#9b8f6b',
  posture: '#b48ead',
  effort: '#8d9aa5',
  action_change: '#e5a07b',
  fall: '#ff5c5c',
  reward_forward: '#5fd18a',
  reward_survive: '#a9e36b',
  reward_ctrl: '#8d9aa5',
  reward_contact: '#d19a66',
};
const EXTRA_COLORS = ['#7fd8cf', '#aab4f5', '#e6dc78', '#d9a3ec'];
const rewardKeys = () => [...new Set(frame.lanes.flatMap((lane) => Object.keys(lane.rewards)))];
const rewardColor = (key, index) => REWARD_COLORS[key] || EXTRA_COLORS[index % EXTRA_COLORS.length];

function renderLegend(keys) {
  const used = keys.filter((key) => frame.lanes.some((lane) => Math.abs(lane.rewards[key] ?? 0) > 1e-4));
  $('reward-legend').innerHTML = used
    .map(
      (key) =>
        `<span><i style="background:${rewardColor(key, keys.indexOf(key))}"></i>${frame.reward_labels[key] || key}</span>`,
    )
    .join('');
}

// Diverging stacked bar on a scale shared by all lanes: penalties left of the zero mark, rewards right.
function stackMarkup(lane, keys, scale) {
  const span = scale.negative + scale.positive,
    zero = (100 * scale.negative) / span;
  let right = zero,
    left = zero,
    total = 0;
  const segments = keys.map((key, index) => {
    const value = lane.rewards[key] ?? 0,
      width = (100 * Math.abs(value)) / span;
    total += value;
    if (width < 0.05) return '';
    const start = value >= 0 ? right : (left -= width);
    if (value >= 0) right += width;
    const label = `${frame.reward_labels[key] || key}: ${number(value, Math.abs(value) >= 0.1 ? 2 : 3)}`;
    return `<i title="${label}" style="left:${start}%;width:${width}%;background:${rewardColor(key, index)}"></i>`;
  });
  return `<div class="stack-row"><div class="stack">${segments.join('')}<i class="zero" style="left:${zero}%"></i></div><b>${number(total, 2)}</b></div>`;
}

function renderTable() {
  const key = JSON.stringify([frame.lanes.map(laneSpec), frame.experiments]);
  if (key !== tableKey) {
    tableKey = key;
    const rows = frame.lanes.map((lane, i) => {
      const row = document.createElement('tr');
      const select = document.createElement('select');
      select.setAttribute('aria-label', t('Ветка и этап дорожки {n}', { n: i + 1 }));
      select.title = lane.name;
      // Grouped by branch: "следить" follows that branch's newest checkpoint.
      for (const branch of frame.experiments) {
        const group = document.createElement('optgroup');
        group.label = branch.name;
        const own = branch.id === lane.experiment;
        group.append(
          new Option(
            `${short(branch.name)} · ${t('следить ({stage})', { stage: steps(branch.stages.at(-1)) })}`,
            `${branch.id}|latest`,
            false,
            own && lane.follow,
          ),
          ...branch.stages.map(
            (step) =>
              new Option(
                `${short(branch.name)} · ${steps(step)}`,
                `${branch.id}|${step}`,
                false,
                own && !lane.follow && step === lane.stage,
              ),
          ),
        );
        select.append(group);
      }
      select.onchange = () => {
        const [experiment, stage] = select.value.split('|');
        const chosen = frame.lanes.map(laneSpec);
        chosen[i] = { experiment, stage: stage === 'latest' ? stage : Number(stage) };
        control({ command: 'lanes', lanes: chosen });
      };
      const remove = document.createElement('button');
      remove.textContent = '×';
      remove.title = t('Убрать дорожку');
      remove.hidden = frame.lanes.length < 2;
      remove.onclick = () =>
        control({ command: 'lanes', lanes: frame.lanes.map(laneSpec).filter((_, j) => j !== i) });
      row.innerHTML =
        `<td><span class="swatch" style="background:${COLORS[i]}"></span></td>` + '<td></td>'.repeat(7);
      row.children[1].append(select, ' ', remove);
      return row;
    });
    $('lanes').replaceChildren(...rows);
    const counts = Array.from({ length: MAX_LANES }, (_, i) => i + 1);
    $('lane-count').replaceChildren(
      ...counts.map((count) => new Option(String(count), count, false, count === frame.lanes.length)),
    );
  }
  const keys = rewardKeys();
  const scale = { positive: 1e-9, negative: 1e-9 };
  for (const lane of frame.lanes) {
    const values = keys.map((key) => lane.rewards[key] ?? 0);
    scale.positive = Math.max(
      scale.positive,
      values.filter((v) => v > 0).reduce((a, b) => a + b, 0),
    );
    scale.negative = Math.max(scale.negative, -values.filter((v) => v < 0).reduce((a, b) => a + b, 0));
  }
  renderLegend(keys);
  frame.lanes.forEach((lane, i) => {
    const cells = $('lanes').children[i].children;
    cells[2].textContent = number(lane.speed_mps, 2);
    cells[3].textContent = number(lane.distance_m, 2);
    cells[4].textContent = number(lane.sideways_m, 2);
    cells[5].textContent = number(lane.steps_per_min);
    cells[6].textContent =
      lane.fell_at != null
        ? t('Упал на {n} с', { n: number(lane.fell_at, 1) })
        : lane.finished
          ? t('Дошёл до конца')
          : t('Идёт');
    cells[7].innerHTML = keys.length ? stackMarkup(lane, keys, scale) : '—';
    $('lanes').children[i].classList.toggle('done', lane.fell_at != null || lane.finished);
  });
}

// Rows: reward components in task order; columns: lanes. Bars share one scale so lanes compare.
function renderRewards() {
  const keys = [...new Set(frame.lanes.flatMap((lane) => Object.keys(lane.rewards)))];
  if (!keys.length) {
    $('rewards').innerHTML = `<tr><td class="hint">${t('Появится после первых шагов забега.')}</td></tr>`;
    return;
  }
  const value = (lane, key) => lane.rewards[key] ?? 0;
  const scale = Math.max(
    1e-9,
    ...frame.lanes.flatMap((lane) => keys.map((key) => Math.abs(value(lane, key)))),
  );
  const cell = (v, bar = true) => {
    const width = bar ? Math.min(100, (100 * Math.abs(v)) / scale) : 0;
    const digits = Math.abs(v) >= 0.1 || v === 0 ? 2 : 3;
    return `<td><div class="cell"><span class="bar ${v < 0 ? 'neg' : 'pos'}" style="width:${width}%"></span><b>${number(v, digits)}</b></div></td>`;
  };
  const head = frame.lanes
    .map(
      (lane, i) =>
        `<th title="${lane.name}"><span class="swatch" style="background:${COLORS[i]}"></span> ${steps(lane.stage)}</th>`,
    )
    .join('');
  const rows = keys
    .map(
      (key) =>
        `<tr><td>${frame.reward_labels[key] || key}</td>${frame.lanes.map((lane) => cell(value(lane, key))).join('')}</tr>`,
    )
    .join('');
  const totals = frame.lanes
    .map((lane) =>
      cell(
        keys.reduce((sum, key) => sum + value(lane, key), 0),
        false,
      ),
    )
    .join('');
  $('rewards').innerHTML =
    `<thead><tr><th>${t('Составляющая')}</th>${head}</tr></thead><tbody>${rows}` +
    `<tr class="total"><td>${t('Итого за шаг')}</td>${totals}</tr></tbody>`;
}

function applyFrame(next) {
  const previous = frame;
  frame = next;
  if (frame.error) showError(t('Забег остановлен: {error}', { error: frame.error }));
  if (frame.lanes.length !== lanes.length) buildLanes(frame.lanes.length);
  bufferPose(frame, previous);
  $('loading').hidden = true;
  $('pause').textContent = frame.paused ? t('▶ Играть') : t('Ⅱ Пауза');
  document
    .querySelectorAll('[data-speed]')
    .forEach((button) => button.classList.toggle('selected', Number(button.dataset.speed) === frame.speed));
  $('scenario-wrap').hidden = !frame.supports_command;
  if (document.activeElement !== $('scenario') && frame.scenario) $('scenario').value = frame.scenario;
  $('race-clock').textContent =
    t('{s} с · забег {round}', { s: number(frame.seconds, 1), round: frame.round }) +
    (frame.command_mps == null ? '' : ' · ' + t('команда {n} м/с', { n: number(frame.command_mps, 1) }));
  $('banner').hidden = !frame.ended && !frame.paused;
  $('banner').textContent = frame.ended ? t('Забег окончен — повтор через 2 с') : t('Пауза');
  renderTable();
  if ($('reward-details').open) renderRewards();
}

$('pause').onclick = () => frame && control({ command: 'pause', value: !frame.paused });
$('restart').onclick = () => control({ command: 'restart' });
$('scenario').onchange = () => control({ command: 'scenario', value: $('scenario').value });
$('lane-count').onchange = () =>
  frame && control({ command: 'lanes', lanes: spread(Number($('lane-count').value)) });
$('reward-details').ontoggle = () => frame && $('reward-details').open && renderRewards();
$('fullscreen').onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await $('viewport').requestFullscreen();
  } catch (error) {
    showError(error.message);
  }
};
document
  .querySelectorAll('[data-speed]')
  .forEach(
    (button) => (button.onclick = () => control({ command: 'speed', value: Number(button.dataset.speed) })),
  );
document.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && !['INPUT', 'BUTTON', 'SELECT'].includes(document.activeElement.tagName)) {
    event.preventDefault();
    $('pause').click();
  }
});

try {
  let sunLight;
  ({
    scene,
    camera,
    orbit,
    sun: sunLight,
  } = createViewer($('viewport'), (sun) => {
    // Follow the leading robot along the lanes (its interpolated pose, so the camera does not lag behind
    // a fast runner); the user can still orbit and zoom.
    const leader = drawPoses();
    const dx = leader == null ? 0 : leader - followedX;
    followedX += dx;
    camera.position.x += dx;
    orbit.target.x += dx;
    sun.position.set(followedX + 3, -4, 8);
    sun.target.position.set(followedX, 0, 0);
    if (lanes.length) updateMarks();
  }));
  // Ten lanes are 20 m wide: allow zooming out and cast shadows on all of them.
  orbit.maxDistance = 45;
  Object.assign(sunLight.shadow.camera, { left: -12, right: 12, top: 12, bottom: -12 });
  sunLight.shadow.camera.updateProjectionMatrix();
  geometry = await api('/api/race/model');
  // ?lanes=exp-…:latest,exp-…:228753408 presets the lanes.
  if (params.get('lanes')) {
    const preset = params
      .get('lanes')
      .split(',')
      .map((item) => {
        const [experiment, stage] = item.split(':');
        return { experiment, stage: stage === 'latest' ? stage : Number(stage) };
      });
    control({ command: 'lanes', lanes: preset });
  }
  const first = await api('/api/race/frame');
  // ?add=exp-… (the lab page's «Открыть в 3D»): the branch's newest stage joins the lanes already racing as a
  // lane that follows new checkpoints (the oldest lane gives way at 10). It leaves the address, so a reload
  // does not add it again.
  const added = params.get('add');
  if (added && !params.get('lanes')) {
    const chosen = first.lanes.map(laneSpec);
    if (!chosen.some((lane) => lane.experiment === added && lane.stage === 'latest')) {
      if (chosen.length >= MAX_LANES) chosen.shift();
      control({ command: 'lanes', lanes: [...chosen, { experiment: added, stage: 'latest' }] });
    }
  }
  if (added) {
    params.delete('add');
    history.replaceState(null, '', location.pathname + (params.toString() ? '?' + params : '') + location.hash);
  }
  applyFrame(first);
  const stream = new EventSource(withLang('/api/race/stream?' + query));
  stream.onmessage = (event) => applyFrame(JSON.parse(event.data));
  stream.onopen = () => showError('');
  stream.onerror = () => showError(t('Нет связи с локальным сервером. Открой humanoid_browser.cmd.'));
} catch (error) {
  $('loading').hidden = true;
  showError(t('Не удалось открыть забег: {error}', { error: error.message }));
}
