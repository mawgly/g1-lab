// «Разъём» of the G1 policy as a card rack, for the lab journal (anatomy kit, see figures/anatomy/README.md).
//
// Truth (humanoid_sim/tasks/g1_tensor_math.py, deploy_terms and history_observation): one frame is 98 numbers
// in deploy.yaml order, 3 + 3 + 3 + 2 + 29 + 29 + 29; with task_config history 5 every term keeps its own buffer
// of the last 5 frames, oldest first, and the terms follow each other: 7 × (5 × size) = 490 inputs, 50 times a
// second. The rack has seven compartments (one per term, as wide as its contacts), each holding five cards
// (frames). Every control tick a new card enters at the front, the stack moves back one slot and the oldest card
// slides off into the tray; the dial counts ticks, one turn = 50 ticks = 1 s of robot time.
//
//   .tools\node\node.exe humanoid_sim\figures\connector.mjs [--light]
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "./anatomy/kit/iso-kit.mjs";
import { kitScript } from "./anatomy/scripts/inline-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const theme = process.argv.includes("--light") ? "light" : "dark";

const TERMS = [3, 3, 3, 2, 29, 29, 29];
const FRAME = TERMS.reduce((a, b) => a + b, 0);
const HISTORY = 5;
const RATE = 50;
const SLOW = 80; // the figure's tick lasts 1.6 s; the robot's lasts 20 ms

const WIDTH = 760;
const HEIGHT = 330;

const PIN = 2.6; // contact pitch along a card's top edge
const WALL = 2.4;
const END = 5;
const CLEAR = 0.8;
const CARD_D = 2.2;
const CARD_H = 24;
const WALL_H = 15;
const END_H = 30;
const BED_TOP = 6;
const SLOT = 7;
const FRONT = 14; // the newest card's y; older ones step back by SLOT

const cardW = (n) => n * PIN + 3;
const LAYOUT = (() => {
  const total = 2 * END + (TERMS.length - 1) * WALL + TERMS.reduce((s, n) => s + cardW(n) + 2 * CLEAR, 0);
  let x = -total / 2;
  const walls = [];
  const cards = [];
  TERMS.forEach((n, index) => {
    const w = index === 0 ? END : WALL;
    walls.push({ x, w });
    x += w + CLEAR;
    cards.push({ x, w: cardW(n), n });
    x += cardW(n) + CLEAR;
  });
  walls.push({ x, w: END });
  return { total, from: -total / 2, to: total / 2, walls, cards };
})();

const BED = { x: LAYOUT.from - 6, y: -26, w: LAYOUT.total + 12, d: 52, r: 2 };
const TRAY = { x: LAYOUT.from + 4, y: -42, w: LAYOUT.total - 8, d: 12, r: 2 };
const DIAL = [LAYOUT.to + 40, 6];
const DIAL_R = 16;
const DIAL_Z = 10;
const DIAL_H = 6;
const DIAL_TOP = DIAL_Z + DIAL_H;
const PLATE = { x: LAYOUT.from - 24, y: -50, w: LAYOUT.total + 24 + 84, d: 96, r: 8 };
const PLATE_H = 12;
const FOOT_H = 5;

const P = k.fitProjection(
  [...k.boxCorners(PLATE, -PLATE_H - FOOT_H, 0), [LAYOUT.from, 0, BED_TOP + END_H], [LAYOUT.to, 0, BED_TOP + END_H], [DIAL[0], DIAL[1], DIAL_TOP]],
  WIDTH,
  HEIGHT,
  { pad: 26, azimuth: 58, elevation: 32 },
);

const at = (point) => k.iso(point, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));

function plate() {
  const out = [`<path class="iso-halo" d="${k.haloOf(PLATE, -PLATE_H - FOOT_H, PLATE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(PLATE, 22))) {
    out.push(S(k.cylinder(x, y, 7, -PLATE_H - FOOT_H, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 5.4, -PLATE_H - FOOT_H + 1.6, FOOT_H - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(PLATE, -PLATE_H, PLATE_H, P, 8, 1.6), { tone: "mid" }));
  out.push(L(k.planOutline(k.insetPlan(PLATE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(PLATE, 10);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.2, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.5 }));
  // Ruler strip along the front: a tick every contact pitch, a major one every 5.
  const strip = { x: LAYOUT.from, y: PLATE.y + PLATE.d - 15, w: LAYOUT.total, d: 7, r: 1 };
  out.push(S(k.slabOf(strip, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const ticks = k.topTicks(strip.x + 4, strip.x + strip.w - 4, PIN, 5, strip.y + 0.8, 0.6, [2, 3.6], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(D([at([strip.x + 2, strip.y + 3.5, 0.6]), at([strip.x + strip.w - 2, strip.y + 3.5, 0.6])], { size: 0.45 }));
  // Label plate with an engraved name, on the right in front of the dial.
  const label = { x: LAYOUT.to + 16, y: PLATE.y + PLATE.d - 17, w: 44, d: 12, r: 1.5 };
  out.push(S(k.slabOf(label, 0, 0.8, P, 4), { tone: "lo", crease: "none" }));
  out.push(L(k.planOutline(k.insetPlan(label, 1.6), 0.8, P), { tone: "faint" }));
  out.push(k.faceTextSvg(k.topMatrix([label.x + label.w / 2, label.y + 5.2], 0.8, P, "x"), "G1 · 490", { size: 3.2, tone: "mid", anchor: "middle" }));
  out.push(L(k.lineOnTop([label.x + 6, label.y + 8.6], [label.x + label.w - 14, label.y + 8.6], 0.8, P), { tone: "lo", free: true }));
  out.push(D([at([label.x + 3, label.y + label.d / 2, 0.8]), at([label.x + label.w - 3, label.y + label.d / 2, 0.8])], { size: 0.45 }));
  return out.join("");
}

function tray() {
  const out = [];
  out.push(S(k.slabOf(TRAY, 0, 4, P, 4, 0.6), { tone: "lo" }));
  out.push(L(k.planOutline(k.insetPlan(TRAY, 2.2), 4, P, 4), { tone: "faint" }));
  out.push(D(k.corners(TRAY, 3).map(([x, y]) => at([x, y, 4])), { size: 0.45 }));
  return out.join("");
}

function bed() {
  const out = [];
  out.push(S(k.slabOf(BED, 0, BED_TOP, P, 4, 0.8), { tone: "mid" }));
  // Five slot grooves per compartment, wall to wall: the five places a frame can be.
  for (let slot = 0; slot < HISTORY; slot += 1) {
    const y = FRONT - SLOT * slot;
    LAYOUT.walls.slice(0, -1).forEach((wall, index) => {
      const next = LAYOUT.walls[index + 1];
      out.push(L(k.lineOnTop([wall.x + wall.w, y - 1.6], [next.x, y - 1.6], BED_TOP, P), { tone: "faint" }));
      out.push(L(k.lineOnTop([wall.x + wall.w, y + 1.6], [next.x, y + 1.6], BED_TOP, P), { tone: "faint" }));
    });
  }
  // The feed slot in front, where the next frame waits.
  LAYOUT.walls.slice(0, -1).forEach((wall, index) => {
    const next = LAYOUT.walls[index + 1];
    out.push(L(k.lineOnTop([wall.x + wall.w, FRONT + SLOT - 1.6], [next.x, FRONT + SLOT - 1.6], BED_TOP, P), { tone: "lo" }));
  });
  return out.join("");
}

function wall(index) {
  const { x, w } = LAYOUT.walls[index];
  const end = index === 0 || index === LAYOUT.walls.length - 1;
  const h = end ? END_H : WALL_H;
  const plan = { x, y: BED.y + 1, w, d: BED.d - 2, r: end ? 1.2 : 0.6 };
  const band = end ? L(k.sideSeam(plan, BED_TOP + h - 6, P, 3), { tone: "faint" }) : "";
  const out = [S(k.slabOf(plan, BED_TOP, h, P, 3, end ? 0.6 : 0.3), { tone: "mid", inner: band })];
  out.push(D([at([x + w / 2, plan.y + 3, BED_TOP + h]), at([x + w / 2, plan.y + plan.d - 3, BED_TOP + h])], { size: end ? 0.5 : 0.4 }));
  if (end) {
    out.push(L(k.planOutline(k.insetPlan(plan, 1.2), BED_TOP + h, P, 3), { tone: "faint" }));
    // A slot index scale on the end plate's top: five marks, one per history slot.
    out.push(D(Array.from({ length: HISTORY }, (_, slot) => at([x + w / 2, FRONT - SLOT * slot, BED_TOP + h])), { size: 0.55, tone: "hi" }));
  }
  return out.join("");
}

// One card of one term at rest in the front slot; the live code moves it along y.
function card(term) {
  const { x, w, n } = LAYOUT.cards[term];
  const plan = { x, y: FRONT - CARD_D / 2, w, d: CARD_D, r: 0.5 };
  // The card's edge connector: two seams round its foot, where it sits in the slot.
  const seams = [3, 5.2].map((z) => L(k.sideSeam(plan, BED_TOP + z, P, 2), { tone: "lo" })).join("");
  const out = [S(k.slabOf(plan, BED_TOP, CARD_H, P, 2, 0), { tone: "hi", crease: "lo", inner: seams })];
  const pins = Array.from({ length: n }, (_, i) => at([x + 1.5 + PIN * (i + 0.5), FRONT, BED_TOP + CARD_H]));
  out.push(D(pins, { size: 0.55, tone: "mid" }));
  return out.join("");
}

function dial() {
  const out = [];
  const [x, y] = DIAL;
  const block = { x: x - 20, y: y - 20, w: 40, d: 40, r: 3 };
  out.push(S(k.slabOf(block, 0, DIAL_Z, P, 4, 0.8), { tone: "mid" }));
  out.push(D(k.corners(block, 3.5).map(([cx, cy]) => at([cx, cy, DIAL_Z])), { size: 0.45 }));
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z, DIAL_H, P, 64, 0.8), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 2.6, DIAL_TOP, 84, P, 0.3), { tone: "lo" }));
  out.push(L(k.sideArc(x, y, DIAL_R, DIAL_TOP - 2.6, P, 40), { tone: "faint" }));
  out.push(L(k.ring(x, y, DIAL_R - 2, DIAL_TOP, P, 64), { tone: "lo" }));
  const ticks = k.radialTicks(x, y, DIAL_R - 2.6, RATE, 5, DIAL_TOP, [1.1, 2.4], P);
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(L(k.ring(x, y, 1.5, DIAL_TOP, P, 16), { tone: "mid" }));
  return out.join("");
}

const needleAt = (turn) => {
  const angle = turn * Math.PI * 2;
  const [x, y] = DIAL;
  return k.lineOnTop([x - Math.sin(angle) * 3, y + Math.cos(angle) * 3], [x + Math.sin(angle) * (DIAL_R - 3.4), y - Math.cos(angle) * (DIAL_R - 3.4)], DIAL_TOP + 0.2, P);
};

// Compartments in painter's order: the viewer sits towards +x and +y, so each wall is drawn before the cards
// on its +x side and after the cards on its -x side; inside a compartment the cards sort by y (live code).
const ROWS = HISTORY + 1; // five in the rack plus the one arriving
const rack = [];
LAYOUT.cards.forEach((_, term) => {
  rack.push(wall(term));
  const cards = Array.from({ length: ROWS }, (_, row) => `<g data-card="${row}" data-term="${term}">${card(term)}</g>`);
  rack.push(`<g data-compartment="${term}">${cards.join("")}</g>`);
});
rack.push(wall(LAYOUT.walls.length - 1));

const svg = k.figureSvg({
  width: WIDTH,
  height: HEIGHT,
  label:
    "Стойка с картами на плите: семь отсеков, по одному на каждое показание робота, шириной по числу контактов (3, 3, 3, 2, 29, 29, 29). В каждом отсеке пять карт: пять последних кадров, старый сзади, новый спереди. Каждый такт спереди входит новая карта, стопка сдвигается назад, старая уходит в лоток; циферблат считает такты, 50 в секунду.",
  body: [plate(), tray(), bed(), ...rack, dial(), `<path data-part="needle" class="iso-line" data-tone="hi" data-free="start" d="${needleAt(0)}" style="stroke-width:1.1"/>`],
});

const live = `
const P = ${JSON.stringify(P)};
const DIAL = ${JSON.stringify(DIAL)};
const DIAL_R = ${DIAL_R}, DIAL_TOP = ${DIAL_TOP};
const ROWS = ${ROWS}, HISTORY = ${HISTORY}, FRAME = ${FRAME}, RATE = ${RATE}, SLOW = ${SLOW};
const FRONT = ${FRONT}, SLOT = ${SLOT};
const MOVE = 0.7, REST = 0.9; // seconds of the slowed tick: ${SLOW} × 20 ms = 1.6 s
const stage = document.querySelector(".iso-stage");
const svg = stage.querySelector("svg");
const needle = svg.querySelector('[data-part="needle"]');
const readout = document.querySelector("[data-readout]");
const still = matchMedia("(prefers-reduced-motion: reduce)");
const compartments = [...svg.querySelectorAll("[data-compartment]")];
const cards = compartments.map((box) => [...box.querySelectorAll("[data-card]")]);
const needleAt = ${needleAt.toString().replace("k.lineOnTop", "lineOnTop")};
// Row r sits at slot (r + ticks) mod ROWS − 1: −1 is arriving in front, 0 the newest frame, 4 the oldest.
let ticks = 0, phase = 1, clock = 0, frame = 0, last = 0, touring = true, visible = false, idle = 0, queued = 0;
const slotOf = (row) => ((row + ticks) % ROWS) - 1;
const yOf = (p) => (p <= HISTORY - 1 ? FRONT - SLOT * p : FRONT - SLOT * (HISTORY - 1) - 12 * (p - HISTORY + 1));
const alphaOf = (p) => (p < 0 ? Math.max(0, 1 + p) : p > HISTORY - 1 ? Math.max(0, 1 - 1.25 * (p - HISTORY + 1)) : 1);
const ease = (u) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));
function order() {
  // Nearer cards (larger y) paint later; only the row that wraps from the back to the front changes rank.
  for (let term = 0; term < cards.length; term += 1) {
    const sorted = [...cards[term]].sort((a, b) => slotOf(+b.dataset.card) - slotOf(+a.dataset.card));
    for (const el of sorted) compartments[term].appendChild(el);
  }
}
function draw() {
  const u = ease(phase);
  for (let term = 0; term < cards.length; term += 1) {
    for (const el of cards[term]) {
      const p = slotOf(+el.dataset.card) + u;
      el.setAttribute("transform", translateAlong(P, [0, yOf(p) - FRONT, 0]));
      el.setAttribute("opacity", alphaOf(p).toFixed(3));
      const solid = el.querySelector(".iso-solid");
      if (Math.round(p) === 0) solid.setAttribute("data-lit", ""); else solid.removeAttribute("data-lit");
    }
  }
  const count = ticks + u;
  needle.setAttribute("d", needleAt((count % RATE) / RATE));
  const newest = ticks + (u > 0.5 ? 1 : 0);
  const text = "такт " + newest + " · в окне кадры " + Math.max(1, newest - HISTORY + 1) + "–" + Math.max(1, newest) + " · на входе " + HISTORY + " × " + FRAME + " = " + HISTORY * FRAME;
  if (readout.textContent !== text) readout.textContent = text;
  stage.setAttribute("aria-valuetext", text);
}
function step() { if (phase >= 1) { phase = 0; return true; } queued = Math.min(queued + 1, 3); return false; }
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 30) : 1 / 60;
  last = now;
  if (touring) { clock += dt; if (clock >= MOVE + REST) { clock = 0; step(); } }
  if (phase < 1) {
    phase = still.matches ? 1 : Math.min(1, phase + dt / MOVE);
    if (phase >= 1) { ticks += 1; phase = 1; order(); if (queued) { queued -= 1; phase = 0; } }
  }
  draw();
  frame = visible && (touring || phase < 1) ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame && visible) { last = 0; frame = requestAnimationFrame(tick); } };
const nudge = () => { touring = false; clock = 0; step(); clearTimeout(idle); idle = setTimeout(() => { touring = !still.matches; run(); }, 6000); run(); };
new IntersectionObserver((entries) => { visible = entries[entries.length - 1].isIntersecting; if (visible) run(); }, { rootMargin: "120px 0px" }).observe(stage);
if (still.matches) touring = false;
stage.addEventListener("click", nudge);
stage.addEventListener("keydown", (event) => {
  if (event.key !== " " && event.key !== "Enter" && event.key !== "ArrowRight") return;
  event.preventDefault();
  nudge();
});
ticks = HISTORY - 1;
order();
draw();
// Inside the lab journal's frame: tell the page our content height whenever it changes (the frame may load while
// its tab is hidden), so the frame fits without a scrollbar.
const report = () => {
  if (parent === window) return;
  const bottom = document.body.getBoundingClientRect().bottom + parseFloat(getComputedStyle(document.body).marginBottom || "0");
  if (bottom > 0) parent.postMessage({ anatomyHeight: Math.ceil(bottom + scrollY) }, "*");
};
new ResizeObserver(report).observe(document.body);
addEventListener("load", report);
addEventListener("message", (event) => { if (event.data && event.data.anatomyMeasure) report(); });
[300, 1000, 2500].forEach((ms) => setTimeout(report, ms));
`;

const body = k.plateHtml({
  fig: "Схема 1",
  title: "«Разъём» политики G1",
  hint: "Нажмите — придёт новый кадр",
  readout: `такт ${HISTORY - 1} · на входе ${HISTORY} × ${FRAME} = ${HISTORY * FRAME}`,
  keys: [
    { mark: "lit", label: "Новый кадр: 98 чисел, пришедших от датчиков за последние 20 мс" },
    { mark: "raised", label: "Семь отсеков — семь показаний; у каждого свой буфер на 5 кадров" },
    { mark: "edge", label: "Циферблат: такты, один оборот — 50 тактов, 1 секунда" },
  ],
  caption: `Так политика видит робота. Один кадр — ${FRAME} чисел в порядке deploy.yaml: угловая скорость корпуса 3, направление силы тяжести 3, команда джойстика 3, фаза шага 2, углы 29 суставов, их скорости 29 и прошлый ответ политики 29. Каждое показание хранит свои ${HISTORY} последних кадра, от старого к новому, и на вход идут отсек за отсеком: ${HISTORY} × ${FRAME} = ${HISTORY * FRAME} чисел, ${RATE} раз в секунду. Здесь такт замедлен в ${SLOW} раз. Если на роботе порядок или число кадров будут другими, политика увидит мир искажённым: поэтому «разъём» сверяется с кодом Unitree до выгрузки.`,
  body: `<div class="iso-stage" tabindex="0" role="button" aria-label="Стойка с кадрами: Enter или пробел подаёт новый кадр">${svg}</div>`,
});

const out = join(HERE, "out");
mkdirSync(out, { recursive: true });
const file = theme === "light" ? "connector-light.html" : "connector.html";
writeFileSync(join(out, file), k.pageHtml({ title: "«Разъём» политики G1", theme, body, script: kitScript() + live, width: 820 }));
console.log("wrote", join(out, file));
