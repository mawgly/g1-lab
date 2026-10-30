// The site's pre-rendered clips (home.js, journal.js, world.js): one player with one button.
// - .mp4 (the static copy, humanoid_sim.site_video): a muted inline <video> over its poster picture. muted and
//   playsinline (attribute and property) are what lets iPhone Safari start it by itself; preload="none" and its src
//   set at the first play, so only a clip that is played downloads anything; its poster is set when it comes near
//   the screen (the picture under it is a lazy <img>, which also gives the box its size before anything loads). It
//   plays while at least half of it is on screen and pauses off screen.
//   A refused play() (Low Power Mode, autoplay off) leaves the button to start it by hand; a clip that cannot play
//   at all falls back to its animated WebP when there is one.
// - .webp (the live lab server, or a clip the export could not convert): the poster picture swapped for the looping
//   animation, the old behaviour; `auto` says whether it starts on screen by itself (home, world) or only by hand
//   (the journal, which has many).
// The button is always there: «▶ Смотреть», and while playing a small «❚❚» in the corner; a pause by hand holds
// until the next press. With reduced motion nothing starts by itself.
import { t } from './i18n.js';

const WATCH = t('▶ Смотреть'),
  HINT = t('Нажмите, чтобы смотреть или остановить');
const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
const watching = 'IntersectionObserver' in window;
const esc = (v) =>
  String(v ?? '').replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c],
  );
const isVideo = (src) => /\.mp4$/i.test(src);

// Per player: { mode: 'video' | 'img', auto, visible, manual: null | 'play' | 'pause', playing }.
const players = new WeakMap();

// The markup of one clip. src: the clip's URL (.mp4 or .webp), poster: its still picture, alt, fallback: a .webp to
// show if an .mp4 cannot play, auto: whether a .webp starts on screen by itself (an .mp4 always does).
export function clipMarkup({ src, poster, alt = '', fallback = '', auto = true }) {
  const video = isVideo(src);
  const picture = `<img class="clip-poster" src="${esc(poster || src)}" alt="${esc(alt)}" loading="lazy" />`;
  const motion = video
    ? `<video muted playsinline loop preload="none" disablepictureinpicture aria-hidden="true" tabindex="-1" data-src="${esc(src)}"></video>`
    : '';
  const button = poster
    ? `<button type="button" class="clip-toggle" aria-pressed="false" title="${esc(HINT)}">${esc(WATCH)}</button>`
    : '';
  return `<div class="clip-player" data-src="${esc(src)}" data-poster="${esc(poster)}" data-fallback="${esc(fallback)}" data-auto="${video || auto ? 1 : 0}">${picture}${motion}${button}</div>`;
}

// The same as an element.
export function clipElement(options) {
  const box = document.createElement('div');
  box.innerHTML = clipMarkup(options);
  return box.firstElementChild;
}

function label(player, playing) {
  const state = players.get(player);
  state.playing = playing;
  player.classList.toggle('playing', playing);
  const button = player.querySelector('.clip-toggle');
  if (!button) return;
  button.setAttribute('aria-pressed', String(playing));
  button.textContent = playing ? '❚❚' : WATCH;
  if (playing) button.setAttribute('aria-label', HINT);
  else button.removeAttribute('aria-label');
}

// The video element gets its file and poster only when it is first played or near the screen.
function dress(player) {
  const video = player.querySelector('video');
  if (!video) return;
  if (!video.getAttribute('poster') && player.dataset.poster) video.poster = player.dataset.poster;
  video.muted = true;
  video.defaultMuted = true;
  video.playsInline = true;
}

// An .mp4 that cannot play: the animated WebP if the export kept one, else the poster alone.
function fallBack(player) {
  const state = players.get(player),
    video = player.querySelector('video');
  if (state.mode !== 'video') return;
  video?.remove();
  state.mode = 'img';
  if (player.dataset.fallback) {
    player.dataset.src = player.dataset.fallback;
    sync(player);
  } else {
    // Nothing to play: the poster stays, without a button that would do nothing.
    label(player, false);
    player.classList.add('failed');
    player.querySelector('.clip-toggle')?.remove();
    state.manual = 'pause';
  }
}

function start(player) {
  const state = players.get(player);
  if (state.mode === 'img') {
    const image = player.querySelector('img');
    if (image.getAttribute('src') !== player.dataset.src) image.src = player.dataset.src;
    return label(player, true);
  }
  const video = player.querySelector('video');
  dress(player);
  if (!video.getAttribute('src')) video.src = video.dataset.src;
  player.classList.remove('blocked');
  const playing = video.play();
  if (playing)
    playing.catch((error) => {
      if (state.mode !== 'video') return;
      if (error.name === 'NotAllowedError') {
        // Autoplay refused: the reader starts it with the button (a press is a gesture iOS accepts).
        if (state.manual !== 'play') player.classList.add('blocked');
        label(player, false);
      } else if (error.name === 'NotSupportedError') fallBack(player);
    });
}

function stop(player) {
  const state = players.get(player);
  if (state.mode === 'img') {
    const image = player.querySelector('img');
    if (player.dataset.poster && image.getAttribute('src') !== player.dataset.poster)
      image.src = player.dataset.poster;
    return label(player, false);
  }
  const video = player.querySelector('video');
  if (!video.paused) video.pause();
  label(player, false);
}

// Plays or stops a clip by what it should do now: a press by hand first, then on screen and automatic.
function sync(player) {
  const state = players.get(player);
  const wanted =
    state.manual === 'pause'
      ? false
      : state.manual === 'play'
        ? state.visible
        : state.auto && !still && watching && state.visible;
  if (wanted) start(player);
  else if (state.playing || state.mode === 'video') stop(player);
}

const near = watching
  ? new IntersectionObserver((entries) => entries.forEach((e) => e.isIntersecting && dress(e.target)), {
      rootMargin: '400px 0px',
    })
  : null;
const seen = watching
  ? new IntersectionObserver(
      (entries) =>
        entries.forEach((e) => {
          const state = players.get(e.target);
          if (!state) return;
          state.visible = e.isIntersecting && e.intersectionRatio >= 0.5;
          sync(e.target);
        }),
      { threshold: [0, 0.5] },
    )
  : null;

// Makes one clip element play (once).
export function wireClip(player) {
  if (players.has(player)) return;
  const video = player.querySelector('video');
  players.set(player, {
    mode: video ? 'video' : 'img',
    auto: player.dataset.auto === '1',
    visible: !watching,
    manual: null,
    playing: false,
  });
  if (video) {
    // Only while the video is the clip: a video dropped for its WebP still sends a pause.
    const current = () => players.get(player)?.mode === 'video';
    video.addEventListener('play', () => current() && label(player, true)); // also while it still loads

    video.addEventListener('pause', () => current() && label(player, false));
    video.addEventListener('error', () => current() && video.getAttribute('src') && fallBack(player));
  }
  // A press on the button or the picture: play or pause by hand.
  player.addEventListener('click', () => {
    const state = players.get(player);
    state.manual = state.playing ? 'pause' : 'play';
    state.visible = true; // pressed, so on screen
    sync(player);
  });
  if (watching) {
    near.observe(player);
    seen.observe(player);
  } else dress(player);
}

// Every clip element in root (or root itself) not yet wired.
export function wireClips(root) {
  const list = root.matches?.('.clip-player') ? [root] : root.querySelectorAll('.clip-player');
  for (const player of list) wireClip(player);
}

// Stops and forgets the clips in root before it is drawn anew or emptied.
export function releaseClips(root) {
  for (const player of root.querySelectorAll('.clip-player')) {
    if (!players.has(player)) continue;
    player.querySelector('video')?.pause();
    near?.unobserve(player);
    seen?.unobserve(player);
    players.delete(player);
  }
}
