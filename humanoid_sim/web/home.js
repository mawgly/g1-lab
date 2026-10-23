// The home page (site.html): the site's frame, the paragraphs with inline markup, the clips in the page's language
// and the e-mail button.
import { lang, t, frame, media } from './site.js';

frame('site.html');

// A paragraph marked data-rich is one text node, so it is translated whole; after translation its **bold**,
// `code` and the project sites become markup.
const escape = (text) =>
  text.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const SITES = /\b(moe3d\.ru|mawgly\.ru|marketplacehero\.ru)\b/g;
for (const element of document.querySelectorAll('[data-rich]'))
  element.innerHTML = escape(element.textContent.replace(/\s+/g, ' ').trim())
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(SITES, '<a href="https://$1">$1</a>');

// Number-only table cells carry Russian decimal commas; English and Chinese use a point.
if (lang !== 'ru')
  for (const cell of document.querySelectorAll('.num'))
    if (!/[А-Яа-яЁё]/.test(cell.textContent)) cell.textContent = cell.textContent.replace(/(\d),(\d)/g, '$1.$2');

// Clips (site-<lang>-<name>.webp, labels baked in per language): the poster until the clip is on screen, then the
// looping animation; off screen it goes back to the poster, so only visible clips are decoded. With reduced motion
// a click plays or stops a clip instead.
const clips = [...document.querySelectorAll('img[data-clip]')];
for (const image of clips) {
  image.dataset.poster = media(`site-${lang}-${image.dataset.clip}-poster.jpg`);
  image.dataset.play = media(`site-${lang}-${image.dataset.clip}.webp`);
  image.src = image.dataset.poster;
}
const show = (image, playing) => {
  const source = playing ? image.dataset.play : image.dataset.poster;
  if (image.getAttribute('src') !== source) image.src = source;
};
if (matchMedia('(prefers-reduced-motion: reduce)').matches || !('IntersectionObserver' in window))
  for (const image of clips) {
    image.title = t('Нажмите, чтобы смотреть или остановить');
    image.style.cursor = 'pointer';
    image.onclick = () => show(image, image.getAttribute('src') !== image.dataset.play);
  }
else {
  const observer = new IntersectionObserver(
    (entries) => entries.forEach((entry) => show(entry.target, entry.isIntersecting)),
    { rootMargin: '120px 0px' },
  );
  clips.forEach((image) => observer.observe(image));
}

// The address is assembled only on a click, from reversed parts, so it is nowhere in the page as one string.
const button = document.getElementById('show-email');
button.onclick = () => {
  const address = ['96ylgwam', 'moc.liamg']
    .map((part) => [...part].reverse().join(''))
    .join(String.fromCharCode(64));
  const link = document.createElement('a');
  link.className = 'button ghost';
  link.href = 'mailto:' + address;
  link.textContent = address;
  button.replaceWith(link);
  link.focus();
};
