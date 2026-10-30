// The home page (site.html): the site's frame, the paragraphs with inline markup, the clips in the page's language,
// the changing part from the lab's data (data('home'): the «updated» line, the number cards and what the robot can
// do, written each morning by scripts/site/home_update.py) and the e-mail button.
import { STATIC, lang, t, frame, media, data } from './site.js';
import { locale } from './i18n.js';
import { clipElement, wireClip } from './clip.js';

frame('site.html');

// A paragraph marked data-rich is one text node, so it is translated whole; after translation its **bold**,
// `code` and the project sites become markup.
const escape = (text) =>
  String(text).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
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

// Clips (site-<lang>-<name>, labels baked in per language) in the shared player (clip.js): the poster until the clip
// is on screen, then it loops; off screen it stops, so only visible clips are fetched and decoded. The static copy
// has them as .mp4 (humanoid_sim.site_video; a clip the export could not convert stays .webp, which the player falls
// back to), the live server as .webp. wire() is called again for the clips the lab's data adds.
function wire(images) {
  for (const image of images) {
    const name = `site-${lang}-${image.dataset.clip}`;
    const player = clipElement({
      src: media(`${name}.${STATIC ? 'mp4' : 'webp'}`),
      fallback: STATIC ? media(`${name}.webp`) : '',
      poster: media(`${name}-poster.jpg`),
      alt: image.alt,
    });
    image.replaceWith(player);
    wireClip(player);
  }
}
wire(document.querySelectorAll('img[data-clip]'));

// A finished video the page names itself (data-media .mp4 with its data-poster, the same file in every language:
// the crowd of 4096 robots), in the same player.
for (const image of document.querySelectorAll('img[data-media]')) {
  const player = clipElement({ src: media(image.dataset.media), poster: media(image.dataset.poster), alt: image.alt });
  image.replaceWith(player);
  wireClip(player);
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

// --- The lab's data (humanoid_sim/home.py): texts and values arrive translated and in the language's number form.

// Where a number comes from (home.WHERE), on a quiet line under it.
const WHERE = {
  bridge: t('физика Unitree'),
  bridge_copy: t('физика Unitree, со столкновением стоп'),
  skills: t('наша симуляция'),
  goal: t('наша симуляция'),
  package: t('пакет для робота'),
};
// A card shows this many numbers (a weak spot among them if there is one); the rest go under «Подробнее».
const SHOWN = 3;

// One number: label and value, then where it comes from, Unitree v0's and the threshold; a weak spot is flagged.
function fact(row) {
  const meta = [
    row.where && (WHERE[row.where] || row.where),
    row.v0 && t('Unitree v0: {n}', { n: row.v0 }),
    row.target && t('порог {name}', { name: row.target }),
  ]
    .filter(Boolean)
    .map(escape)
    .join(' · ');
  const flag = row.weak ? `<span class="flag">${escape(t('слабое место'))}</span> ` : '';
  return `<li${row.weak ? ' class="weak"' : ''}>${flag}<span class="label">${escape(row.label)}</span> <span class="value">${escape(row.value)}</span>${meta ? `<span class="meta">${meta}</span>` : ''}</li>`;
}

// A field without a number is not shown; its reason stays in the data and in the card's tooltip.
const shown = (rows) => rows.filter((row) => row.value !== null);
const unknown = (rows) =>
  rows
    .filter((row) => row.value === null)
    .map((row) => `${row.label}: ${row.why}`)
    .join('\n');

// The first numbers in the source's order, with the first weak spot among them, the rest folded away.
function facts(rows) {
  const weak = rows.find((row) => row.weak);
  let first = rows.slice(0, SHOWN);
  if (weak && !first.includes(weak)) first = [...rows.filter((row) => !row.weak).slice(0, SHOWN - 1), weak];
  const rest = rows.filter((row) => !first.includes(row));
  return (
    `<ul class="facts">${first.map(fact).join('')}</ul>` +
    (rest.length
      ? `<details class="more"><summary>${escape(t('Подробнее'))} (${rest.length})</summary><ul class="facts">${rest.map(fact).join('')}</ul></details>`
      : '')
  );
}

function skillCard(card) {
  const rows = shown(card.rows);
  if (!rows.length) return '';
  const tip = unknown(card.rows) ? ` title="${escape(unknown(card.rows))}"` : '';
  if (!card.clip) return `<article class="card facts-card"${tip}><b>${escape(card.title)}</b>${facts(rows)}</article>`;
  return `<figure class="clip-card"${tip}>
<div class="media"><img data-clip="${escape(card.clip)}" alt="${escape(card.alt || card.title)}" /></div>
<figcaption><b>${escape(card.title)}</b>${facts(rows)}</figcaption></figure>`;
}

function candidate(item) {
  const clips = item.skills.filter((card) => card.clip),
    others = item.skills.filter((card) => !card.clip);
  return `<section class="candidate" id="skills-${escape(item.id)}">
<div class="candidate-head"><h3>${escape(item.title)}</h3>${item.about ? `<p>${escape(item.about)}</p>` : ''}
<ul class="facts summary">${shown(item.summary).map(fact).join('')}</ul></div>
<div class="grid cards-260 clips">${clips.map(skillCard).join('')}</div>
${others.length ? `<div class="grid cards-340 facts-grid">${others.map(skillCard).join('')}</div>` : ''}
</section>`;
}

function updated(home) {
  const day = new Date(`${home.date}T12:00:00`)
    .toLocaleDateString(locale, { day: 'numeric', month: 'long', year: 'numeric' })
    .replace(/\s*г\.$/, '');
  const { active, total } = home.roadmap;
  const stage = !active.length
    ? ''
    : (active.length === 1
        ? t('этап {n} из {n1}', { n: active[0].number, n1: total })
        : t('этапы {name} из {n}', { name: active.map((s) => s.number).join(', '), n: total })) +
      ': ' +
      active.map((s) => s.title).join('; ');
  return [t('Обновлено {name}', { name: day }), stage, t('только симуляция, робота пока нет')]
    .filter(Boolean)
    .join(' · ');
}

// A research experiment (home.json «research»): its own block after the candidates, the robot's clip with the licence
// and what it shows.
const researchBlock = (block, index) => `<h2${index ? '' : ' id="research-title"'}>${escape(block.title)}</h2>
<article class="story-card wide research"><div class="media"><img data-clip="${escape(block.clip)}" alt="${escape(block.alt)}" /></div>
<div class="body"><span class="text">${escape(block.text)}</span><span class="licence">${escape(block.licence)}</span></div></article>`;

// «Как устроена лаборатория»: the lab's own counts (humanoid_sim.stack, data('stack')); without them the page's
// numbers stay as written.
data('stack')
  .then((counts) => {
    for (const element of document.querySelectorAll('[data-count]')) {
      const value = counts[element.dataset.count];
      if (value !== undefined && value !== null) element.textContent = String(value).replace('.', lang === 'ru' ? ',' : '.');
    }
  })
  .catch(() => {});

const stats = document.getElementById('home-stats'),
  candidates = document.getElementById('home-candidates'),
  research = document.getElementById('home-research');
try {
  const home = await data('home');
  if (home.empty) throw new Error('no home data');
  document.getElementById('home-updated').textContent = updated(home);
  // The number cards are off the page since 30.10 (Pavel: not on the home page for now); home.json keeps them.
  if (stats)
    stats.innerHTML = home.stats
      .map(
        (stat) =>
          `<div class="card${stat.accent ? ' accent' : ''}"><span class="stat">${escape(stat.value)}</span><span>${escape(stat.label)}</span></div>`,
      )
      .join('');
  candidates.innerHTML = home.candidates.map(candidate).join('');
  wire(candidates.querySelectorAll('img[data-clip]'));
  if (home.research?.length) {
    research.innerHTML = home.research.map(researchBlock).join('');
    research.closest('section').hidden = false;
    wire(research.querySelectorAll('img[data-clip]'));
  }
  // The page grew above and around the anchor after the browser had jumped to it: jump again, below the header.
  if (location.hash) document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
} catch (error) {
  if (stats) stats.closest('section').hidden = true;
  candidates.innerHTML = `<p class="note">${escape(t('Данные главной ещё не собраны'))}</p>`;
}
