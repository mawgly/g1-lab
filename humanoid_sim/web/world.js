// «Модель мира» (world.html): the site's frame and the journal's clips the page shows (media: the server's journal
// pictures live, media/<file> in the static copy, where site_export copies the journal's media and writes each
// animated WebP as an .mp4, renaming data-media), each in the shared player (clip.js) with its data-poster.
import { frame, media } from './site.js';
import { clipElement, wireClip } from './clip.js';

frame('world.html');

for (const image of document.querySelectorAll('img[data-media]')) {
  if (!image.dataset.poster) {
    image.src = media(image.dataset.media);
    continue;
  }
  const player = clipElement({ src: media(image.dataset.media), poster: media(image.dataset.poster), alt: image.alt });
  image.replaceWith(player);
  wireClip(player);
}
