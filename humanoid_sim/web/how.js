// «Как это работает» (how.html): the site's frame, the versions of the tools and the lab's counts (chapter 2), read
// by humanoid_sim.stack from the lab's own files (data('stack'): /api/lab/stack live, data/stack.<lang>.json static).
import { frame, data } from './site.js';

frame('how.html');

data('stack')
  .then((versions) => {
    for (const element of document.querySelectorAll('[data-version], [data-count]')) {
      const value = versions[element.dataset.version || element.dataset.count];
      if (value) element.textContent = value;
    }
  })
  .catch(() => {}); // without the data the cards stay as they are, only the version numbers are missing
