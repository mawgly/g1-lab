// Lists the Cyrillic text a lab page shows (text, readable attributes, the title), also inside same-origin
// frames. Pages: the site (site.html, journal.html, lab.html), the race (race.html) and the arm's stage player
// (index.html). Paste into the browser console of a page opened in English or Chinese, or run it with the
// preview's JavaScript tool; a row is "where | text  [nearest id or class]".
// Branch names, notes and journal paragraphs stay Russian until humanoid_sim.journal_translate has translated
// them; the language buttons of the site's header (.languages) name each language in itself on purpose and are
// skipped.
(() => {
  const cyrillic = /[А-Яа-яЁё]/;
  const rows = [];
  const where = (element) => {
    const holder = element.closest('[id]');
    return holder ? '#' + holder.id : element.className || element.tagName;
  };
  const scan = (doc, frame) => {
    if (cyrillic.test(doc.title)) rows.push(`${frame} | <title> ${doc.title}`);
    const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const parent = node.parentElement,
        text = node.nodeValue.replace(/\s+/g, ' ').trim();
      if (parent && !['SCRIPT', 'STYLE'].includes(parent.tagName) && !parent.closest('.languages'))
        if (cyrillic.test(text)) rows.push(`${frame} | ${text.slice(0, 140)}  [${where(parent)}]`);
    }
    for (const element of doc.body.querySelectorAll('*'))
      for (const name of ['title', 'aria-label', 'alt', 'placeholder', 'label']) {
        const value = element.getAttribute(name);
        if (value && cyrillic.test(value)) rows.push(`${frame} | @${name} ${value.slice(0, 140)}  [${where(element)}]`);
      }
    for (const child of doc.querySelectorAll('iframe'))
      try {
        if (child.contentDocument?.body) scan(child.contentDocument, `${frame} > ${child.title || 'frame'}`);
      } catch {}
  };
  scan(document, location.pathname);
  return [...new Set(rows)];
})();
