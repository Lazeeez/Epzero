// Runs before first paint: marks that JavaScript is on (for the scroll-reveal styles) and applies the
// saved theme, so the page never flashes the wrong one. Inlined into index.html at build time (no
// extra request before first paint), and allowed by its hash in the Content Security Policy, which
// still forbids every other inline script.
document.documentElement.classList.add("js");
try {
  // Settings saved before the project was renamed from Oryx move to their new keys, once.
  for (var i = localStorage.length - 1; i >= 0; i--) {
    var key = localStorage.key(i);
    if (key && key.indexOf("oryx-") === 0) {
      if (localStorage.getItem("epzero-" + key.slice(5)) === null) localStorage.setItem("epzero-" + key.slice(5), localStorage.getItem(key));
      localStorage.removeItem(key);
    }
  }
  if (localStorage.getItem("epzero-theme") === "light") document.documentElement.dataset.theme = "light";
} catch (e) {
  /* storage blocked: default theme */
}
