// Runs before first paint: marks that JavaScript is on (for the scroll-reveal styles) and applies the
// saved theme, so the page never flashes the wrong one. Inlined into index.html at build time (no
// extra request before first paint), and allowed by its hash in the Content Security Policy, which
// still forbids every other inline script.
document.documentElement.classList.add("js");
try {
  if (localStorage.getItem("oryx-theme") === "light") document.documentElement.dataset.theme = "light";
} catch (e) {
  /* storage blocked: default theme */
}
