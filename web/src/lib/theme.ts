// Theme: dark by default (the site's look), light on request. The choice is a per-viewer
// convenience, so storage failures (private mode, blocked site data) are ignored.

const root = document.documentElement;
const listeners = new Set<() => void>();

export function isDark(): boolean {
  return root.dataset.theme !== "light";
}

/** Calls `listener` whenever the theme changes; returns an unsubscribe function. */
export function onThemeChange(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Reads a design token (CSS custom property) for the current theme. */
export function token(name: string): string {
  return getComputedStyle(root).getPropertyValue(name).trim();
}

export function setTheme(theme: "dark" | "light") {
  if (theme === "light") root.dataset.theme = "light";
  else delete root.dataset.theme;
  try {
    localStorage.setItem("epzero-theme", theme);
  } catch {
    /* ignore */
  }
  listeners.forEach((listener) => listener());
}

export function toggleTheme() {
  setTheme(isDark() ? "light" : "dark");
}

/** Applies the saved theme before first paint. */
export function restoreTheme() {
  try {
    if (localStorage.getItem("epzero-theme") === "light") root.dataset.theme = "light";
  } catch {
    /* storage blocked */
  }
}
