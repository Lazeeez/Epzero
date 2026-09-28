// Light/dark theme: follows the OS unless the viewer picks one with the toggle. The choice is a
// per-viewer convenience, so storage failures (private mode, blocked site data) are ignored.

const root = document.documentElement;
const listeners: (() => void)[] = [];

export function isDark(): boolean {
  return root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
}

/** Calls `listener` whenever the effective theme changes (toggle or OS setting). */
export function onThemeChange(listener: () => void) {
  listeners.push(listener);
}

/** Reads a design token (CSS custom property) for the current theme. */
export function token(name: string): string {
  return getComputedStyle(root).getPropertyValue(name).trim();
}

export function initTheme(toggle: HTMLButtonElement | null) {
  try {
    const saved = localStorage.getItem("oryx-theme");
    if (saved === "light" || saved === "dark") root.dataset.theme = saved;
  } catch {
    /* storage blocked: follow the OS setting */
  }
  const notify = () => listeners.forEach((listener) => listener());
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", notify);
  toggle?.addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try {
      localStorage.setItem("oryx-theme", root.dataset.theme);
    } catch {
      /* ignore */
    }
    notify();
  });
}
