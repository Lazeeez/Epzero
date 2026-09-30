// Client-side routing for the three pages, with the View Transitions API for page changes.
// Paths are relative to the site's base (e.g. /Oryx/).

export type Route = "home" | "try" | "play" | "benchmarks" | "notfound";

const BASE = import.meta.env.BASE_URL.replace(/\/$/, "");

const ROUTES: Record<string, Route> = { "": "home", "/": "home", "/try": "try", "/play": "play", "/benchmarks": "benchmarks" };

function parse(pathname: string): { route: Route; hash: string } {
  let path = pathname.startsWith(BASE) ? pathname.slice(BASE.length) : pathname;
  path = path.replace(/\/index\.html$/, "").replace(/\/$/, "");
  // Unknown paths get a real "not found" page rather than silently showing the landing page.
  return { route: ROUTES[path] ?? "notfound", hash: location.hash };
}

class Router {
  route = $state<Route>(parse(location.pathname).route);

  href(route: Route, hash = ""): string {
    const path = route === "home" ? "/" : `/${route}`;
    return `${BASE}${path}${hash}`;
  }

  go(route: Route, hash = "") {
    const url = this.href(route, hash);
    if (url === location.pathname + location.hash) return;
    const update = () => {
      history.pushState({}, "", url);
      this.apply(route, hash);
    };
    if (route !== this.route && "startViewTransition" in document && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      (document as Document & { startViewTransition(cb: () => void): unknown }).startViewTransition(update);
    } else {
      update();
    }
  }

  private apply(route: Route, hash: string) {
    const changed = route !== this.route;
    this.route = route;
    queueMicrotask(() => {
      let target: Element | null = null;
      try {
        target = hash ? document.querySelector(decodeURIComponent(hash)) : null;
      } catch {
        target = null; // not a valid selector (e.g. a stray #fragment)
      }
      if (target) target.scrollIntoView({ behavior: changed ? "instant" : "smooth", block: "start" });
      else if (changed) window.scrollTo({ top: 0, behavior: "instant" });
      // Screen readers: move focus to the new page's content, as a full page load would.
      if (changed) document.getElementById("main")?.focus({ preventScroll: true });
    });
  }

  /** Intercepts same-site link clicks and history navigation. Call once. */
  start() {
    addEventListener("popstate", () => this.apply(parse(location.pathname).route, location.hash));
    document.addEventListener("click", (event) => {
      if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      const anchor = (event.target as Element).closest("a");
      if (!anchor || anchor.target || anchor.hasAttribute("download") || anchor.origin !== location.origin) return;
      if (!anchor.pathname.startsWith(BASE)) return;
      event.preventDefault();
      const next = parse(anchor.pathname);
      this.go(next.route, anchor.hash);
    });
  }
}

export const router = new Router();
