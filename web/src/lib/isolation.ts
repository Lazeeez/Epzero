// Cross-origin isolation (needed for WebAssembly threads) as set up by public/sw-register.js.
type IsolationState = "isolated" | "reloading" | "waiting" | "unavailable";

export function isolationState(): IsolationState {
  return (window as unknown as { oryxIsolation?: IsolationState }).oryxIsolation ?? (self.crossOriginIsolated ? "isolated" : "unavailable");
}

// Resolves once it's worth starting large downloads. While sw-register.js is about to reload the
// page (first visit, or after a hard refresh) anything downloading would be thrown away, so wait;
// in every other case go at once.
export function isolationSettled(): Promise<void> {
  const state = isolationState();
  if (state === "isolated" || state === "unavailable" || location.port === "5173") return Promise.resolve();
  return new Promise((resolve) => {
    const check = () => {
      const now = isolationState();
      if (now === "isolated" || now === "unavailable") {
        removeEventListener("oryx-isolation", check);
        resolve();
      }
    };
    addEventListener("oryx-isolation", check);
    setTimeout(resolve, 6000); // a safety net if the service worker never takes control
  });
}

/** Reloads so the service worker can serve (and isolate) the page: the "use all cores" action. */
export function reloadForThreads() {
  try {
    sessionStorage.removeItem("oryx-sw-reloads"); // an explicit request: allow the reload again
  } catch {
    /* storage blocked */
  }
  location.reload();
}
