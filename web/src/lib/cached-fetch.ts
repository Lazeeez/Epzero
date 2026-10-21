// Reads a data file straight from the service worker's cache when it's there, and from the
// network otherwise. Going through the service worker's fetch handler instead means its single
// thread relays every byte, so on a repeat visit the dataset queued behind the model downloads it
// was also relaying (measured: 5 MB of metadata took over 5 s to arrive).
export const DATA_CACHE = "oryx-data-v1"; // must match src/sw/sw-template.js

export async function cachedFetch(url: string, { fresh = false } = {}): Promise<Response> {
  // Small files that say what the others are (the manifest) always check the network first; the
  // service worker falls back to its cached copy when offline.
  if (fresh) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response;
  }
  try {
    const hit = await (await caches.open(DATA_CACHE)).match(url, { ignoreSearch: true, ignoreVary: true });
    if (hit) return hit;
  } catch {
    // No Cache API here (e.g. some private modes): fall through to the network.
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

/** Drops cached files under `base` that `keep` no longer names, such as an older vector format. */
export async function pruneCache(base: string, keep: string[]): Promise<void> {
  try {
    const root = new URL(base, location.href).href;
    const wanted = new Set(keep.map((file) => new URL(file, root).href));
    const cache = await caches.open(DATA_CACHE);
    for (const request of await cache.keys()) {
      const url = request.url.split("?")[0];
      if (url.startsWith(root) && !wanted.has(url)) await cache.delete(request);
    }
  } catch {
    // No Cache API, or it failed: nothing to tidy.
  }
}
