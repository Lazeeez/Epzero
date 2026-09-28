/* Oryx service worker (generated at build time from src/sw/sw-template.js).
 *
 * 1. Cross-origin isolation: WebAssembly threads need SharedArrayBuffer, which browsers only enable
 *    on pages served with COOP/COEP headers. GitHub Pages cannot send headers, so this worker adds
 *    them to every same-origin response.
 * 2. Offline: the app shell is precached at install; the engine, dataset and models are cached the
 *    first time they are used. After one visit the whole demo works without a network.
 */
const BUILD = "__BUILD__";
const PRECACHE = __PRECACHE__;
const SHELL_CACHE = `oryx-shell-${BUILD}`;
// Large, rarely-changing files keep their own cache across deploys (bump the version to refresh).
const DATA_CACHE = "oryx-data-v1";
const IMAGE_CACHE = "oryx-images-v1";
const BASE = new URL(self.registration.scope).pathname;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(SHELL_CACHE)
      .then((cache) => cache.addAll(PRECACHE.map((path) => BASE + path)))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((k) => k.startsWith("oryx-shell-") && k !== SHELL_CACHE).map((k) => caches.delete(k))),
      )
      .then(() => self.clients.claim()),
  );
});

function isolate(response) {
  if (response.status === 0) return response; // opaque: headers cannot be added
  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request, { ignoreSearch: true });
  if (hit) return hit;
  const response = await fetch(request);
  if (response.ok) cache.put(request, response.clone());
  return response;
}

async function networkFirst(request) {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone());
    return response;
  } catch {
    const hit = (await cache.match(request)) || (await cache.match(BASE + "index.html"));
    if (hit) return hit;
    throw new Error("offline and not cached");
  }
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);

  if (url.origin === self.location.origin) {
    if (!url.pathname.startsWith(BASE)) return;
    const path = url.pathname.slice(BASE.length);
    let response;
    if (request.mode === "navigate") response = networkFirst(request);
    else if (/^(data|models|ort)\//.test(path)) response = cacheFirst(request, DATA_CACHE);
    else response = cacheFirst(request, SHELL_CACHE); // hashed assets, wasm, fonts, icons
    event.respondWith(response.then(isolate));
    return;
  }

  // Photo thumbnails (Flickr sends CORS headers): keep the ones already seen for offline use.
  if (url.hostname.endsWith("staticflickr.com") && request.mode === "cors") {
    event.respondWith(cacheFirst(request, IMAGE_CACHE));
  }
});
