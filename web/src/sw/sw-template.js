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
        Promise.all(
          keys
            // Old builds' shells, data/image caches from an older version of their name, and the
            // second copy of the models transformers.js used to keep (this worker caches them).
            .filter(
              (k) =>
                k === "transformers-cache" ||
                (/^oryx-(shell|data|images)-/.test(k) && ![SHELL_CACHE, DATA_CACHE, IMAGE_CACHE].includes(k)),
            )
            .map((k) => caches.delete(k)),
        ),
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

// Same-origin files are re-requested as a plain GET rather than by passing the page's request
// through: Safari (WebKit) rejects a service worker's fetch of a module worker script's original
// request ("FetchEvent.respondWith received an error: Load failed"), which stopped the search
// engine's worker from starting on every visit after the first.
function plain(request) {
  return new URL(request.url).origin === self.location.origin
    ? new Request(request.url, { credentials: "same-origin" })
    : request;
}

// Cached entries match by URL alone. The server marks responses "Vary: Origin", and a module
// worker's request carries an Origin header the stored one doesn't: honouring Vary made the engine's
// own files miss the cache for the worker, so the site failed offline. For these static files the
// header makes no difference to the content.
const MATCH = { ignoreSearch: true, ignoreVary: true };

// Storing can fail (storage full, private browsing): the response is still served.
const store = (cache, request, response) => cache.put(request, response).catch(() => {});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const hit = await cache.match(request, MATCH);
  if (hit) return hit;
  const response = await fetch(plain(request));
  if (response.ok) store(cache, request, response.clone());
  return response;
}

// Photo thumbnails: cached like other files, but the cache is capped (oldest dropped first) so
// browsing thousands of photos can't fill the device, and a retry (?retry=n) goes to the network.
const IMAGE_LIMIT = 800;
let imagesStored = 0;
// The photo goes to the page as soon as it arrives; storing it and trimming the cache happen
// afterwards (kept alive by waitUntil), so a page of new results never waits on cache writes.
async function cachedImage(request, event) {
  const cache = await caches.open(IMAGE_CACHE);
  const retry = new URL(request.url).searchParams.has("retry");
  const hit = !retry && (await cache.match(request, MATCH));
  if (hit) return hit;
  const response = await fetch(request);
  if (response.ok) event.waitUntil(storeImage(cache, request.url.split("?")[0], response.clone()));
  return response;
}
async function storeImage(cache, key, response) {
  await store(cache, key, response);
  if (++imagesStored % 50 === 0) {
    const keys = await cache.keys();
    await Promise.all(keys.slice(0, Math.max(0, keys.length - IMAGE_LIMIT)).map((key) => cache.delete(key)));
  }
}

async function networkFirst(request, cacheName, fallbackToShell) {
  const cache = await caches.open(cacheName);
  try {
    // Navigations must keep their own request (mode "navigate", redirects); other files go plain.
    const response = await fetch(request.mode === "navigate" ? request : plain(request));
    if (response.ok) store(cache, request, response.clone());
    return response;
  } catch {
    const hit = (await cache.match(request, MATCH)) || (fallbackToShell && (await cache.match(BASE + "index.html", MATCH)));
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
    if (request.mode === "navigate") response = networkFirst(request, SHELL_CACHE, true);
    // The dataset manifest is tiny and says what the data files are (e.g. a new prebuilt graph):
    // always check for a fresh one, falling back to the cached copy offline.
    else if (/^data\/.*manifest\.json$/.test(path)) response = networkFirst(request, DATA_CACHE, false);
    else if (/^(data|models|ort)\//.test(path)) response = cacheFirst(request, DATA_CACHE);
    else response = cacheFirst(request, SHELL_CACHE); // hashed assets, wasm, fonts, icons
    event.respondWith(response.then(isolate));
    return;
  }

  // Photo thumbnails (Flickr sends CORS headers): keep the ones already seen for offline use. If the
  // cache itself is unusable, the photo still loads straight from the network.
  if (url.hostname.endsWith("staticflickr.com") && request.mode === "cors") {
    event.respondWith(cachedImage(request, event).catch(() => fetch(request)));
  }
});
