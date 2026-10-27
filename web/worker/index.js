// epzero.dev on Cloudflare Workers. Everything is a static asset (dist/, served at the edge with the
// headers in public/_headers) except the CLIP models: at 64 and 89 MB they exceed the static-asset
// limit (25 MiB a file), so they live in R2 and this Worker serves them from the same origin, which
// keeps the page cross-origin isolated without CORS, and sends them brotli-compressed when it can.
// Only /models/* runs this code (run_worker_first in wrangler.jsonc); every other request goes
// straight to the assets.

const TYPES = { onnx: "application/octet-stream", json: "application/json", txt: "text/plain; charset=utf-8" };

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname.startsWith("/models/")) return model(request, env, ctx, url);
    return env.ASSETS.fetch(request);
  },
};

async function model(request, env, ctx, url) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    return new Response("Method not allowed", { status: 405, headers: { Allow: "GET, HEAD" } });
  }
  const key = decodeURIComponent(url.pathname.slice(1)); // "models/..."
  if (key.includes("..")) return new Response("Not found", { status: 404 });

  // Whole-file GETs are served from the edge cache when possible: the models never change under a
  // given path, and each R2 read is a billed operation.
  const ranged = request.headers.has("Range");
  const whole = request.method === "GET" && !ranged;
  if (whole && key.endsWith(".onnx") && /\bbr\b/.test(request.headers.get("Accept-Encoding") ?? "")) {
    const compressed = await brotli(request, env, key);
    if (compressed) return compressed;
  }
  const cache = caches.default;
  if (whole) {
    const hit = await cache.match(request);
    if (hit) return hit;
  }

  const object = await env.MODELS.get(key, ranged ? { range: request.headers, onlyIf: request.headers } : { onlyIf: request.headers });
  if (object === null) return new Response("Not found", { status: 404, headers: { "Cache-Control": "no-store" } });

  const headers = new Headers();
  object.writeHttpMetadata(headers);
  if (!headers.has("Content-Type")) headers.set("Content-Type", TYPES[key.split(".").pop()] ?? "application/octet-stream");
  headers.set("ETag", object.httpEtag);
  headers.set("Accept-Ranges", "bytes");
  headers.set("Cache-Control", "public, max-age=2592000");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  headers.set("X-Content-Type-Options", "nosniff");

  if (!("body" in object)) return new Response(null, { status: 304, headers }); // If-None-Match matched
  if (ranged && object.range) {
    const start = object.range.offset ?? 0;
    const length = object.range.length ?? object.size - start;
    headers.set("Content-Range", `bytes ${start}-${start + length - 1}/${object.size}`);
    headers.set("Content-Length", String(length));
    return new Response(request.method === "HEAD" ? null : object.body, { status: 206, headers });
  }
  headers.set("Content-Length", String(object.size));
  headers.set("Vary", "Accept-Encoding");
  const response = new Response(request.method === "HEAD" ? null : object.body, { headers });
  if (whole) ctx.waitUntil(cache.put(request, response.clone()));
  return response;
}

// A model as brotli, for browsers that accept it: CI stores <file>.br next to each model, about 37%
// smaller (the int8 weights cluster near zero). Sent as is (encodeBody "manual"), and the browser
// decompresses it. Read from R2 each time rather than the edge cache, whose entries for the plain
// file would otherwise answer first. Null when there's no compressed copy yet.
async function brotli(request, env, key) {
  const object = await env.MODELS.get(`${key}.br`, { onlyIf: request.headers });
  if (object === null) return null;
  const headers = new Headers();
  headers.set("Content-Type", "application/octet-stream");
  headers.set("Content-Encoding", "br");
  headers.set("Vary", "Accept-Encoding");
  headers.set("ETag", object.httpEtag);
  headers.set("Cache-Control", "public, max-age=2592000");
  headers.set("Cross-Origin-Resource-Policy", "same-origin");
  headers.set("X-Content-Type-Options", "nosniff");
  if (!("body" in object)) return new Response(null, { status: 304, headers });
  headers.set("Content-Length", String(object.size));
  return new Response(object.body, { headers, encodeBody: "manual" });
}
