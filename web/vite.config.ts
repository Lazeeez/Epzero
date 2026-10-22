import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { loadArticle, type Article } from "./scripts/article";
import { PAGES, SITE, type Page } from "./src/lib/pages";
import { defineConfig, type Plugin } from "vite";

// WebAssembly threads need SharedArrayBuffer, which browsers only expose on cross-origin isolated
// pages. The dev and preview servers send the isolation headers directly, as Cloudflare does in
// production (public/_headers); on a host that can't set headers, the site's service worker adds them.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// The site is a single-page app with client-side routes. Give every route its own index.html so a
// deep link (e.g. /try) loads with a 200 on a static host, and keep a 404.html fallback. Each page's
// HTML carries its own title, description, link-preview tags and canonical address (so search
// engines index every page on its own, without running the app), plus a no-JavaScript fallback
// with its heading and links to the others; the 404 page asks not to be indexed. The sitemap is
// written from the same list (src/lib/pages.ts).
const ROUTES = ["try", "play", "cluster", "benchmarks", "blog"] as const;
const escapeHtml = (text: string) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function pageHtml(html: string, page: Page, indexed: boolean, article?: Article): string {
  const url = SITE + page.path;
  const title = escapeHtml(page.title);
  const description = escapeHtml(page.description);
  const swaps: [RegExp, string][] = [
    [/<title>[^<]*<\/title>/, `<title>${title}</title>`],
    [/<meta\s+name="description"\s+content="[^"]*"\s*\/>/, `<meta name="description" content="${description}" />`],
    [/<meta property="og:title" content="[^"]*" \/>/, `<meta property="og:title" content="${title}" />`],
    [/<meta property="og:description" content="[^"]*" \/>/, `<meta property="og:description" content="${description}" />`],
    [/<meta property="og:url" content="[^"]*" \/>/, `<meta property="og:url" content="${url}" />`],
    [
      /<link rel="canonical" href="[^"]*" \/>/,
      indexed ? `<link rel="canonical" href="${url}" />` : `<meta name="robots" content="noindex" />`,
    ],
    [/<noscript>[\s\S]*?<\/noscript>/, `<noscript>${fallback(page, article)}</noscript>`],
  ];
  // The article page: its text for crawlers that don't run the app, and its structured data.
  if (article) swaps.push([/<\/head>/, `${articleData(page, article)}
  </head>`]);
  for (const [pattern, replacement] of swaps) {
    if (!pattern.test(html)) throw new Error(`index.html: ${pattern} not found`);
    html = html.replace(pattern, () => replacement);
  }
  return html;
}

// What a visitor (or crawler) without JavaScript gets: the page's heading and summary, and links.
function fallback(page: Page, article?: Article): string {
  const links = (["home", ...ROUTES] as const)
    .map((name) => `<a href="/${PAGES[name].path}">${escapeHtml(name === "home" ? "Home" : PAGES[name].title.split(" · ")[0])}</a>`)
    .join(" · ");
  return (
    `<main style="max-width:720px;margin:96px auto;padding:0 24px;font:16px/1.6 system-ui,sans-serif;color:#d7dae0">` +
    `<h1>${escapeHtml(page.heading)}</h1>` +
    (article
      ? `<p><em>${escapeHtml(article.subtitle)}</em></p>${article.html}`
      : `<p>${escapeHtml(page.description)}</p><p>The live demo runs a C++ search engine in your browser, so it needs JavaScript.</p>`) +
    `<nav>${links}</nav></main>`
  );
}

// Structured data for the article (schema.org BlogPosting), beside the site-wide block.
function articleData(page: Page, article: Article): string {
  const data = {
    "@context": "https://schema.org",
    "@type": "BlogPosting",
    headline: article.title,
    description: page.description,
    url: SITE + page.path,
    mainEntityOfPage: SITE + page.path,
    image: `${SITE}og.jpg`,
    wordCount: article.words,
    inLanguage: "en",
    author: { "@type": "Person", name: "Lajat", url: SITE },
    publisher: { "@type": "Person", name: "Lajat", url: SITE },
    about: ["Vector database", "HNSW", "Approximate nearest neighbor search", "Performance engineering"],
  };
  return `<script type="application/ld+json">${JSON.stringify(data)}</script>`;
}

// The article as a module for the app ("virtual:article"), converted from Markdown at build time.
function articleModule(): Plugin {
  const id = "virtual:article";
  return {
    name: "epzero-article",
    resolveId: (source) => (source === id ? "\0" + id : null),
    async load(resolved) {
      if (resolved !== "\0" + id) return null;
      const { title, subtitle, html, words } = await loadArticle();
      return `export default ${JSON.stringify({ title, subtitle, html, words })};`;
    },
  };
}

function staticRoutes(): Plugin {
  let outDir = "dist";
  return {
    name: "epzero-static-routes",
    apply: "build",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    async closeBundle() {
      const index = join(outDir, "index.html");
      const html = await readFile(index, "utf8");
      await writeFile(index, pageHtml(html, PAGES.home, true));
      const article = await loadArticle();
      for (const route of ROUTES) {
        await mkdir(join(outDir, route), { recursive: true });
        await writeFile(join(outDir, route, "index.html"), pageHtml(html, PAGES[route], true, route === "blog" ? article : undefined));
      }
      await writeFile(join(outDir, "404.html"), pageHtml(html, PAGES.notfound, false));
      const today = new Date().toISOString().slice(0, 10);
      const urls = (["home", ...ROUTES] as const)
        .map((name, i) => `  <url><loc>${SITE}${PAGES[name].path}</loc><lastmod>${today}</lastmod><priority>${i === 0 ? "1.0" : "0.8"}</priority></url>`)
        .join("\n");
      await writeFile(
        join(outDir, "sitemap.xml"),
        `<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n${urls}\n</urlset>\n`,
      );
    },
  };
}

// src/boot.js must run before first paint. It's inlined (a separate file would be one more request
// blocking the first paint), and the Content Security Policy allows exactly that script by its hash.
function inlineBoot(): Plugin {
  return {
    name: "epzero-inline-boot",
    transformIndexHtml: {
      order: "pre", // before Vite looks at the script tags
      async handler(html) {
        const code = (await readFile(new URL("./src/boot.js", import.meta.url), "utf8")).trim();
        const hash = createHash("sha256").update(code).digest("base64");
        const tag = /<script src="[^"]*boot\.js"><\/script>/;
        if (!tag.test(html) || !html.includes("script-src 'self'")) throw new Error("index.html: boot script or CSP not found");
        return html.replace(tag, () => `<script>${code}</script>`).replace("script-src 'self'", `script-src 'self' 'sha256-${hash}'`);
      },
    },
  };
}

// transformers.js references its own copy of ONNX Runtime's wasm (27 MB), which Vite emits. The
// encoders load the self-hosted build in public/ort instead (wasmPaths), so drop the unused copy.
function dropBundledOrt(): Plugin {
  return {
    name: "epzero-drop-bundled-ort",
    apply: "build",
    generateBundle(_, bundle) {
      for (const name of Object.keys(bundle)) if (/ort-wasm[^/]*\.wasm$/.test(name)) delete bundle[name];
    },
  };
}

// The shell precached at install: pages, code, styles and Latin fonts. The encoder workers and
// other font subsets are cached on first use instead, so installing stays small.
const precached = (name: string) =>
  !/worker|\.wasm$/.test(name) && (!name.endsWith(".woff2") || /-latin-wght/.test(name));

// Emits sw.js from src/sw/sw-template.js with this build's precache list and version, so every
// deploy gets a fresh shell cache while large data/model caches survive.
function serviceWorker(): Plugin {
  return {
    name: "epzero-service-worker",
    apply: "build",
    async generateBundle(_, bundle) {
      const files = Object.keys(bundle).filter((name) => !name.endsWith(".map") && name !== "sw.js");
      const precache = ["", "index.html", ...files.filter((name) => name !== "index.html" && precached(name))];
      // The version covers the bundle's (hashed) names and the engine's WebAssembly files, whose
      // names never change: a C++-only change must still replace the cached engine.
      const hash = createHash("sha256").update(files.sort().join("\n"));
      for (const variant of ["mt", "st"]) {
        for (const file of ["epzero.js", "epzero.wasm"]) {
          try {
            hash.update(await readFile(new URL(`./public/wasm/${variant}/${file}`, import.meta.url)));
          } catch {
            /* not built yet (e.g. a docs-only build): nothing to version */
          }
        }
      }
      const build = hash.digest("hex").slice(0, 12);
      const template = await readFile(new URL("./src/sw/sw-template.js", import.meta.url), "utf8");
      this.emitFile({
        type: "asset",
        fileName: "sw.js",
        source: template.replace("__BUILD__", build).replace("__PRECACHE__", JSON.stringify(precache)),
      });
    },
  };
}

export default defineConfig({
  // The site lives at the root of epzero.dev; EPZERO_BASE serves it under a path instead.
  base: process.env.EPZERO_BASE ?? "/",
  plugins: [svelte(), articleModule(), inlineBoot(), dropBundledOrt(), staticRoutes(), serviceWorker()],
  // The benchmark pages import their numbers straight from bench/results at the repo root.
  server: { headers: isolation, fs: { allow: [".."] } },
  // EPZERO_NO_ISOLATION serves without the isolation headers, for the service worker's own tests.
  preview: { headers: process.env.EPZERO_NO_ISOLATION ? {} : isolation },
  worker: { format: "es" },
  build: {
    target: "es2022",
    // three.js and the 3D scene load only with the pages that need them.
    chunkSizeWarningLimit: 800,
  },
});
