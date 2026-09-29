import { createHash } from "node:crypto";
import { copyFile, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig, type Plugin } from "vite";

// WebAssembly threads need SharedArrayBuffer, which browsers only expose on cross-origin isolated
// pages. The dev and preview servers send the isolation headers directly; on GitHub Pages (which
// cannot set headers) the site's service worker adds them.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

// The site is a single-page app with client-side routes. Give every route its own index.html so a
// deep link (e.g. /Oryx/try) loads with a 200 on a static host, and keep a 404.html fallback.
const ROUTES = ["try", "play", "benchmarks"];
function staticRoutes(): Plugin {
  let outDir = "dist";
  return {
    name: "oryx-static-routes",
    apply: "build",
    configResolved(config) {
      outDir = config.build.outDir;
    },
    async closeBundle() {
      const index = join(outDir, "index.html");
      for (const route of ROUTES) {
        await mkdir(join(outDir, route), { recursive: true });
        await copyFile(index, join(outDir, route, "index.html"));
      }
      await copyFile(index, join(outDir, "404.html"));
    },
  };
}

// transformers.js references its own copy of ONNX Runtime's wasm (27 MB), which Vite emits. The
// encoders load the self-hosted build in public/ort instead (wasmPaths), so drop the unused copy.
function dropBundledOrt(): Plugin {
  return {
    name: "oryx-drop-bundled-ort",
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
    name: "oryx-service-worker",
    apply: "build",
    async generateBundle(_, bundle) {
      const files = Object.keys(bundle).filter((name) => !name.endsWith(".map") && name !== "sw.js");
      const precache = ["", "index.html", ...files.filter((name) => name !== "index.html" && precached(name))];
      // The version covers the bundle's (hashed) names and the engine's WebAssembly files, whose
      // names never change: a C++-only change must still replace the cached engine.
      const hash = createHash("sha256").update(files.sort().join("\n"));
      for (const variant of ["mt", "st"]) {
        for (const file of ["oryx.js", "oryx.wasm"]) {
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
  base: process.env.ORYX_BASE ?? "/Oryx/",
  plugins: [svelte(), dropBundledOrt(), staticRoutes(), serviceWorker()],
  // The benchmark pages import their numbers straight from bench/results at the repo root.
  server: { headers: isolation, fs: { allow: [".."] } },
  // ORYX_NO_ISOLATION serves like GitHub Pages (no headers), for the service worker's own tests.
  preview: { headers: process.env.ORYX_NO_ISOLATION ? {} : isolation },
  worker: { format: "es" },
  build: {
    target: "es2022",
    // three.js and the 3D scene load only with the pages that need them.
    chunkSizeWarningLimit: 800,
  },
});
