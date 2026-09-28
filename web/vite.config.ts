import { defineConfig } from "vite";

// WebAssembly threads need SharedArrayBuffer, which browsers only expose on cross-origin isolated
// pages. The dev and preview servers send the isolation headers directly; on GitHub Pages (which
// cannot set headers) public/coi-serviceworker.js adds them.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  base: process.env.ORYX_BASE ?? "/Oryx/",
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: "es" },
  build: { target: "es2022" },
});
