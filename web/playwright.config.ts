import { defineConfig, devices } from "@playwright/test";

// Serves the production build with the same isolation headers GitHub Pages gets from the
// service worker, then drives a full benchmark run in Chromium.
export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  use: { baseURL: "http://localhost:4173/Oryx/", ...devices["Desktop Chrome"] },
  webServer: {
    command: "npm run preview -- --port 4173 --strictPort",
    url: "http://localhost:4173/Oryx/",
    reuseExistingServer: !process.env.CI,
  },
});
