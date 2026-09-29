import { defineConfig, devices } from "@playwright/test";

// By default serves the production build with the isolation headers GitHub Pages gets from the
// service worker. Set E2E_BASE_URL to test a deployed site instead (e.g. the live Pages URL).
const deployed = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  // These drive a real browser, a fake webcam and ~150 MB of model downloads on a shared CI runner:
  // retry once there so a timing hiccup doesn't block a deploy (a real failure fails twice).
  retries: process.env.CI ? 1 : 0,
  use: { baseURL: deployed ?? "http://localhost:4173/Oryx/", ...devices["Desktop Chrome"] },
  webServer: deployed
    ? undefined
    : {
        command: "npm run preview -- --port 4173 --strictPort",
        url: "http://localhost:4173/Oryx/",
        reuseExistingServer: !process.env.CI,
      },
});
