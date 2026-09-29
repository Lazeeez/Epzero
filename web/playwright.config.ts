import { defineConfig, devices } from "@playwright/test";

// Two local servers for the production build:
//   - 4173 sends the cross-origin isolation headers itself (most tests: isolation is a given);
//   - 4174 sends none, like GitHub Pages, so the service worker must provide isolation
//     (isolation.spec.ts tests exactly that).
// Set E2E_BASE_URL to test a deployed site instead (e.g. the live Pages URL).
const deployed = process.env.E2E_BASE_URL;

export default defineConfig({
  testDir: "e2e",
  timeout: 180_000,
  // These drive a real browser, a fake webcam and ~150 MB of model downloads on a shared CI runner:
  // retry once there so a timing hiccup doesn't block a deploy (a real failure fails twice).
  retries: process.env.CI ? 1 : 0,
  use: devices["Desktop Chrome"],
  projects: [
    {
      name: "chromium",
      testIgnore: /isolation\.spec\.ts/,
      use: { baseURL: deployed ?? "http://localhost:4173/Oryx/" },
    },
    {
      name: "pages-like",
      testMatch: /isolation\.spec\.ts/,
      use: { baseURL: deployed ?? "http://localhost:4174/Oryx/" },
    },
  ],
  webServer: deployed
    ? undefined
    : [
        {
          command: "npm run preview -- --port 4173 --strictPort",
          url: "http://localhost:4173/Oryx/",
          reuseExistingServer: !process.env.CI,
        },
        {
          command: "npm run preview -- --port 4174 --strictPort",
          url: "http://localhost:4174/Oryx/",
          env: { ORYX_NO_ISOLATION: "1" },
          reuseExistingServer: !process.env.CI,
        },
      ],
});
