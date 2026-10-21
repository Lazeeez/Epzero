import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

// Automated WCAG 2.1 A/AA checks (axe-core) on every page, in both themes, with the dynamic
// content loaded: search results, the explainer, the game in progress and the lab's results.
const scan = async (page: Page, name: string) => {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    // The WebGL canvas and the explainer's canvas are drawings described by aria-labels.
    .exclude(".stage canvas")
    .analyze();
  const summary = results.violations.map(
    (v) =>
      `${name}: [${v.impact}] ${v.id}: ${v.help}\n    ` +
      v.nodes
        .slice(0, 3)
        .map((n) => `${n.target.join(" ")} (${(n.failureSummary ?? "").split("\n").slice(1).join(" ").trim()}) ${n.html.slice(0, 90)}`)
        .join("\n    "),
  );
  expect(summary, summary.join("\n")).toEqual([]);
};

for (const theme of ["dark", "light"]) {
  test(`accessibility (${theme} theme)`, async ({ page }) => {
    test.setTimeout(400_000);
    await page.addInitScript((t) => {
      try {
        localStorage.setItem("oryx-theme", t);
        localStorage.setItem("oryx-view-hint", "1");
      } catch {
        /* ignore */
      }
    }, theme);
    await page.setViewportSize({ width: 1280, height: 900 });

    const violations: string[] = [];
    page.on("console", (m) => /Content Security Policy/.test(m.text()) && violations.push(m.text()));
    await page.goto("./", { waitUntil: "domcontentloaded" });
    // The inlined theme bootstrap ran before the app (the CSP allows it by hash): the saved theme is
    // already on the page, with nothing blocked.
    expect(await page.evaluate(() => document.documentElement.dataset.theme ?? "dark")).toBe(theme);
    await expect(page.locator(".telemetry")).toContainText(/Index (ready|built)/, { timeout: 150_000 });
    expect(violations).toEqual([]);
    // Reveal every section (they fade in on scroll) so they're all checked at full opacity.
    await page.evaluate(() => document.querySelectorAll("[data-reveal]").forEach((el) => el.classList.add("revealed")));
    await page.locator("#how").scrollIntoViewIfNeeded();
    await expect(page.locator("#how .readout")).toContainText("layers", { timeout: 20_000 });
    await page.waitForTimeout(1200); // transitions settle before contrast is measured
    await scan(page, "home");

    await page.goto("try");
    await expect(page.locator(".result")).toHaveCount(12, { timeout: 150_000 });
    await page.waitForTimeout(1500);
    await scan(page, "try");

    await page.goto("play");
    await page.getByRole("button", { name: "Start navigating" }).click({ timeout: 150_000 });
    await expect(page.locator(".options button").first()).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(800);
    await scan(page, "play");

    await page.goto("benchmarks");
    await page.evaluate(() => document.querySelectorAll("[data-reveal]").forEach((el) => el.classList.add("revealed")));
    const lab = page.locator("#lab");
    await lab.scrollIntoViewIfNeeded();
    await lab.locator("label", { hasText: "20k" }).click();
    await lab.getByRole("button", { name: "Run on this device" }).click({ timeout: 60_000 });
    await expect(lab.locator('.stages li[data-state="done"]')).toHaveCount(6, { timeout: 150_000 });
    await page.waitForTimeout(1200);
    await scan(page, "benchmarks");

    await page.goto("cluster");
    await expect(page.locator(".node")).toHaveCount(4, { timeout: 60_000 });
    await page.evaluate(() => document.querySelectorAll("[data-reveal]").forEach((el) => el.classList.add("revealed")));
    await page.getByRole("button", { name: "Check linearizability" }).click();
    await expect(page.locator(".verdict")).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(1200); // the reveal transitions finish
    await scan(page, "cluster");
  });
}
