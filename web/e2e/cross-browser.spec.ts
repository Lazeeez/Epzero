import { expect, test } from "@playwright/test";

// The core journeys in Firefox and WebKit (Safari's engine); Chromium runs the full suites.
test("landing, text search, similar photos, the game and the lab all work", async ({ page, browserName }) => {
  test.setTimeout(600_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 900 });

  await page.goto("./");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("A vector database");
  await expect(page.locator(".telemetry")).toContainText(/Index (ready|built)/, { timeout: 200_000 });

  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 200_000 });
  const input = panel.getByRole("searchbox", { name: "Describe a photo to search for" });
  await expect(input).toHaveAttribute("placeholder", "Describe a photo…", { timeout: 300_000 });
  await input.fill("a red apple");
  await input.press("Enter");
  await expect(panel.locator(".results h2")).toContainText("a red apple", { timeout: 60_000 });
  const captions = await panel.locator(".caption > span:first-child").allTextContents();
  expect(captions.filter((c) => /apple/i.test(c)).length, `${browserName}: ${captions.join(" | ")}`).toBeGreaterThanOrEqual(4);
  await panel.locator(".result").nth(2).click();
  await expect(panel.locator(".results h2")).toContainText("Photos like");

  await page.goto("play");
  await page.getByRole("button", { name: "Start navigating" }).click({ timeout: 150_000 });
  await expect(page.locator(".options button").first()).toBeVisible({ timeout: 30_000 });
  for (let hop = 0; hop < 3 && !(await page.locator(".result").isVisible()); hop++) {
    await page.getByRole("button", { name: /Ask Epzero/ }).click();
    await page.locator(".options button.hinted").click();
  }

  await page.goto("benchmarks");
  const lab = page.locator("#lab");
  await lab.scrollIntoViewIfNeeded();
  await expect(lab.locator(".env")).toContainText("SIMD128", { timeout: 60_000 });
  await lab.locator("label", { hasText: "20k" }).click();
  await lab.getByRole("button", { name: "Run on this device" }).click();
  await expect(lab.locator('.stages li[data-state="done"]')).toHaveCount(6, { timeout: 300_000 });
  const recalls = await lab.locator("tbody tr td:nth-child(2)").allTextContents();
  expect(Number(recalls.at(-1))).toBeGreaterThan(0.99);
  console.log(`${browserName}: lab ran on "${(await lab.locator(".env").textContent())?.replace(/\s+/g, " ").trim()}"`);
  expect(errors, errors.join("\n")).toEqual([]);
});
