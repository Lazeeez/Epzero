import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

test("navigator game: hop through the graph to the target, then compare with Oryx", async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("play");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("out-navigate");
  const start = page.getByRole("button", { name: "Start navigating" });
  await expect(start).toBeVisible({ timeout: 150_000 });
  await start.click();

  const options = page.locator(".options button");
  await expect(options.first()).toBeVisible({ timeout: 30_000 });
  await expect(page.locator(".status")).toContainText("hop 0/25");
  if (shots) {
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${shots}/play.png` });
  }

  // Follow Oryx's hint (the neighbour closest to the target) each hop: greedy routing on the base
  // layer. It must finish the round, won or lost, and show the comparison.
  const result = page.locator(".result");
  for (let hop = 0; hop < 26 && !(await result.isVisible()); hop++) {
    await page.getByRole("button", { name: /Ask Oryx/ }).click();
    await page.locator(".options button.hinted").click();
  }
  await expect(result).toBeVisible();
  await expect(result).toContainText("Oryx");
  const sides = result.locator(".compare dl");
  await expect(sides.nth(0).locator("dd").first()).toHaveText(/\d+ hops/);
  await expect(sides.nth(1).locator("dd").first()).toHaveText(/^\d+ \+ \d+$/); // express hops + expansions
  if (shots) {
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${shots}/play-result.png` });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("button", { name: "Play again" }).click();
    await expect(options.first()).toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${shots}/play-mobile.png`, fullPage: true });
  }
  await page.setViewportSize({ width: 360, height: 780 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360);
  expect(errors).toEqual([]);
});
