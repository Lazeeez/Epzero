import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

test("landing page: live hero, and the HNSW explainer runs real searches", async ({ page }) => {
  test.setTimeout(300_000); // builds the 29,825-photo index twice with screenshots on
  const errors: string[] = [];
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("./");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("A vector database");
  await expect(page).toHaveTitle(/Oryx/);

  // The engine builds behind the hero and starts answering live queries.
  await expect(page.locator(".telemetry")).toContainText(/Index (ready|built)/, { timeout: 150_000 });
  await expect(page.getByRole("complementary", { name: "Live query" })).toBeVisible({ timeout: 15_000 });
  if (shots) {
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `${shots}/home-hero.png` });
  }

  // The live comparison in "The problem" uses a measured query.
  await page.locator("#problem").scrollIntoViewIfNeeded();
  await expect(page.locator("#problem .facts")).toContainText("Faster");

  // Scrollytelling: each step morphs the toy index; the last step takes clicks.
  const how = page.locator("#how");
  await how.scrollIntoViewIfNeeded();
  const readout = how.locator(".readout");
  await expect(readout).toContainText("layers", { timeout: 20_000 });
  const steps = how.locator(".step");
  await expect(steps).toHaveCount(6);
  for (let i = 0; i < 5; i++) {
    await steps.nth(i).evaluate((el) => el.scrollIntoView({ block: "center" }));
    await expect(steps.nth(i)).toHaveClass(/active/);
    if (shots && (i === 1 || i === 3)) {
      await page.waitForTimeout(1600);
      await page.screenshot({ path: `${shots}/home-how-${i}.png` });
    }
  }
  await expect(readout).toContainText("true neighbours found", { timeout: 15_000 });

  await steps.nth(5).evaluate((el) => el.scrollIntoView({ block: "center" }));
  await expect(steps.nth(5)).toHaveClass(/active/);
  const canvas = how.locator("canvas");
  await page.waitForTimeout(1200); // the morph settles
  const box = (await canvas.boundingBox())!;
  await canvas.click({ position: { x: box.width * 0.5, y: box.height * 0.72 } });
  await expect(readout).toContainText("true neighbours found", { timeout: 15_000 });
  // Keyboard alternative: arrows move the query.
  await canvas.focus();
  await page.keyboard.press("Shift+ArrowUp");
  await expect(readout).toContainText("true neighbours found", { timeout: 15_000 });
  await how.getByRole("button", { name: "Flat" }).click();
  await how.getByRole("slider").fill("5");
  await expect(how.locator(".controls output")).toHaveText("5");
  if (shots) {
    await page.waitForTimeout(2000);
    await page.screenshot({ path: `${shots}/home-how-5.png` });
  }

  // Phones: nothing may scroll sideways.
  await page.setViewportSize({ width: 360, height: 780 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360);
  await page.setViewportSize({ width: 1440, height: 900 });

  // Navigating by the nav bar is client-side (the engine survives).
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Benchmarks" }).click();
  await expect(page).toHaveURL(/\/benchmarks$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Faster than the libraries");

  if (shots) {
    await page.goto("./");
    await expect(page.locator(".telemetry")).toContainText(/Index (ready|built)/, { timeout: 150_000 });
    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await page.waitForTimeout(3000);
    await page.screenshot({ path: `${shots}/home-light.png` });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${shots}/home-mobile.png` });
    await page.locator("#how .step").nth(3).evaluate((el) => el.scrollIntoView({ block: "end" }));
    await page.waitForTimeout(2500);
    await page.screenshot({ path: `${shots}/home-how-mobile.png` });
  }
  expect(errors).toEqual([]);
});
