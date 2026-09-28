import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

test("benchmarks page: interactive charts, and the lab runs on this device", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("benchmarks");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Faster than the libraries");
  await expect(page.locator(".stats dd").first()).toHaveText(/^1\.\d\d×$/);

  // Hovering a point compares the libraries at the same ef.
  const chart = page.getByRole("img", { name: /Recall versus queries per second on SIFT1M/ });
  await chart.scrollIntoViewIfNeeded();
  const box = (await chart.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.55, box.y + box.height * 0.5);
  for (let x = 0.3; x < 0.95 && !(await page.locator(".tooltip").isVisible()); x += 0.05) {
    for (let y = 0.1; y < 0.9 && !(await page.locator(".tooltip").isVisible()); y += 0.1) {
      await page.mouse.move(box.x + box.width * x, box.y + box.height * y);
    }
  }
  await expect(page.locator(".tooltip")).toContainText("ef =");
  await expect(page.locator(".tooltip .ratio")).toContainText("Oryx");
  if (shots) await page.screenshot({ path: `${shots}/bench-chart.png` });

  // Quantization tabs.
  await page.getByRole("tab", { name: /CLIP/ }).click();
  await expect(page.getByRole("table", { name: /CLIP/ })).toContainText("0.9981");

  // The lab: the real engine in WebAssembly, threads and SIMD, on this machine.
  const lab = page.locator("#lab");
  await lab.scrollIntoViewIfNeeded();
  await expect(lab.locator(".env")).toContainText("SIMD128", { timeout: 30_000 });
  await expect(lab.locator(".env")).toContainText("threads");
  await lab.locator("label", { hasText: "20k" }).click();
  await lab.getByRole("button", { name: "Run on this device" }).click();
  await expect(lab.locator('.stages li[data-state="done"]')).toHaveCount(6, { timeout: 150_000 });
  await expect(lab.locator("tbody tr")).toHaveCount(10);
  const recalls = await lab.locator("tbody tr td:nth-child(2)").allTextContents();
  expect(Number(recalls.at(-1))).toBeGreaterThan(0.99);

  // Phones: nothing may scroll sideways.
  await page.setViewportSize({ width: 360, height: 780 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360);
  await page.setViewportSize({ width: 1440, height: 900 });

  if (shots) {
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${shots}/bench-dark.png`, fullPage: true });
    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${shots}/bench-light.png`, fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${shots}/bench-mobile.png`, fullPage: true });
  }
  expect(errors).toEqual([]);
});
