import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

test("benchmarks page: interactive charts, and the lab runs on this device", async ({ page }) => {
  const errors: string[] = [];
  // A photo since deleted from Flickr 404s; the page shows a placeholder, but the browser still logs it.
  page.on("console", (m) => m.type() === "error" && !m.location().url.includes("staticflickr.com") && errors.push(m.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("benchmarks");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Faster than hnswlib and FAISS");
  await expect(page.locator(".stats dd").first()).toHaveText(/^1\.\d\d×$/);

  // Against the field: switching datasets re-ranks the bars, fastest first.
  const field = page.locator("section[aria-labelledby=field-title]");
  await field.scrollIntoViewIfNeeded();
  await expect(field.locator(".row")).toHaveCount(8);
  await field.getByRole("radio", { name: "DEEP10M" }).click();
  await expect(field.getByRole("radio", { name: "DEEP10M" })).toHaveAttribute("aria-checked", "true");
  await expect(field.locator(".meta")).toContainText("9,990,000");
  await expect(field.locator(".row.epzero")).toHaveCount(2);

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
  await expect(page.locator(".tooltip .ratio")).toContainText("Epzero");
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
  // Configured automatically (the default): the plan, with a reason per choice, and the sweep includes
  // the calibrated beam width, the highlighted operating point.
  await expect(lab.getByRole("radio", { name: /Configure automatically/ })).toHaveAttribute("aria-checked", "true");
  await lab.getByRole("button", { name: "Run on this device" }).click();
  await expect(lab.locator('.stages li[data-state="done"]')).toHaveCount(6, { timeout: 150_000 });
  await expect(lab.locator(".plan")).toContainText(/Calibrated ef \d+ for recall 0\.95/);
  await expect(lab.locator(".plan dt").first()).toBeVisible();
  await expect(lab.locator("tbody tr.op")).toHaveCount(1);
  expect(Number(await lab.locator("tbody tr.op td:nth-child(2)").textContent())).toBeGreaterThan(0.9);
  await expect(lab.locator(".tile.hero")).toContainText("calibrated beam width");
  expect(await lab.locator("tbody tr").count()).toBeGreaterThanOrEqual(10);
  if (shots) await lab.screenshot({ path: `${shots}/lab-auto.png` });

  // By hand: M and efConstruction as chosen, ten beam widths, no plan.
  await lab.getByRole("radio", { name: /By hand/ }).click();
  await lab.getByRole("button", { name: "Run on this device" }).click();
  await expect(lab.locator('.stages li[data-state="done"]')).toHaveCount(6, { timeout: 150_000 });
  await expect(lab.locator(".plan")).toHaveCount(0);
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
