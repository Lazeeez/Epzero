import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

// A phone: touch input, no hover. Tapping a star must show its photo, and searching from it must work.
test.use({ viewport: { width: 412, height: 915 }, hasTouch: true, isMobile: true });

test("touch: tap a star for its photo card, search from it, hide and show the panel", async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 150_000 });

  // Hide the panel to explore full screen.
  await page.getByRole("button", { name: "Hide the search panel" }).click();
  await expect(panel).toHaveClass(/collapsed/);
  await page.getByRole("button", { name: "Auto-rotate" }).click(); // hold the camera still

  // Tap around the graph until a star is hit.
  const card = page.getByRole("dialog", { name: "Selected photo" });
  const { width, height } = page.viewportSize()!;
  outer: for (let y = 0.3; y <= 0.75; y += 0.05) {
    for (let x = 0.2; x <= 0.8; x += 0.05) {
      await page.touchscreen.tap(width * x, height * y);
      if (await card.isVisible()) break outer;
    }
  }
  await expect(card).toBeVisible();
  if (shots) await page.screenshot({ path: `${shots}/touch-card.png` });
  const id = Number((await card.locator(".info span").textContent())!.match(/#(\d+)/)![1]);
  await card.getByRole("button", { name: "Find similar" }).click();
  await expect(panel).not.toHaveClass(/collapsed/);
  // The view toolbar sits above the re-opened panel, never on top of it.
  await expect
    .poll(async () => {
      const bar = (await page.getByRole("toolbar", { name: "View controls" }).boundingBox())!;
      const box = (await panel.boundingBox())!;
      return bar.y + bar.height <= box.y + 1;
    })
    .toBe(true);
  await expect(panel.locator(".results h2")).toContainText(`#${id}`);

  // Layer isolation and zoom work from the toolbar.
  await page.getByRole("button", { name: /isolate a layer/ }).click();
  await expect(page.getByRole("button", { name: /Showing layer 0/ })).toBeVisible();
  await page.getByRole("button", { name: "Zoom in" }).click();

  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(412);
  if (shots) {
    await page.waitForTimeout(1200);
    await page.screenshot({ path: `${shots}/touch-layer.png` });
  }
  expect(errors).toEqual([]);
});
