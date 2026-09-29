import { expect, test, type Page } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

async function openCamera(page: Page) {
  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 150_000 });
  await panel.getByRole("tab", { name: "Camera" }).click();
  return panel;
}

// Chrome's fake webcam: an animated test pattern, auto-approved.
test.use({
  launchOptions: { args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
  permissions: ["camera"],
});

test.describe("camera, permission granted", () => {
  test("preview shows frames; capture, live mode and turning off all work", async ({ page }) => {
    test.setTimeout(400_000);
    const errors: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
    await page.setViewportSize({ width: 1440, height: 900 });
    const panel = await openCamera(page);

    await panel.getByRole("button", { name: /Point your camera/ }).click();
    const video = panel.locator("video");
    await expect(video).toBeVisible();
    // The stream is attached and playing: the preview has real frames.
    await expect.poll(() => video.evaluate((v: HTMLVideoElement) => v.videoWidth), { timeout: 10_000 }).toBeGreaterThan(0);
    await expect.poll(() => video.evaluate((v: HTMLVideoElement) => !v.paused)).toBe(true);

    const capture = panel.getByRole("button", { name: "Capture & search" });
    await expect(capture).toBeEnabled({ timeout: 200_000 }); // the CLIP image model has loaded
    await capture.click();
    const heading = panel.locator(".results h2");
    await expect(heading).toContainText("camera frame", { timeout: 30_000 });
    await expect(heading.locator("img")).toHaveJSProperty("complete", true);

    // Live mode keeps searching new frames (the fake camera's pattern changes every frame).
    const firstScore = await panel.locator(".stats").textContent();
    await panel.getByRole("button", { name: "Go live" }).click();
    await expect(panel.getByText("LIVE", { exact: true })).toBeVisible();
    await expect.poll(async () => panel.locator(".stats").textContent(), { timeout: 20_000 }).not.toBe(firstScore);
    if (shots) await page.screenshot({ path: `${shots}/camera-live.png` });
    await panel.getByRole("button", { name: "Live" }).click();
    await expect(panel.getByText("LIVE", { exact: true })).toHaveCount(0);

    // Turning it off releases the camera; switching tabs does too.
    await panel.getByRole("button", { name: "Turn the camera off" }).click();
    await expect(video).toHaveCount(0);
    await expect(panel.getByRole("button", { name: /Point your camera/ })).toBeVisible();
    await panel.getByRole("button", { name: /Point your camera/ }).click();
    await expect
      .poll(() => panel.locator("video").evaluate((v: HTMLVideoElement) => `${v.videoWidth}|${!!v.srcObject}|${v.readyState}`), { timeout: 15_000 })
      .toMatch(/^[1-9]\d*\|true\|[2-4]$/); // restarted: a new stream, attached, with frames
    await panel.getByRole("tab", { name: "Describe" }).click();
    await expect(panel.locator("video")).toHaveCount(0);
    expect(errors).toEqual([]);
  });

  test("phone: camera preview and capture fit and work", async ({ browser }) => {
    test.setTimeout(400_000);
    const context = await browser.newContext({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true, permissions: ["camera"] });
    const page = await context.newPage();
    const panel = await openCamera(page);
    await panel.getByRole("button", { name: /Point your camera/ }).click();
    await expect.poll(() => panel.locator("video").evaluate((v: HTMLVideoElement) => v.videoWidth), { timeout: 10_000 }).toBeGreaterThan(0);
    const capture = panel.getByRole("button", { name: "Capture & search" });
    await expect(capture).toBeEnabled({ timeout: 200_000 });
    await capture.tap();
    await expect(panel.locator(".results h2")).toContainText("camera frame", { timeout: 30_000 });
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
    if (shots) {
      await panel.locator(".viewfinder").scrollIntoViewIfNeeded();
      await page.screenshot({ path: `${shots}/camera-phone.png` });
    }
    await context.close();
  });
});
