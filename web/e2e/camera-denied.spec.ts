import { expect, test } from "@playwright/test";

// Every way getUserMedia can fail for a real visitor gets a clear, actionable message, and the rest
// of the page keeps working. The browser API is stubbed to fail with each error in turn.
test("camera errors: blocked, missing, busy and unsupported each explain what to do", async ({ page }) => {
  test.setTimeout(300_000);
  await page.addInitScript(() => {
    const w = window as unknown as { cameraError: string };
    w.cameraError = "NotAllowedError";
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("stubbed", w.cameraError));
  });
  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 150_000 });
  await panel.getByRole("tab", { name: "Camera" }).click();

  const cases: [string, RegExp][] = [
    ["NotAllowedError", /blocked.*Allow the camera/],
    ["NotFoundError", /No camera was found/],
    ["NotReadableError", /busy in another app/],
    ["NotSupportedError", /can't open the camera/],
  ];
  for (const [name, message] of cases) {
    await page.evaluate((n) => ((window as unknown as { cameraError: string }).cameraError = n), name);
    await panel.getByRole("button", { name: /Point your camera/ }).click();
    await expect(panel.getByRole("alert")).toContainText(message);
    await expect(panel.getByRole("button", { name: /Point your camera/ })).toBeEnabled();
  }

  // The rest of the page still works.
  await panel.getByRole("button", { name: "Surprise me" }).click();
  await expect(panel.locator(".results h2")).toContainText("Photos like");
});
