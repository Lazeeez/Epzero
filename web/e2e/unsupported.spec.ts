import { expect, test } from "@playwright/test";

// A browser too old for WebAssembly SIMD (e.g. Safari before 16.4): every page explains what to do,
// with no dead-end "Try again" button and no crash.
test("without WebAssembly SIMD: a clear explanation everywhere", async ({ page }) => {
  test.setTimeout(300_000);
  await page.addInitScript(() => {
    WebAssembly.validate = () => false; // what an old browser answers for the SIMD probe
  });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("./");
  await expect(page.locator(".telemetry")).toContainText("needs WebAssembly SIMD", { timeout: 30_000 });
  await expect(page.locator(".telemetry").getByRole("button", { name: "Try again" })).toHaveCount(0);

  await page.goto("try");
  await expect(page.getByRole("status").filter({ hasText: "too old for the engine" })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("button", { name: "Try again" })).toHaveCount(0);

  await page.goto("benchmarks");
  await page.locator("#lab").scrollIntoViewIfNeeded();
  await expect(page.locator("#lab .env")).toContainText("needs WebAssembly SIMD", { timeout: 30_000 });
  expect(errors, errors.join("\n")).toEqual([]);
});
