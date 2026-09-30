import { expect, test } from "@playwright/test";

// A browser with WebGL switched off (disabled GPU, some remote desktops): no 3D view, but
// everything else works and nothing throws.
test.use({ launchOptions: { args: ["--disable-webgl", "--disable-webgl2", "--disable-3d-apis"] } });

test("without WebGL: search still works, and the missing 3D view is explained", async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  expect(await page.evaluate(() => !!document.createElement("canvas").getContext("webgl2"))).toBe(false);

  await page.goto("./");
  await expect(page.locator(".telemetry")).toContainText(/Index (ready|built)/, { timeout: 150_000 });

  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 150_000 });
  await expect(page.getByRole("note")).toContainText("3D view unavailable");
  await panel.locator(".result").nth(1).click();
  await expect(panel.locator(".results h2")).toContainText("Photos like");
  await page.getByRole("button", { name: "Race vs. exact" }).click();
  await expect(page.getByRole("dialog", { name: /HNSW vs\. an exact scan/ })).toBeVisible();
  expect(errors, errors.join("\n")).toEqual([]);
});
