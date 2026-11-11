import { expect, test } from "@playwright/test";

test("download page: reached from the header, code for every workflow, copyable", async ({ page, context, browserName }) => {
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1280, height: 900 });

  // The header's Download button, from another page; it is marked as the current page once there.
  await page.goto("benchmarks");
  const button = page.locator(".nav .cta");
  await expect(button).toHaveText(/Download/);
  await button.click();
  await expect(page).toHaveURL(/\/download$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("pip install epzero");
  await expect(button).toHaveAttribute("aria-current", "page");

  // The code tabs: by click and by arrow keys.
  const tabs = page.getByRole("tab");
  await expect(tabs).toHaveCount(8);
  await expect(page.getByRole("tabpanel")).toContainText("epzero.build(vectors");
  await page.getByRole("tab", { name: /Keep it accurate/ }).click();
  await expect(page.getByRole("tabpanel")).toContainText("index.requantize()");
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: /Metadata filters/ })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("tabpanel")).toContainText("meta.select(");
  await page.getByRole("tab", { name: /Evaluate retrieval/ }).click();
  await expect(page.getByRole("tabpanel")).toContainText("ir.evaluate(run, qrels)");

  // Copying the install command (Chromium grants clipboard access to tests).
  if (browserName === "chromium") {
    await context.grantPermissions(["clipboard-read", "clipboard-write"]);
    await page.getByRole("button", { name: "Copy the install command" }).click();
    await expect(page.getByRole("button", { name: "Copy the install command" })).toHaveText("Copied");
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe("pip install epzero");
  }

  // The BEIR results.
  const beir = page.locator("section[aria-labelledby=beir-title]");
  await beir.scrollIntoViewIfNeeded();
  await expect(beir.locator("tbody tr")).toHaveCount(2);
  await expect(beir).toContainText("0.713");

  // Phones: nothing scrolls sideways, and the menu leads with Download.
  await page.setViewportSize({ width: 360, height: 780 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(360);
  await page.getByRole("button", { name: "Open the menu" }).click();
  await expect(page.locator(".menu .menu-cta")).toContainText("Download Epzero");
  expect(errors).toEqual([]);
});
