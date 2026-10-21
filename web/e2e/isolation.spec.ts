import { expect, test, type Page } from "@playwright/test";

// Runs against a server that sends no isolation headers (a host that can't set them): multithreading only
// works if the service worker isolates the page, including after a hard refresh (which bypasses it).
// The page reloads itself while recovering, which can interrupt a check: count that as "not yet".
const isolated = (page: Page) => page.evaluate(() => self.crossOriginIsolated).catch(() => false);

test("the service worker makes the page multithreaded, even after a hard refresh", async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));

  // First visit: not isolated until the worker takes control and the page reloads itself once.
  await page.goto("benchmarks");
  await expect.poll(() => isolated(page), { timeout: 30_000 }).toBe(true);

  const lab = page.locator("#lab");
  await lab.scrollIntoViewIfNeeded();
  await expect(lab.locator(".env")).toContainText("threads", { timeout: 30_000 });
  await expect(lab.locator(".env")).not.toContainText("1 thread");

  // Hard refresh (Ctrl/Cmd+Shift+R) bypasses the worker; the page must recover by itself.
  const session = await page.context().newCDPSession(page);
  await session.send("Page.reload", { ignoreCache: true });
  await expect.poll(() => isolated(page), { timeout: 30_000 }).toBe(true);
  await page.locator("#lab").scrollIntoViewIfNeeded();
  await expect(page.locator("#lab .env")).toContainText("threads", { timeout: 30_000 });

  // The index loads multithreaded too, and a search works.
  await page.goto("try");
  await expect(page.locator(".result")).toHaveCount(12, { timeout: 150_000 });
  expect(await isolated(page)).toBe(true);
  expect(errors).toEqual([]);
});

test("when isolation is impossible, it says so and stays usable, without reload loops", async ({ page }) => {
  test.setTimeout(300_000);
  // Simulate a browser that refuses service workers (e.g. some private modes).
  await page.addInitScript(() => {
    Object.defineProperty(navigator.serviceWorker, "register", { value: () => Promise.reject(new Error("blocked")) });
  });
  let loads = 0;
  page.on("load", () => loads++);
  await page.goto("benchmarks");
  const lab = page.locator("#lab");
  await lab.scrollIntoViewIfNeeded();
  await expect(lab.locator(".env")).toContainText("1 thread", { timeout: 30_000 });
  await expect(lab.getByRole("button", { name: /Reload to use all \d+ cores/ })).toBeVisible();
  await page.waitForTimeout(3000);
  expect(loads).toBe(1); // no automatic reload loop
});

test("after one visit, the playground works offline", async ({ page, context }) => {
  test.setTimeout(300_000);
  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 150_000 });
  await expect.poll(() => isolated(page), { timeout: 30_000 }).toBe(true);
  await page.waitForTimeout(2000); // let the service worker finish writing the downloads to its cache

  await context.setOffline(true);
  await page.reload();
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 60_000 });
  await panel.locator(".result").nth(1).click();
  await expect(panel.locator(".results h2")).toContainText("Photos like");
  expect(await isolated(page)).toBe(true); // still multithreaded offline
  await context.setOffline(false);
});
