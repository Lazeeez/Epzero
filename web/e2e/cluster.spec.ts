import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

test("cluster: the Go router in WebAssembly fails over a killed node, and strong reads check out linearizable", async ({ page }) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

  await page.goto("cluster");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Break the cluster");
  const nodes = page.locator(".node");
  await expect(nodes).toHaveCount(4); // drawn while the simulator starts
  // Once it runs, every shard has a primary and a replica somewhere.
  await expect(page.locator(".copies li.primary")).toHaveCount(4, { timeout: 60_000 });
  await expect(page.locator(".copies li.replica")).toHaveCount(4, { timeout: 15_000 });
  // Searches run and match the exact answers.
  await expect(page.locator(".tile").first().locator("b")).toHaveText("100.0%", { timeout: 20_000 });
  if (shots) await page.screenshot({ path: `${shots}/cluster.png`, fullPage: true });

  // Kill node-1: the router declares it down and promotes the replicas of its primaries.
  const first = nodes.first();
  await first.getByRole("button", { name: "Kill" }).click();
  await expect(first.locator(".pill")).toHaveText("Down");
  await expect(page.locator(".log li.failover").first()).toBeVisible({ timeout: 10_000 });
  await expect(first.getByRole("button", { name: "Restart" })).toBeVisible();
  if (shots) await page.screenshot({ path: `${shots}/cluster-failover.png` });

  // Strong reads, under a lossy link, still give a linearizable history.
  await page.getByRole("radio", { name: "Strong" }).click();
  await first.getByRole("button", { name: "Restart" }).click();
  await expect(first.locator(".pill")).toHaveText("Up", { timeout: 10_000 });
  await page.waitForTimeout(6000); // let a few hundred reads and writes accumulate
  await page.getByRole("button", { name: "Check linearizability" }).click();
  await expect(page.locator(".verdict")).toContainText("Linearizable.", { timeout: 30_000 });

  // Leaving the page stops the simulator's worker.
  await page.getByRole("link", { name: "Benchmarks" }).first().click();
  await expect(page).toHaveURL(/benchmarks/);
  expect(errors).toEqual([]);
});

test("cluster: phone layout stacks the nodes", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("cluster");
  await expect(page.locator(".node")).toHaveCount(4, { timeout: 60_000 });
  const widths = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
  expect(widths[0]).toBeLessThanOrEqual(widths[1]); // no sideways scrolling
  // Four nav links still leave room for the theme toggle.
  await expect(page.getByRole("button", { name: /Switch to (light|dark) theme/ })).toBeInViewport({ ratio: 1 });
  if (shots) await page.screenshot({ path: `${shots}/cluster-phone.png`, fullPage: true });
});
