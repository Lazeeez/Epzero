import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

test("builds the graph in the browser and replays searches in 3D", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto("./");
  await expect(page.locator("#dataset-count")).not.toHaveText("Thousands of", { timeout: 30_000 });

  // The page runs a first search on its own once the graph is built.
  await expect(page.locator("#results li")).toHaveCount(10, { timeout: 150_000 });
  await expect(page.locator("#stats")).toContainText("HNSW search");
  const agreement = await page.locator("#stats .stat").nth(1).locator("b").textContent();
  expect(Number(agreement!.split("/")[0])).toBeGreaterThanOrEqual(8); // recall vs exact scan

  // Clicking a result searches for similar photos; the query photo ranks first.
  await page.locator("#results .result").nth(3).click();
  await expect(page.locator("#results .rank").first()).toHaveText("query");

  // Changing ef re-runs the query live.
  await page.locator("#ef").fill("96");
  await expect(page.locator("#ef-value")).toHaveText("96");

  if (shots) {
    await page.waitForTimeout(3500); // let the traversal replay finish
    await page.screenshot({ path: `${shots}/explorer-dark.png` });
    await page.emulateMedia({ colorScheme: "light" });
    await page.waitForTimeout(600);
    await page.screenshot({ path: `${shots}/explorer-light.png` });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#surprise").click();
    await page.waitForTimeout(3500);
    await page.screenshot({ path: `${shots}/explorer-mobile.png` });
  }
  expect(errors).toEqual([]);
});
