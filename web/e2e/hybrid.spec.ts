import { expect, test } from "@playwright/test";

// Hybrid search: a word CLIP doesn't know ("tagine") still finds the photo whose caption has it,
// marked as a keyword match, and switching hybrid search off brings back the image search alone.
test("hybrid search finds captions' words that the image search misses", async ({ page }) => {
  test.setTimeout(400_000); // the CLIP text model downloads on a cold cache
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && !m.location().url.includes("staticflickr.com") && errors.push(m.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  const input = panel.getByRole("searchbox", { name: "Describe a photo to search for" });
  await expect(input).toHaveAttribute("placeholder", "Describe a photo…", { timeout: 300_000 });
  const toggle = panel.getByRole("switch", { name: /Match words in captions too/ });
  await expect(toggle).toBeChecked(); // on by default

  await input.fill("tagine");
  await input.press("Enter");
  await expect(panel.locator(".results h2")).toContainText("tagine");
  const tagine = panel.locator(".result", { hasText: "Tunisian Tagine Malsouka" });
  await expect(tagine).toHaveCount(1);
  await expect(tagine.locator(".words")).toHaveText("Aa");
  await expect(panel.locator(".stats .keywords")).toContainText("matched 1 captions");
  // The exact-scan comparison still describes the image search.
  await expect(panel.locator(".detail")).toContainText("same as the exact scan (image search)");

  // Off: the same words, image search only (re-run at once by the switch).
  await toggle.uncheck();
  await expect(panel.locator(".result .words")).toHaveCount(0);
  await expect(panel.locator(".result", { hasText: "Tunisian Tagine Malsouka" })).toHaveCount(0);
  await expect(panel.locator(".stats .keywords")).toHaveCount(0);

  // The choice is remembered.
  await page.reload();
  await expect(panel.getByRole("switch", { name: /Match words in captions too/ })).not.toBeChecked({ timeout: 300_000 });
  expect(errors).toEqual([]);
});
