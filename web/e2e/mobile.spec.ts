import { expect, test } from "@playwright/test";

// A phone: the explainer is a stepper, the search panel a sheet that expands, and the blog's wide
// tables are cards. Nothing scrolls sideways.
test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });

const fits = (page: import("@playwright/test").Page) => page.evaluate(() => document.documentElement.scrollWidth <= innerWidth);

test("phone: explainer stepper, expandable search sheet and blog table cards", async ({ page }) => {
  test.setTimeout(240_000);

  // The explainer steps with buttons, without scrolling the page; only the current step is reachable.
  await page.goto("./");
  const stepper = page.getByRole("navigation", { name: "Explanation steps" });
  await stepper.scrollIntoViewIfNeeded();
  const steps = page.locator("#how .step");
  await expect(steps.nth(0)).toHaveClass(/active/);
  await page.waitForTimeout(1000); // let lazily rendered sections settle
  const top = async () => (await stepper.boundingBox())!.y;
  const y = await top();
  await stepper.getByRole("button", { name: "Next" }).click();
  await expect(steps.nth(1)).toHaveClass(/active/);
  await expect(stepper.getByRole("button", { name: /^Step 2:/ })).toHaveAttribute("aria-current", "step");
  await expect(steps.nth(0)).toHaveAttribute("inert", "");
  expect(Math.abs((await top()) - y)).toBeLessThan(2);
  await stepper.getByRole("button", { name: "Previous step" }).click();
  await expect(steps.nth(0)).toHaveClass(/active/);
  expect(await fits(page)).toBe(true);

  // The search sheet expands for the results and shrinks back to show the graph.
  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  await expect(panel.locator(".result").first()).toBeVisible({ timeout: 150_000 });
  const half = (await panel.boundingBox())!.height;
  await page.getByRole("button", { name: "Expand the search panel" }).click();
  await expect(panel).toHaveClass(/expanded/);
  await expect.poll(async () => (await panel.boundingBox())!.height).toBeGreaterThan(half * 1.4);
  await expect(page.locator(".knobs")).toHaveCSS("opacity", "0");
  await page.getByRole("button", { name: "Shrink the search panel" }).click();
  await expect(panel).not.toHaveClass(/expanded/);
  await expect(page.locator(".knobs")).toHaveCSS("opacity", "1");
  expect(await fits(page)).toBe(true);

  // A three-column blog table becomes labelled cards.
  await page.goto("blog");
  const card = page.locator(".table-wrap.stack tbody tr").first();
  await card.scrollIntoViewIfNeeded();
  await expect(page.locator(".table-wrap.stack thead")).toBeHidden();
  await expect(card.locator("td").nth(1)).toHaveAttribute("data-label", "Result");
  const cardBox = (await card.boundingBox())!;
  expect(cardBox.width).toBeGreaterThan(300);
  expect(await fits(page)).toBe(true);
});
