import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// Phones get the pages in a menu behind the top-right button; wider screens get them as tabs.
test("phone menu: opens, navigates, closes on Escape and outside taps, and is accessible", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("benchmarks");
  const button = page.getByRole("button", { name: "Open the menu" });
  await expect(page.getByRole("navigation", { name: "Main" })).toBeHidden();
  await expect(button).toHaveAttribute("aria-expanded", "false");

  await button.click();
  const menu = page.getByRole("navigation", { name: "Menu" });
  await expect(menu).toBeVisible();
  await expect(page.getByRole("button", { name: "Close the menu" })).toHaveAttribute("aria-expanded", "true");
  await expect(menu.getByRole("link", { name: "Benchmarks" })).toHaveAttribute("aria-current", "page");
  await expect(menu.getByRole("link").first()).toBeFocused();
  const axe = await new AxeBuilder({ page }).include("#site-menu").withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(axe.violations.map((v) => v.id)).toEqual([]);

  // Escape closes it and hands focus back to the button.
  await page.keyboard.press("Escape");
  await expect(menu).toBeHidden();
  await expect(page.getByRole("button", { name: "Open the menu" })).toBeFocused();

  // A tap outside closes it.
  await button.click();
  await page.mouse.click(30, 600);
  await expect(menu).toBeHidden();

  // Following a link opens the page and closes the menu.
  await button.click();
  await menu.getByRole("link", { name: "Blog" }).click();
  await expect(page).toHaveURL(/\/blog$/);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Vector search is a memory problem");
  await expect(menu).toBeHidden();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);

  // Wider screens show the tabs, Blog among them, and no menu button.
  await page.setViewportSize({ width: 1280, height: 800 });
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Blog" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Open the menu" })).toBeHidden();
});
