import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

test("runs the full benchmark in the browser without errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("lab.html");
  await expect(page.locator("#env .chip").first()).toContainText("SIMD128");
  await expect(page.locator("#env")).toContainText("cross-origin isolated");

  await page.locator('label:has(input[name="n"][value="20000"])').click();
  await page.getByRole("button", { name: "Run benchmark" }).click();

  // The last stage finishing means every earlier one did.
  await expect(page.locator('.stages li[data-stage="kernel"]')).toHaveAttribute("data-state", "done", {
    timeout: 150_000,
  });
  await expect(page.locator("#sweep-table tbody tr")).toHaveCount(10);
  await expect(page.locator(".tile")).toHaveCount(4);

  // Recall must rise with ef and reach a useful operating point.
  const recalls = await page.locator("#sweep-table tbody tr td:nth-child(2)").allTextContents();
  expect(Number(recalls.at(-1))).toBeGreaterThan(0.99);

  // Hover the chart for the tooltip.
  const plot = page.locator("#chart svg rect").last();
  const box = (await plot.boundingBox())!;
  await plot.hover({ position: { x: box.width * 0.6, y: box.height / 2 } });
  await expect(page.locator(".tooltip")).toBeVisible();
  await expect(page.locator(".tooltip")).toContainText("Recall@10");

  if (shots) {
    const capture = async (name: string) => {
      await page.mouse.move(0, 0);
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.waitForTimeout(400); // let theme transitions settle
      await page.screenshot({ path: `${shots}/${name}.png`, fullPage: true });
    };
    await capture("lab-light");
    await page.emulateMedia({ colorScheme: "dark" });
    await capture("lab-dark");
    await page.setViewportSize({ width: 390, height: 844 });
    await capture("lab-mobile");
  }
  expect(errors).toEqual([]);
});
