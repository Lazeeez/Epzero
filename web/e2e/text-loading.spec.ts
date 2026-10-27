import { expect, test } from "@playwright/test";

// While the text model downloads, the examples search at once (their vectors ship with the page) and
// a typed search waits for the model, then runs. The model is held back here until the test lets it go.
// Without the service worker: its requests bypass the test's hold on the download.
test.use({ serviceWorkers: "block" });

test("text search before the model has loaded: examples now, typed searches when it's ready", async ({ page }) => {
  test.setTimeout(300_000);
  let release!: () => void;
  const held = new Promise<void>((resolve) => (release = resolve));
  await page.route("**/text_model_quantized.onnx", async (route) => {
    await held;
    await route.continue();
  });

  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  const input = panel.getByRole("searchbox", { name: "Describe a photo to search for" });
  await expect(input).toHaveAttribute("data-model", "loading", { timeout: 150_000 });
  await expect(input).toBeEnabled({ timeout: 150_000 });
  await expect(page.locator("#text-model-status")).toContainText("type now, or tap an example");

  const example = panel.getByRole("group", { name: "Example searches" }).getByRole("button", { name: "snowy mountains" });
  await example.click();
  await expect(panel.getByText("Results for “snowy mountains”")).toBeVisible();
  await expect(panel.locator(".result")).toHaveCount(12);

  await input.fill("a cat sleeping on a sofa");
  await input.press("Enter");
  await expect(page.locator("#text-model-status")).toContainText("Searching as soon as the text model is ready");
  release();
  await expect(input).toHaveAttribute("data-model", "ready", { timeout: 200_000 });
  await expect(panel.getByText("Results for “a cat sleeping on a sofa”")).toBeVisible({ timeout: 30_000 });
  await expect(page.locator("#text-model-status")).toHaveCount(0);
});
