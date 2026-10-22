import { expect, test, type Page } from "@playwright/test";

// The visitor's own photos: embedded in the browser, indexed, searched, kept and forgotten. The
// "photos" are drawn here (red, green and blue shapes), so the test needs no image files.
async function drawPhotos(page: Page) {
  const photos = await page.evaluate(async () => {
    const colours: [string, string][] = [
      ["red", "#d42a2a"],
      ["green", "#2a9d3a"],
      ["blue", "#2a52d4"],
    ];
    const out: { name: string; data: string }[] = [];
    for (const [name, fill] of colours) {
      for (let i = 0; i < 4; i++) {
        const canvas = new OffscreenCanvas(640, 480);
        const ctx = canvas.getContext("2d")!;
        ctx.fillStyle = fill;
        ctx.fillRect(0, 0, 640, 480);
        ctx.fillStyle = "#f4f1ea";
        ctx.beginPath();
        if (i % 2) ctx.arc(320, 240, 90 + i * 20, 0, Math.PI * 2);
        else ctx.rect(200 - i * 10, 140, 240 + i * 20, 200);
        ctx.fill();
        const blob = await canvas.convertToBlob({ type: "image/jpeg", quality: 0.9 });
        const bytes = new Uint8Array(await blob.arrayBuffer());
        let binary = "";
        for (const b of bytes) binary += String.fromCharCode(b);
        out.push({ name: `${name}-${i + 1}.jpg`, data: btoa(binary) });
      }
    }
    return out;
  });
  return photos.map((p) => ({ name: p.name, mimeType: "image/jpeg", buffer: Buffer.from(p.data, "base64") }));
}

test("indexes the visitor's own photos in the browser, searches them, keeps and forgets them", async ({ page }) => {
  test.setTimeout(600_000); // the image model (89 MB) and the text model on a cold cache
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && !m.location().url.includes("staticflickr.com") && errors.push(m.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 150_000 }); // the demo is up first

  const source = panel.getByRole("radiogroup", { name: "Photos to search" });
  await source.getByRole("radio", { name: /Your photos/ }).click();
  const dialog = page.getByRole("dialog", { name: "Search your own photos" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("Nothing is uploaded");

  // A folder brings other files along: they're ignored.
  const files = [...(await drawPhotos(page)), { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("not a photo") }];
  await dialog.locator('input[type="file"][accept="image/*"]').setInputFiles(files);
  await expect(dialog.getByRole("list", { name: "Progress" })).toBeVisible();
  await expect(dialog).toBeHidden({ timeout: 300_000 });
  await expect(page.locator(".toast")).toContainText("12 of your photos indexed");

  await expect(panel.locator("h1")).toHaveText("Search 12 of your photos, locally.");
  await expect(source.getByRole("radio", { name: /Your photos/ })).toHaveAttribute("aria-checked", "true");
  await expect(panel.locator(".result")).toHaveCount(12); // a first search over them, on its own
  await expect(panel.locator(".by")).toHaveCount(0); // no credits: they're the visitor's own
  await expect(page.getByRole("button", { name: "Share this search" })).toBeDisabled();

  // Text search across their own photos: the red ones come first.
  const input = panel.getByRole("searchbox", { name: "Describe a photo to search for" });
  await expect(input).toHaveAttribute("placeholder", "Describe a photo…", { timeout: 120_000 });
  await input.fill("a red picture");
  await input.press("Enter");
  await expect(panel.locator(".results h2")).toContainText("a red picture");
  const top = await panel.locator(".caption > span:first-child").allTextContents();
  expect(top.slice(0, 4).every((name) => name.startsWith("red-")), top.join(", ")).toBe(true);
  expect(new URL(page.url()).searchParams.get("q")).toBeNull(); // nothing about them goes in the URL

  // Kept in this browser: a reload offers them again without re-embedding.
  await page.reload();
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 150_000 });
  await expect(source.getByRole("radio", { name: /Your photos 12/ })).toBeVisible();
  await source.getByRole("radio", { name: /Your photos/ }).click();
  await expect(panel.locator("h1")).toHaveText("Search 12 of your photos, locally.", { timeout: 60_000 });

  // Back to the demo, then forget them for good.
  await source.getByRole("radio", { name: /Demo photos/ }).click();
  await expect(panel.locator("h1")).toHaveText("Search 29,825 photos, locally.", { timeout: 60_000 });
  await source.getByRole("radio", { name: /Your photos/ }).click();
  await expect(panel.locator("h1")).toHaveText("Search 12 of your photos, locally.", { timeout: 60_000 });
  await panel.getByRole("button", { name: "Forget them" }).click();
  await expect(panel.locator("h1")).toHaveText("Search 29,825 photos, locally.", { timeout: 60_000 });
  await page.reload();
  await expect(panel.locator(".result")).toHaveCount(12, { timeout: 150_000 });
  await expect(source.getByRole("radio", { name: /Your photos/ })).not.toContainText("12");

  expect(errors).toEqual([]);
});
