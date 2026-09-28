import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

// A fake camera, so the webcam search runs headless.
test.use({
  launchOptions: { args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
  permissions: ["camera"],
});

test("searches 29,825 photos by text, photo and camera, with the debugger and race", async ({ page, request }) => {
  const errors: string[] = [];
  page.on("console", (message) => message.type() === "error" && errors.push(message.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("try");
  const panel = page.getByRole("complementary", { name: "Search" });
  const results = panel.locator(".result");

  // The page runs a first search on its own once the graph is built.
  await expect(results).toHaveCount(12, { timeout: 150_000 });
  await expect(panel.locator(".stats")).toContainText("HNSW search");
  const agreement = await panel.locator(".detail strong").nth(1).textContent();
  expect(Number(agreement!.split("/")[0])).toBeGreaterThanOrEqual(10); // recall vs the exact scan

  // Clicking a result searches for similar photos; the query photo ranks first.
  await results.nth(3).click();
  await expect(panel.locator(".rank").first()).toHaveText("query");
  expect(new URL(page.url()).searchParams.get("photo")).not.toBeNull();

  // Text search: the in-browser CLIP text encoder lands in the same space as the photo embeddings.
  const input = panel.getByRole("searchbox", { name: "Describe a photo to search for" });
  await expect(input).toHaveAttribute("placeholder", "Describe a photo…", { timeout: 120_000 });
  await input.fill("a red apple");
  await input.press("Enter");
  await expect(panel.locator(".results h2")).toContainText("a red apple");
  const captions = await panel.locator(".caption > span:first-child").allTextContents();
  expect(captions.filter((c) => /apple/i.test(c)).length).toBeGreaterThanOrEqual(4);
  expect(new URL(page.url()).searchParams.get("q")).toBe("a red apple");

  // Step-through debugger: stepping pauses the replay and explains each decision.
  const debuggerBar = page.getByRole("region", { name: "Search debugger" });
  await debuggerBar.getByRole("button", { name: "Next step" }).click();
  await expect(debuggerBar.locator(".inspector")).toContainText("distance");
  const step = async () => Number((await debuggerBar.locator(".count").textContent())!.split("/")[0].replace(/,/g, ""));
  const before = await step();
  await page.keyboard.press("ArrowRight");
  await expect.poll(step).toBe(before + 1); // paused: one event per keypress
  await page.keyboard.press("ArrowLeft");
  await expect.poll(step).toBe(before);

  // Race against the exact scan.
  await page.getByRole("button", { name: "Race vs. exact" }).click();
  const race = page.getByRole("dialog", { name: "HNSW vs. an exact scan, same query" });
  await expect(race.locator("footer p")).toHaveClass(/show/, { timeout: 10_000 });
  await expect(race.locator("footer p")).toContainText("identical");
  if (shots) await page.screenshot({ path: `${shots}/race.png` });
  await page.keyboard.press("Escape");
  await expect(race).toHaveCount(0);

  // Photo search: upload one of the collection's own photos; it should find itself.
  const src = (await results.first().locator("img").getAttribute("src"))!;
  const photo = await request.get(src);
  await panel.getByRole("tab", { name: /Photo/ }).click();
  await panel.locator('input[type="file"]').setInputFiles({ name: "apple.jpg", mimeType: "image/jpeg", buffer: await photo.body() });
  await expect(panel.locator(".results h2")).toContainText("your photo", { timeout: 150_000 });
  const top3 = await results.locator("img").evaluateAll((imgs) => imgs.slice(0, 3).map((i) => (i as HTMLImageElement).src));
  expect(top3.map((s) => s.replace(/_\w\.jpg$/, ""))).toContain(src.replace(/_\w\.jpg$/, ""));

  // Camera search with Chrome's fake webcam.
  await panel.getByRole("tab", { name: /Camera/ }).click();
  await panel.getByRole("button", { name: /Start the camera|Turn on|camera/i }).first().click();
  await panel.getByRole("button", { name: "Capture & search" }).click();
  await expect(panel.locator(".results h2")).toContainText("camera frame", { timeout: 30_000 });

  // Changing ef re-runs the query live.
  const ef = page.getByRole("slider", { name: /Beam width/ });
  await ef.fill("128");
  await expect(page.locator(".knobs output")).toHaveText("128");

  if (shots) {
    await page.getByRole("tab", { name: /Describe/ }).click();
    await input.fill("a lighthouse by the sea");
    await input.press("Enter");
    await page.waitForTimeout(4000); // let the traversal replay finish
    await page.screenshot({ path: `${shots}/try-dark.png` });
    await page.getByRole("button", { name: "Switch to light theme" }).click();
    await page.waitForTimeout(800);
    await page.screenshot({ path: `${shots}/try-light.png` });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(1500);
    await page.screenshot({ path: `${shots}/try-mobile.png` });
  }
  expect(errors).toEqual([]);
});
