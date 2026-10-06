import { expect, test } from "@playwright/test";

const shots = process.env.SCREENSHOT_DIR;

// A fake camera, so the webcam search runs headless.
test.use({
  launchOptions: { args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
  permissions: ["camera", "clipboard-read", "clipboard-write"],
});

test("searches 29,825 photos by text, photo and camera, with the debugger and race", async ({ page, request }) => {
  test.setTimeout(600_000); // downloads both CLIP models (156 MB) on a cold cache, on a GPU-less CI runner
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
  await expect(panel.getByRole("group", { name: "Recent searches" })).toContainText("a red apple");

  // Share copies a link that reopens this exact search.
  await page.getByRole("button", { name: "Share this search" }).click();
  await expect(page.getByRole("status").filter({ hasText: "Link copied" })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain("q=a+red+apple");

  // "?" lists every shortcut; Escape closes it.
  await page.keyboard.press("?");
  const shortcuts = page.getByRole("dialog", { name: "Keyboard shortcuts" });
  await expect(shortcuts).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(shortcuts).toHaveCount(0);

  // Step-through debugger: stepping pauses the replay and explains each decision.
  const debuggerBar = page.getByRole("region", { name: "Search debugger" });
  await debuggerBar.getByRole("button", { name: "Next step" }).click();
  await expect(debuggerBar.locator(".inspector")).toContainText("distance");
  const step = async () => Number((await debuggerBar.locator(".count").textContent())!.split("/")[0].replace(/,/g, ""));
  // Back then forward: works whether the search was still replaying or already shown finished (as it
  // is with reduced motion, or on a GPU-less machine).
  const before = await step();
  await page.keyboard.press("ArrowLeft");
  await expect.poll(step).toBe(before - 1); // paused: one event per keypress
  await page.keyboard.press("ArrowRight");
  await expect.poll(step).toBe(before);

  // Free look: everything but the top bar slides away (search panel, settings bar, debugger, HUD and
  // view controls), leaving the graph. The top bar's button, H or Esc bring it all back.
  const hud = page.getByRole("complementary", { name: "Navigator instruments" });
  const chrome = [".knobs", ".dock", ".view-anchor"];
  await expect(hud).toBeVisible();
  await page.getByRole("button", { name: "Free look", exact: true }).click();
  await expect(panel).toHaveClass(/collapsed/);
  for (const part of chrome) await expect(page.locator(part)).toHaveClass(/away/);
  await expect(hud).toHaveCount(0);
  // Below the top bar, nothing is left over the graph.
  await expect
    .poll(() =>
      page.evaluate(() => {
        const seen = new Set<string>();
        for (let x = 0.03; x < 1; x += 0.047)
          for (let y = 0.1; y < 1; y += 0.045) {
            const el = document.elementFromPoint(innerWidth * x, innerHeight * y);
            if (el && el.tagName !== "CANVAS") seen.add(`${el.tagName.toLowerCase()}.${el.className}`);
          }
        return [...seen];
      }),
    )
    .toEqual([]);
  if (shots) await page.screenshot({ path: `${shots}/free-look.png` });
  const showPanels = page.getByRole("button", { name: "Show the panels" });
  await expect(showPanels).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(panel).not.toHaveClass(/collapsed/);
  for (const part of chrome) await expect(page.locator(part)).not.toHaveClass(/away/);
  await expect(hud).toBeVisible();
  await page.keyboard.press("h");
  await expect(panel).toHaveClass(/collapsed/);
  await showPanels.click();
  await expect(panel).not.toHaveClass(/collapsed/);
  // From the panel's own button, the keyboard focus moves to the way back.
  await panel.getByRole("button", { name: "Hide the panels" }).focus();
  await page.keyboard.press("Enter");
  await expect(showPanels).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(panel).not.toHaveClass(/collapsed/);
  await expect(page.getByRole("button", { name: "Free look", exact: true })).toBeFocused(); // stays on the button just used

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

  // Blend: a collection photo plus words, mixed by a slider; the link reproduces it.
  await panel.getByRole("tab", { name: "Describe" }).click();
  await results.nth(1).click();
  await expect(panel.locator(".results h2")).toContainText("Photos like");
  await panel.getByRole("tab", { name: "Blend" }).click();
  await panel.getByRole("button", { name: "in the snow" }).click();
  await expect(panel.locator(".results h2")).toContainText("in the snow", { timeout: 30_000 });
  await panel.getByRole("slider", { name: /Balance between the photo and the words/ }).fill("0.8");
  await expect.poll(() => new URL(page.url()).searchParams.get("mix")).toBe("0.80");
  expect(new URL(page.url()).searchParams.get("plus")).toBe("in the snow");

  // Navigator HUD follows the replay.
  await expect(page.getByRole("complementary", { name: "Navigator instruments" })).toContainText("Heading");

  // Tune and rebuild: a new graph, measured against an exact scan.
  await page.getByRole("button", { name: "Index settings" }).click();
  const tune = page.getByRole("region", { name: "Tune the index" });
  await expect(tune.locator("tbody tr")).toHaveCount(1, { timeout: 60_000 });
  await tune.getByRole("button", { name: "8", exact: true }).click();
  await tune.getByRole("button", { name: "Rebuild index" }).click();
  await expect(tune.locator("tbody tr")).toHaveCount(2, { timeout: 120_000 });
  await expect(tune.locator("tbody tr").nth(1).locator("td").nth(5)).toHaveText(/^0\.\d{3}$/, { timeout: 60_000 });
  if (shots) await page.screenshot({ path: `${shots}/tune.png` });
  await page.keyboard.press("Escape");
  await expect(tune).toHaveCount(0);

  // The 3D graph takes the mouse: hovering a star shows its photo, clicking searches from it.
  await page.getByRole("button", { name: "Auto-rotate" }).click(); // hold the camera still
  await page.waitForTimeout(3000); // let any camera flight finish
  const hovercard = page.locator(".hovercard");
  const heading = panel.locator(".results h2");
  // Probe outward from the dense middle of the graph (the free area right of the panel).
  let searched = false;
  const [cx, cy] = [(424 + 1440) / 2, 430];
  for (let i = 0; i < 60 && !searched; i++) {
    const angle = i * 2.4;
    const radius = 14 * Math.sqrt(i);
    await page.mouse.move(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    // The scene picks what's under the pointer once per rendered frame: wait for two frames (on a
    // GPU-less CI runner one can take several hundred ms) instead of a fixed delay.
    await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
    if (!(await hovercard.isVisible())) continue;
    await page.mouse.down();
    await page.mouse.up();
    searched = await heading.getByText("Photos like").waitFor({ timeout: 3_000 }).then(() => true, () => false);
  }
  expect(searched).toBe(true);

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
