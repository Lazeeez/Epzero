import { expect, test } from "@playwright/test";

// What search engines see before running any JavaScript: every page's own title, description,
// canonical address and link-preview tags, a fallback with its heading and links, the structured
// data, the sitemap and robots.txt, and a 404 page that asks not to be indexed.
const pages = [
  { path: "", title: /^Epzero · A vector database/ },
  { path: "try", title: /^Search 29,825 photos/ },
  { path: "play", title: /out-navigate/ },
  { path: "cluster", title: /^Break a vector database cluster/ },
  { path: "benchmarks", title: /^Benchmarks: Epzero vs/ },
];

test("every page's raw HTML is ready for search engines", async ({ request, baseURL }) => {
  const titles = new Set<string>();
  const descriptions = new Set<string>();
  for (const { path, title } of pages) {
    // "try/" reaches try/index.html on any static server (Cloudflare redirects it to "try").
    const response = await request.get(path ? `${path}/` : path);
    expect(response.status(), path).toBe(200);
    const html = await response.text();
    const pageTitle = html.match(/<title>([^<]+)<\/title>/)?.[1] ?? "";
    const description = html.match(/<meta name="description" content="([^"]+)"/)?.[1] ?? "";
    expect(pageTitle, path).toMatch(title);
    expect(pageTitle.length, `${path} title length`).toBeLessThanOrEqual(60);
    expect(description.length, `${path} description length`).toBeGreaterThan(70);
    expect(description.length, `${path} description length`).toBeLessThanOrEqual(160);
    titles.add(pageTitle);
    descriptions.add(description);
    expect(html, path).toContain(`<link rel="canonical" href="https://epzero.dev/${path}" />`);
    expect(html, path).toContain(`<meta property="og:url" content="https://epzero.dev/${path}" />`);
    expect(html, path).toMatch(/<noscript><main[^>]*><h1>[^<]+<\/h1>/);
    const ld = JSON.parse(html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)![1]);
    expect(ld["@graph"].map((node: { "@type": string }) => node["@type"])).toEqual(["WebSite", "Person", "WebApplication"]);
  }
  expect(titles.size).toBe(pages.length); // no two pages share a title or a description
  expect(descriptions.size).toBe(pages.length);

  const sitemap = await (await request.get("sitemap.xml")).text();
  for (const { path } of pages) expect(sitemap).toContain(`<loc>https://epzero.dev/${path}</loc>`);
  expect(await (await request.get("robots.txt")).text()).toContain("Sitemap: https://epzero.dev/sitemap.xml");

  // Production answers an unknown address with 404.html and a 404 status; the preview server falls
  // back to the home page, so there the file itself is checked.
  const unknown = await request.get(`${baseURL}no-such-page`);
  const missing = unknown.status() === 404 ? await unknown.text() : await (await request.get("404.html")).text();
  expect(missing).toContain('<meta name="robots" content="noindex" />');
  expect(missing).not.toContain('rel="canonical"');
});

test("moving between pages keeps the title, description and canonical in step", async ({ page }) => {
  await page.goto("benchmarks");
  await expect(page).toHaveTitle(/^Benchmarks: Epzero vs/);
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Play" }).click();
  await expect(page).toHaveTitle(/out-navigate/);
  await expect(page.locator('link[rel="canonical"]')).toHaveAttribute("href", "https://epzero.dev/play");
  await expect(page.locator('meta[name="description"]')).toHaveAttribute("content", /^A game: hop from photo to photo/);
});
