import { expect, test } from "@playwright/test";

// The benchmark lab on the visitor's own vectors: a NumPy file and a CSV, made here.

// Clustered unit vectors, like real embeddings: n x dim float32, seeded.
function vectors(n: number, dim: number, seed = 7): Float32Array {
  let s = seed;
  const random = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
  const centres = Array.from({ length: 40 }, () => Float32Array.from({ length: dim }, random));
  const out = new Float32Array(n * dim);
  for (let i = 0; i < n; i++) {
    const c = centres[i % centres.length];
    let norm = 0;
    for (let j = 0; j < dim; j++) {
      const x = c[j] + 0.35 * random();
      out[i * dim + j] = x;
      norm += x * x;
    }
    norm = Math.sqrt(norm);
    for (let j = 0; j < dim; j++) out[i * dim + j] /= norm;
  }
  return out;
}

function npy(data: Float32Array, n: number, dim: number): Buffer {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${n}, ${dim}), }`;
  header += " ".repeat(64 - ((10 + header.length + 1) % 64)) + "\n";
  const prefix = Buffer.alloc(10);
  prefix.write("\x93NUMPY", 0, "latin1");
  prefix[6] = 1;
  prefix[7] = 0;
  prefix.writeUInt16LE(header.length, 8);
  return Buffer.concat([prefix, Buffer.from(header, "latin1"), Buffer.from(data.buffer)]);
}

test("the lab benchmarks the visitor's own vectors: .npy and CSV, with a readable error for bad files", async ({ page }) => {
  test.setTimeout(300_000);
  const errors: string[] = [];
  page.on("console", (m) => m.type() === "error" && !m.location().url.includes("staticflickr.com") && errors.push(m.text()));
  page.on("pageerror", (error) => errors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("benchmarks");
  const lab = page.locator("#lab");
  await lab.scrollIntoViewIfNeeded();
  await expect(lab.locator(".env")).toContainText("SIMD128", { timeout: 30_000 });
  await lab.getByRole("radio", { name: /Your vectors/ }).click();
  const run = lab.getByRole("button", { name: "Run on this device" });
  await expect(run).toBeDisabled(); // nothing to run yet

  // A file that isn't vectors: a clear message, and still nothing to run.
  await lab.getByLabel("Vector file").setInputFiles({ name: "notes.csv", mimeType: "text/csv", buffer: Buffer.from("a,b\n1,2\n3,4,5\n") });
  await expect(lab.locator(".file-error")).toContainText("Line 3 has 3 values, the first row 2");
  await expect(run).toBeDisabled();

  // NumPy float32, unit length: cosine is suggested.
  const n = 6000;
  const dim = 96;
  await lab.getByLabel("Vector file").setInputFiles({ name: "embeddings.npy", mimeType: "application/octet-stream", buffer: npy(vectors(n, dim), n, dim) });
  await expect(lab.locator(".file")).toContainText("6,000 × 96 · NumPy float32");
  await expect(lab.locator(".file")).toContainText("unit length");
  await expect(lab.getByRole("radio", { name: /Cosine/ })).toBeChecked();
  await expect(lab.locator(".seg label.on", { hasText: "Cosine" })).toContainText("suggested");

  await run.click();
  await expect(lab.locator('.stages li[data-state="done"]')).toHaveCount(6, { timeout: 150_000 });
  await expect(lab.locator(".stages li").first()).toContainText("Load your vectors");
  await expect(lab.locator(".stages li").first()).toContainText("5,400 vectors · 600 queries · 96-D · cosine"); // 10% held out
  await expect(lab.locator("tbody tr")).toHaveCount(12);
  const recalls = await lab.locator("tbody tr td:nth-child(2)").allTextContents();
  expect(Number(recalls.at(-1))).toBeGreaterThan(0.98);

  // A CSV with a header row, as L2.
  const small = vectors(1500, 16, 3);
  const rows = ["d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15"];
  for (let i = 0; i < 1500; i++) rows.push(Array.from(small.subarray(i * 16, (i + 1) * 16), (x) => (x * 3).toFixed(5)).join(","));
  await lab.getByLabel("Vector file").setInputFiles({ name: "points.csv", mimeType: "text/csv", buffer: Buffer.from(rows.join("\n")) });
  await expect(lab.locator(".file")).toContainText("1,500 × 16 · CSV");
  await expect(lab.getByRole("radio", { name: /^L2/ })).toBeChecked(); // not unit length
  await run.click();
  await expect(lab.locator('.stages li[data-state="done"]')).toHaveCount(6, { timeout: 150_000 });
  await expect(lab.locator(".stages li").first()).toContainText("1,350 vectors · 150 queries · 16-D · l2");

  expect(errors).toEqual([]);
});
