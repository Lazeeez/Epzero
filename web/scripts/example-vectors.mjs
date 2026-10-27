// Embeds the Try page's example searches (src/lib/engine/example-queries.json) with the site's own
// int8 CLIP text model, so tapping one searches at once, before the model has downloaded. Writes
// src/lib/engine/example-vectors.json (base64 float32, unit length). Run after `npm run assets`
// whenever the examples change:
//   node scripts/example-vectors.mjs
import { readFile, writeFile } from "node:fs/promises";
import { AutoTokenizer, CLIPTextModelWithProjection, env } from "@huggingface/transformers";

const root = new URL("../", import.meta.url);
env.allowRemoteModels = false;
env.localModelPath = new URL("public/models/", root).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const MODEL = "clip-vit-base-patch32";
const queries = JSON.parse(await readFile(new URL("src/lib/engine/example-queries.json", root), "utf8"));

const tokenizer = await AutoTokenizer.from_pretrained(MODEL);
const model = await CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: "q8" });
const vectors = {};
for (const query of queries) {
  // One at a time, as the page encodes them (padding a batch would change nothing, but keep it exact).
  const { text_embeds } = await model(tokenizer([query], { padding: true, truncation: true }));
  const vector = Float32Array.from(text_embeds.data);
  const norm = Math.hypot(...vector) || 1;
  for (let i = 0; i < vector.length; i++) vector[i] /= norm;
  vectors[query] = Buffer.from(vector.buffer).toString("base64");
}
await writeFile(new URL("src/lib/engine/example-vectors.json", root), JSON.stringify(vectors, null, 1) + "\n");
console.log(`embedded ${queries.length} examples`);
