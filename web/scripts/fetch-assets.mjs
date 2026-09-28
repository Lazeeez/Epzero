// Puts the text-search assets next to the site so the live demo depends on no third-party CDN:
//   public/ort/     ONNX Runtime Web (SIMD + threads build), copied from node_modules
//   public/models/  CLIP ViT-B/32 text encoder (int8 ONNX) + tokenizer, downloaded once
// Both folders are generated (git-ignored) and cached between CI runs.
import { copyFile, mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const ortDist = join(root, "node_modules", "onnxruntime-web", "dist");
const ortOut = join(root, "public", "ort");
const modelOut = join(root, "public", "models", "clip-vit-base-patch32");
const HF = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/";
const MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
  "onnx/text_model_quantized.onnx"];

const exists = (path) => stat(path).then((s) => s.size > 0, () => false);

await mkdir(ortOut, { recursive: true });
for (const file of ["ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"]) {
  await copyFile(join(ortDist, file), join(ortOut, file));
}

for (const file of MODEL_FILES) {
  const target = join(modelOut, file);
  if (await exists(target)) continue;
  await mkdir(dirname(target), { recursive: true });
  const response = await fetch(HF + file);
  if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
  await writeFile(target, new Uint8Array(await response.arrayBuffer()));
  console.log(`fetched ${file} (${((await stat(target)).size / 1e6).toFixed(1)} MB)`);
}
console.log("text-search assets ready");
