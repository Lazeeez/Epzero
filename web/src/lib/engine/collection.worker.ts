// Owns the Oryx engine for the site: streams the int8 CLIP vectors and the prebuilt HNSW graph
// (or builds the graph on every core when there isn't one, and on request), and answers searches
// with full traversal traces. Nothing here touches the DOM.
import { cachedFetch } from "../cached-fetch";
import { stageEnds, type Evaluation, type Explanation, type FilterInfo, type FromWorker, type Neighborhood, type Request, type ToWorker, type Trace } from "./protocol";

interface Collection {
  load(vectors: Int8Array, scales: Float32Array, dim: number): void;
  beginBuild(M: number, efConstruction: number): void;
  loadGraph(bytes: Uint8Array, M: number, efConstruction: number): void;
  buildStep(count: number, threads: number): number;
  levels(): Uint8Array;
  edges(layer: number): Uint32Array;
  search(query: Float32Array, k: number, ef: number): Trace;
  hybridSearch(query: Float32Array, keywords: Keywords, words: string, k: number, ef: number): Trace;
  searchById(id: number, k: number, ef: number): Trace;
  evaluate(samples: number, k: number, ef: number, threads: number): Omit<Evaluation, "ef">;
  setFilter(ids: Uint32Array | null): FilterInfo;
  explain(query: Float32Array, k: number, label: number, ef: number): Explanation;
  explainById(id: number, k: number, label: number, ef: number): Explanation;
  info(): { maxLevel: number; memoryBytes: number };
}

interface Keywords {
  set(texts: string): number;
}

const post = (message: FromWorker, transfer: Transferable[] = []) =>
  (self as unknown as Worker).postMessage(message, transfer);

const variant: "mt" | "st" = self.crossOriginIsolated ? "mt" : "st";
const threads = variant === "mt" ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;
const engine = import(/* @vite-ignore */ `${import.meta.env.BASE_URL}wasm/${variant}/oryx.js`).then(
  (module) => module.default() as Promise<{ Collection: new () => Collection; Keywords: new () => Keywords }>,
);
let collection: Collection | null = null;
// The captions' keyword index (BM25), built once from the "keywords" message.
let keywords: Keywords | null = null;

// Words too common to mean anything as keywords. Fusion goes by rank, so even a weak match would
// otherwise pull a photo into the results.
const COMMON = new Set(
  "a an and are as at be by for from has have in is it its of on or that the this to was were with photo photos picture image".split(" "),
);
function keywordsOf(text: string): string {
  return (text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []).filter((w) => w.length > 1 && !COMMON.has(w)).join(" ");
}
// The int8 codes and per-dimension scales, kept to answer vector and similarity requests in JS.
// They are the photos' coordinates on the dataset's principal axes, and only the leading `dim` of
// the `fullDim` axes until the rest have downloaded (see load).
let codes = new Int8Array(0);
let scales = new Float32Array(0);
let dim = 0;
let fullDim = 0;
let count = 0;
// The principal axes, one row of fullDim per axis: rotates queries in and photo vectors out.
let axes = new Float32Array(0);
// Base-layer adjacency (CSR), built on first use and dropped on rebuild.
let adjacency: { offsets: Uint32Array; targets: Uint32Array } | null = null;
// The page's filter (allowed photo ids), kept so it survives a rebuild of the index.
let filterIds: Uint32Array | null = null;

// float16 -> float32 (the rotation is stored at half precision: plenty for rotating queries).
function half(h: number): number {
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  const sign = h & 0x8000 ? -1 : 1;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

// The prebuilt graph records the dimension it was built at, and refuses others. It holds only
// links, which a prefix of the same vectors navigates as well, so it is relabelled with the
// dimension being loaded (the field after the 8-byte magic and the 4-byte version).
function graphFor(graph: Uint8Array, dims: number): Uint8Array {
  const copy = graph.slice();
  new DataView(copy.buffer).setBigUint64(12, BigInt(dims), true);
  return copy;
}

// Downloads the vectors, which are stored dimension-block by dimension-block on the dataset's
// principal axes (pipeline/progressive_vectors.py), and searches as soon as the first block has
// arrived: 64 of 512 dimensions, under 2 MB. Each further block reloads the engine with more
// dimensions and the same prebuilt graph, so results sharpen while the rest streams in.
//
// A repeat visit reads straight from the Cache API rather than through the service worker's fetch
// handler, which had to relay every chunk through its own thread (measured: a cached 15 MB read
// took longer than the network).
async function load(message: Extract<ToWorker, { type: "load" }>) {
  const { stages } = message;
  const ends = stageEnds(message.count, message.dim, stages);
  const total = ends[ends.length - 1];
  fullDim = message.dim;
  count = message.count;
  axes = new Float32Array(fullDim * fullDim);
  scales = new Float32Array(fullDim);
  // The prebuilt graph (~1.3 MB) downloads alongside; if it's missing, the graph is built here.
  const graph = message.graph
    ? cachedFetch(`${message.base}${message.graph}`)
        .then((r) => r.arrayBuffer())
        .then((b) => new Uint8Array(b))
        .catch(() => null)
    : Promise.resolve(null);
  const response = await cachedFetch(`${message.base}${message.vectors}`);
  if (!response.body) throw new Error(`${message.vectors}: empty response`);
  const file = new Uint8Array(total);
  const reader = response.body.getReader();
  let length = 0;
  let stage = 0;
  let prebuilt = true;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (length + value.length > total) throw new Error(`${message.vectors}: larger than the manifest says`);
    file.set(value, length);
    length += value.length;
    post({ type: "progress", stage: "download", done: length, total });
    while (stage < stages.length && length >= ends[stage]) {
      const start = performance.now();
      addBlock(file, stage === 0 ? 0 : ends[stage - 1], stage === 0 ? 0 : stages[stage - 1], stages[stage]);
      const module = await engine;
      const bytes = prebuilt ? await graph : null;
      collection ??= new module.Collection();
      collection.load(codes, scales.subarray(0, dim), dim);
      prebuilt = false;
      if (bytes) {
        try {
          collection.loadGraph(graphFor(bytes, dim), message.M, message.efConstruction);
          prebuilt = true;
        } catch (error) {
          console.warn("Prebuilt graph rejected; building once the vectors are in:", error);
        }
      }
      if (prebuilt && stage === 0) publish(message.M, message.efConstruction, performance.now() - start, true);
      else if (prebuilt) {
        if (filterIds) collection.setFilter(filterIds);
        post({ type: "precision", dims: dim });
      }
      stage++;
    }
  }
  if (length !== total) throw new Error(`${message.vectors}: the download stopped early`);
  if (!prebuilt) build(message.M, message.efConstruction);
}

// Appends the block of dimensions [lo, hi) at `offset` in the file: its scales, its rows of the
// rotation, and its codes, interleaved into the row-major codes of the dimensions so far.
function addBlock(file: Uint8Array, offset: number, lo: number, hi: number) {
  const w = hi - lo;
  scales.set(new Float32Array(file.buffer, offset, w), lo);
  const rotation = new Uint16Array(file.buffer, offset + 4 * w, w * fullDim);
  for (let i = 0; i < rotation.length; i++) axes[lo * fullDim + i] = half(rotation[i]);
  const block = new Int8Array(file.buffer, offset + 4 * w + 2 * w * fullDim, count * w);
  const next = new Int8Array(count * hi);
  for (let i = 0; i < count; i++) {
    if (lo) next.set(codes.subarray(i * lo, (i + 1) * lo), i * hi);
    next.set(block.subarray(i * w, (i + 1) * w), i * hi + lo);
  }
  codes = next;
  dim = hi;
}

// A query (in CLIP's own coordinates) on the leading principal axes the engine has.
function rotate(query: Float32Array): Float32Array {
  if (query.length !== fullDim) throw new Error("query has the wrong dimension");
  const out = new Float32Array(dim);
  for (let j = 0; j < dim; j++) {
    let sum = 0;
    for (let i = 0, row = j * fullDim; i < fullDim; i++) sum += axes[row + i] * query[i];
    out[j] = sum;
  }
  return out;
}

// Builds the graph in slices so the page can animate nodes arriving while every core inserts.
function build(M: number, efConstruction: number) {
  const index = collection!;
  adjacency = null;
  index.beginBuild(M, efConstruction);
  const start = performance.now();
  const slice = Math.max(500, Math.round(count / 60));
  for (let built = 0; built < count; ) {
    built = index.buildStep(slice, threads);
    post({ type: "progress", stage: "build", done: built, total: count });
  }
  publish(M, efConstruction, performance.now() - start, false);
}

// Sends the finished graph's topology to the page (for the 3D view and stats).
function publish(M: number, efConstruction: number, buildMs: number, prebuilt: boolean) {
  const index = collection!;
  if (filterIds) index.setFilter(filterIds); // a new index starts unfiltered
  const info = index.info();
  const edges: Uint32Array[] = [new Uint32Array(0)];
  for (let layer = 1; layer <= info.maxLevel; layer++) edges.push(index.edges(layer));
  const levels = index.levels();
  post(
    {
      type: "built",
      graph: { levels, edges, maxLevel: info.maxLevel, buildMs, memoryBytes: info.memoryBytes, threads, variant, M, efConstruction, prebuilt },
    },
    [levels.buffer, ...edges.map((e) => e.buffer)],
  );
}

// A photo's vector on the principal axes, dequantized exactly as the engine does it (per-dimension
// scale, normalized). Inner products are the same as in CLIP's coordinates.
function vectorOf(id: number): Float32Array {
  if (!(id >= 0 && id < count)) throw new Error(`no photo #${id}`);
  const out = new Float32Array(dim);
  let norm = 0;
  for (let d = 0; d < dim; d++) {
    const x = (codes[id * dim + d] * scales[d]) / 127;
    out[d] = x;
    norm += x * x;
  }
  norm = Math.sqrt(norm) || 1;
  for (let d = 0; d < dim; d++) out[d] /= norm;
  return out;
}

function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let d = 0; d < a.length; d++) sum += a[d] * b[d];
  return sum;
}

function neighborsOf(id: number): Uint32Array {
  if (!adjacency) {
    const edges = collection!.edges(0);
    const offsets = new Uint32Array(count + 1);
    for (let e = 0; e < edges.length; e++) offsets[edges[e] + 1]++;
    for (let i = 0; i < count; i++) offsets[i + 1] += offsets[i];
    const targets = new Uint32Array(edges.length);
    const fill = offsets.slice(0, count);
    for (let e = 0; e < edges.length; e += 2) {
      targets[fill[edges[e]]++] = edges[e + 1];
      targets[fill[edges[e + 1]]++] = edges[e];
    }
    adjacency = { offsets, targets };
  }
  return adjacency.targets.slice(adjacency.offsets[id], adjacency.offsets[id + 1]);
}

function answer(request: Request): { value: unknown; transfer: Transferable[] } {
  if (!collection) throw new Error("the index is not built yet");
  switch (request.method) {
    case "vector": {
      // Back in CLIP's coordinates, to mix with text and image queries.
      const rotated = vectorOf(request.id);
      const vector = new Float32Array(fullDim);
      for (let j = 0; j < dim; j++) for (let i = 0, row = j * fullDim; i < fullDim; i++) vector[i] += rotated[j] * axes[row + i];
      let norm = 0;
      for (let i = 0; i < fullDim; i++) norm += vector[i] * vector[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < fullDim; i++) vector[i] /= norm;
      return { value: vector, transfer: [vector.buffer] };
    }
    case "neighbors": {
      const ids = neighborsOf(request.id);
      const target = vectorOf(request.target);
      const similarities = Float32Array.from(ids, (n) => dot(vectorOf(n), target));
      const value: Neighborhood = { ids, similarities };
      return { value, transfer: [ids.buffer, similarities.buffer] };
    }
    case "similarity":
      return { value: dot(vectorOf(request.a), vectorOf(request.b)), transfer: [] };
    case "evaluate": {
      const value: Evaluation = { ...collection.evaluate(request.samples, 10, request.ef, threads), ef: request.ef };
      return { value, transfer: [] };
    }
    case "filter": {
      filterIds = request.ids;
      return { value: collection.setFilter(request.ids), transfer: [] };
    }
    case "explain": {
      const { query, label, k, ef } = request;
      const value =
        query.kind === "id" ? collection.explainById(query.id, k, label, ef) : collection.explain(rotate(query.vector), k, label, ef);
      return { value, transfer: [] };
    }
  }
}

self.onmessage = async (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      await load(message);
    } else if (message.type === "rebuild") {
      if (!collection) throw new Error("the index is not built yet");
      if (dim < fullDim) throw new Error("the full vectors are still loading");
      build(message.M, message.efConstruction);
    } else if (message.type === "request") {
      const { value, transfer } = answer(message.request);
      post({ type: "reply", seq: message.seq, value }, transfer);
    } else if (message.type === "keywords") {
      const module = await engine;
      keywords ??= new module.Keywords();
      keywords.set(message.texts);
    } else if (message.type === "search") {
      if (!collection) throw new Error("the index is not built yet");
      const { query } = message;
      const words = message.hybrid && query.kind === "vector" && keywords ? keywordsOf(query.text) : "";
      let trace: Trace;
      if (query.kind === "id") trace = collection.searchById(query.id, message.k, message.ef);
      else if (words) {
        trace = collection.hybridSearch(rotate(query.vector), keywords!, words, message.k, message.ef);
        trace.hybrid!.words = words;
      } else trace = collection.search(rotate(query.vector), message.k, message.ef);
      post({ type: "result", seq: message.seq, query, trace });
    }
  } catch (error) {
    const seq = message.type === "search" || message.type === "request" ? message.seq : undefined;
    post({ type: "error", seq, message: error instanceof Error ? error.message : String(error) });
  }
};
