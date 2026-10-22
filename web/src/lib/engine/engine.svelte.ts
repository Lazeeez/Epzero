// The site-wide engine: dataset, the WebAssembly index (in a worker) and search. One instance for
// the whole app, so the graph is built once and survives navigation between pages.
import { cachedFetch, pruneCache } from "../cached-fetch";
import { isolationSettled } from "../isolation";
import { UNSUPPORTED_BROWSER, wasmSimdSupported } from "../support";
import { release, type OwnCollection } from "./own-photos.svelte";
import { stageEnds, type Evaluation, type Explanation, type FilterInfo, type FromWorker, type GraphInfo, type Neighborhood, type Query, type Request, type Trace } from "./protocol";

/** Photo metadata, column-oriented (format 2, written by pipeline/build_dataset.py). */
export interface Meta {
  format: 2;
  host: number[];
  path: string[];
  /** Every label once; `labels` holds indices into it. */
  labelNames: string[];
  labels: number[][];
  /** Only for photos without labels (elsewhere ""): their caption. */
  title: string[];
  author: string[];
  /** Flickr account for the photo page, or the page's full URL. */
  owner: string[];
  license: number[];
  licenses: string[];
}

interface Manifest {
  count: number;
  dim: number;
  quantization: { recallAt10VsFloat: number };
  files: { vectors: string; layout: string; graph?: string; meta?: string };
  graph?: { M: number; efConstruction: number };
  /** The vector file's stages (pipeline/progressive_vectors.py): searchable at each dimension count. */
  vectors: { stages: VectorStage[] };
}

export interface VectorStage {
  dims: number;
  /** Recall@10 of an exact search at this many dimensions against the float vectors, for photo
   * queries (text queries need more of the dimensions; see pipeline/progressive_vectors.py). */
  recallAt10VsFloat: number;
}

export type EngineStatus = "idle" | "loading" | "building" | "ready" | "error";

/** A category filter: photos carrying any of these labels. */
export interface ActiveFilter extends FilterInfo {
  labels: string[];
  /** Allowed photo ids, for the 3D view. */
  ids: Uint32Array;
}

export interface Category {
  label: string;
  count: number;
}

export interface SearchOutcome {
  query: Query;
  trace: Trace;
  ef: number;
}

const DATA = `${import.meta.env.BASE_URL}data/openimages/`;

const HYBRID_KEY = "epzero-hybrid";
function readHybrid(): boolean {
  try {
    return localStorage.getItem(HYBRID_KEY) !== "0";
  } catch {
    return true;
  }
}

// Flickr titles arrive HTML-escaped (&quot;, &amp;, &#39;...).
const ENTITIES: Record<string, string> = { amp: "&", quot: '"', apos: "'", lt: "<", gt: ">", nbsp: " " };
function decodeEntities(text: string): string {
  return text.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (whole, code: string) => {
    if (code[0] === "#") {
      const n = code[1] === "x" || code[1] === "X" ? parseInt(code.slice(2), 16) : parseInt(code.slice(1), 10);
      return Number.isFinite(n) ? String.fromCodePoint(n) : whole;
    }
    return ENTITIES[code.toLowerCase()] ?? whole;
  });
}

class Engine {
  status = $state<EngineStatus>("idle");
  /** 0..1 across download + build */
  progress = $state(0);
  stage = $state("");
  error = $state("");
  count = $state(0);
  dim = $state(512);
  graph = $state<GraphInfo | null>(null);
  buildMs = $state(0);
  meta = $state.raw<Meta | null>(null);
  layout = $state.raw<Float32Array | null>(null);
  quantizationRecall = $state(0);
  /** Dimensions searches use now: fewer than `dim` while the rest of the vectors stream in. */
  dims = $state(0);
  /** The stages the vectors arrive in, and the share of their bytes downloaded so far. */
  stages = $state.raw<VectorStage[]>([]);
  streamed = $state(0);
  /** Nodes inserted so far during the build (drives the build animation). */
  built = $state(0);
  last = $state.raw<SearchOutcome | null>(null);
  /** Graph parameters of the current index. */
  params = $state({ M: 16, efConstruction: 128 });
  /** A rebuild with new parameters is running (searches wait for it). */
  rebuilding = $state(false);
  rebuildProgress = $state(0);
  /** Recall and speed of the current graph, measured in this tab after each (re)build. */
  evaluation = $state.raw<Evaluation | null>(null);
  evaluating = $state(false);
  /** The browser can't run the engine at all (no WebAssembly SIMD): retrying won't help. */
  unsupported = $state(false);
  /** The category filter every search honors (null: all photos). */
  filter = $state.raw<ActiveFilter | null>(null);
  /** Hybrid search: text queries also match their words against the photos' captions. */
  hybrid = $state(readHybrid());
  /** Which photos the engine indexes: the demo set, or the visitor's own (own-photos.svelte.ts). */
  source = $state<"demo" | "mine">("demo");
  /** The visitor's photos, once chosen (kept while the demo is shown, to switch back quickly). */
  own = $state.raw<OwnCollection | null>(null);
  /** Changes whenever a different set of photos is indexed. */
  collection = $state(0);
  /** The visitor chose their own photos: the playground returns to them after other pages used the demo. */
  preferOwn = false;

  private worker: Worker | null = null;
  private pending = new Map<number, { resolve: (o: SearchOutcome) => void; reject: (e: Error) => void; ef: number }>();
  private replies = new Map<number, { resolve: (value: never) => void; reject: (e: Error) => void }>();
  private rebuilt: (() => void) | null = null;
  private keywordsSent = false;
  private seq = 0;
  private started: Promise<void> | null = null;

  /** Loads the dataset and builds the index (idempotent). */
  start(): Promise<void> {
    this.started ??= isolationSettled().then(() => this.load());
    return this.started;
  }

  /** After a failed load (e.g. the network dropped mid-download), starts over. */
  retry(): Promise<void> {
    if (this.status !== "error") return this.start();
    this.worker?.terminate();
    this.worker = null;
    this.started = null;
    this.error = "";
    this.progress = 0;
    this.built = 0;
    this.dims = 0;
    this.streamed = 0;
    return this.start();
  }

  private graphFile: string | undefined;
  private vectorsFile = "";
  private graphParams = { M: 16, efConstruction: 128 };

  private async load() {
    if (!wasmSimdSupported()) {
      this.fail(UNSUPPORTED_BROWSER);
      this.unsupported = true;
      return;
    }
    this.status = "loading";
    this.stage = "Loading the dataset";
    // Only the tiny manifest is needed before the big downloads start. The vectors (in the worker),
    // photo metadata and map layout then download in parallel, so the metadata (2 MB) is no longer
    // on the critical path; the index is "ready" once the graph and the metadata are both there.
    let side: Promise<void>;
    try {
      const manifest = (await (await cachedFetch(`${DATA}manifest.json`, { fresh: true })).json()) as Manifest;
      this.count = manifest.count;
      this.dim = manifest.dim;
      this.quantizationRecall = manifest.quantization.recallAt10VsFloat;
      this.graphFile = manifest.files.graph;
      this.vectorsFile = manifest.files.vectors;
      this.stages = manifest.vectors.stages;
      // Returning visitors' caches may still hold files this manifest replaced (e.g. 15 MB of
      // vectors in the previous format).
      void pruneCache(DATA, ["manifest.json", ...Object.values(manifest.files)]);
      if (manifest.graph) this.graphParams = { M: manifest.graph.M, efConstruction: manifest.graph.efConstruction };
      side = Promise.all([
        cachedFetch(`${DATA}${manifest.files.meta ?? "meta.json"}`)
          .then((r) => r.json() as Promise<Meta>)
          .then((meta) => void (this.meta = meta)),
        cachedFetch(`${DATA}layout.u16`)
          .then((r) => r.arrayBuffer())
          .then((layout) => void (this.layout = Float32Array.from(new Uint16Array(layout), (v) => (v / 65535) * 2 - 1))),
      ]).then(() => {});
    } catch (error) {
      this.fail(`Could not load the dataset: ${error}`);
      return;
    }

    await new Promise<void>((resolve) => {
      side.catch((error) => {
        this.fail(`Could not load the dataset: ${error}`);
        resolve();
      });
      this.spawn(side, resolve).postMessage({
        type: "load",
        base: DATA,
        vectors: this.vectorsFile,
        stages: this.stages.map((s) => s.dims),
        count: this.count,
        dim: this.dim,
        graph: this.graphFile,
        ...this.graphParams,
      });
    });
  }

  // Starts an engine worker; `resolve` once its first graph is ready (and `side`, the metadata).
  private spawn(side: Promise<void>, resolve: () => void): Worker {
    const worker = new Worker(new URL("./collection.worker.ts", import.meta.url), { type: "module" });
    this.worker = worker;
    worker.onmessage = (event: MessageEvent<FromWorker>) => {
      if (worker !== this.worker) return; // replaced by a switch to other photos
      const message = event.data;
      switch (message.type) {
        case "progress":
          if (message.stage === "download") {
            this.streamed = message.done / message.total;
            if (this.status === "ready") break;
            // With a prebuilt graph, search starts once the first block of dimensions is in;
            // without one, the graph is built from the whole file.
            const needed = this.graphFile ? stageEnds(this.count, this.dim, this.stages.map((s) => s.dims))[0] : message.total;
            this.stage = `Downloading ${(needed / 1e6).toFixed(1)} MB of CLIP vectors`;
            this.progress = (this.graphFile ? 0.95 : 0.35) * Math.min(1, message.done / needed);
          } else if (this.rebuilding) {
            this.rebuildProgress = message.done / message.total;
          } else {
            this.status = "building";
            this.stage = `Building the HNSW graph · ${message.done.toLocaleString()} photos`;
            this.progress = 0.35 + 0.65 * (message.done / message.total);
            this.built = message.done;
          }
          break;
        case "built": {
          const graph = message.graph;
          // Captions and thumbnails need the metadata: go ready once it has arrived too.
          side.then(() => {
            if (worker !== this.worker) return;
            this.graph = graph;
            this.dims ||= this.graphFile ? this.stages[0].dims : this.dim;
            if (!graph.prebuilt) this.dims = this.dim;
            this.buildMs = graph.buildMs;
            this.params = { M: graph.M, efConstruction: graph.efConstruction };
            this.built = this.count;
            this.progress = 1;
            this.stage = "Ready";
            this.status = "ready";
            this.evaluation = null;
            this.rebuilding = false;
            this.rebuilt?.();
            this.rebuilt = null;
            this.sendKeywords();
            resolve();
          });
          break;
        }
        case "precision":
          this.dims = message.dims;
          break;
        case "reply": {
          const waiter = this.replies.get(message.seq);
          this.replies.delete(message.seq);
          waiter?.resolve(message.value as never);
          break;
        }
        case "result": {
          const waiter = this.pending.get(message.seq);
          if (!waiter) return;
          this.pending.delete(message.seq);
          const outcome = { query: message.query, trace: message.trace, ef: waiter.ef };
          this.last = outcome;
          waiter.resolve(outcome);
          break;
        }
        case "error":
          if (message.seq !== undefined) {
            // One query failed; the others are unaffected.
            this.pending.get(message.seq)?.reject(new Error(message.message));
            this.pending.delete(message.seq);
            this.replies.get(message.seq)?.reject(new Error(message.message));
            this.replies.delete(message.seq);
          } else if (this.rebuilding) {
            this.rebuilding = false;
            this.rebuilt = null;
            this.error = message.message;
          } else if (this.status !== "ready") {
            this.fail(message.message);
            resolve();
          } else {
            for (const waiter of this.pending.values()) waiter.reject(new Error(message.message));
            this.pending.clear();
          }
          break;
      }
    };
    worker.onerror = (event) => {
      if (worker !== this.worker) return;
      this.fail(event.message || "The engine worker failed to start.");
      resolve();
    };
    return worker;
  }

  // Drops the current index and everything derived from it, before another set of photos loads.
  private reset() {
    this.worker?.terminate();
    this.worker = null;
    for (const waiter of this.pending.values()) waiter.reject(new Error("The photos changed"));
    this.pending.clear();
    for (const waiter of this.replies.values()) waiter.reject(new Error("The photos changed"));
    this.replies.clear();
    this.rebuilt = null;
    this.rebuilding = false;
    this.error = "";
    this.progress = 0;
    this.built = 0;
    this.dims = 0;
    this.streamed = 0;
    this.graph = null;
    this.last = null;
    this.evaluation = null;
    this.filter = null;
    this.categoryList = null;
    this.keywordsSent = false;
    this.started = null;
    this.collection++;
  }

  /** Indexes the visitor's own photos instead of the demo set: built here, on every core. */
  useOwn(own: OwnCollection): Promise<void> {
    if (this.own && this.own !== own) release(this.own);
    this.reset();
    this.own = own;
    this.source = "mine";
    this.preferOwn = true;
    const count = own.photos.length;
    this.count = count;
    this.dim = own.dim;
    this.stages = [{ dims: own.dim, recallAt10VsFloat: 1 }];
    this.quantizationRecall = 1;
    this.graphFile = undefined;
    this.meta = null;
    this.layout = own.layout;
    this.status = "building";
    this.stage = `Building the HNSW graph · ${count.toLocaleString()} photos`;
    // A small collection gets a denser graph: it costs little and keeps recall near perfect.
    const params = { M: 16, efConstruction: 200 };
    this.started = new Promise<void>((resolve) => {
      const vectors = own.vectors.slice(); // the worker takes this copy; the collection keeps its own
      this.spawn(Promise.resolve(), resolve).postMessage({ type: "own", vectors, count, dim: own.dim, ...params }, [vectors.buffer]);
    });
    return this.started;
  }

  /** Back to the demo photos (read from cache, with their prebuilt graph: a second or two). */
  useDemo(): Promise<void> {
    if (this.source === "demo") return this.start();
    this.reset();
    this.source = "demo";
    this.layout = null;
    this.meta = null;
    this.status = "idle";
    return this.start();
  }

  /** Forgets the visitor's photos on this page (not the copy kept in the browser). */
  dropOwn() {
    this.preferOwn = false;
    if (this.source === "mine") void this.useDemo();
    release(this.own);
    this.own = null;
  }

  // The captions, one line per photo (its title, or its labels), for the worker's keyword index.
  // Sent once, after the first build: a few hundred kilobytes, indexed in WebAssembly in the worker.
  // Built when the browser is idle, so it never competes with the page's first moments.
  private sendKeywords() {
    if (this.keywordsSent || !this.worker || !this.meta || this.source === "mine") return;
    this.keywordsSent = true;
    const send = () => {
      const m = this.meta!;
      const lines = m.title.map((title, id) =>
        (title ? decodeEntities(title) : m.labels[id].map((l) => m.labelNames[l]).join(" ")).replace(/\s+/g, " "),
      );
      this.worker?.postMessage({ type: "keywords", texts: lines.join("\n") });
    };
    if ("requestIdleCallback" in window) requestIdleCallback(send, { timeout: 3000 });
    else setTimeout(send, 500);
  }

  /** Turns hybrid search on or off (remembered on this device). */
  setHybrid(on: boolean) {
    this.hybrid = on;
    try {
      localStorage.setItem(HYBRID_KEY, on ? "1" : "0");
    } catch {
      /* private mode: just not remembered */
    }
  }

  private fail(message: string) {
    this.status = "error";
    this.error = message;
    this.stage = message;
  }

  /** Runs a search; resolves with the full traversal trace and an exact-scan comparison. */
  search(query: Query, ef = 64, k = 12): Promise<SearchOutcome> {
    if (!this.worker || this.status !== "ready") return Promise.reject(new Error("The index is not ready yet"));
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(seq, { resolve, reject, ef });
      const hybrid = this.hybrid && query.kind === "vector" && query.source === "text";
      this.worker!.postMessage({ type: "search", seq, query, k, ef, hybrid });
    });
  }

  private request<T>(request: Request): Promise<T> {
    if (!this.worker || this.status !== "ready") return Promise.reject(new Error("The index is not ready yet"));
    const seq = ++this.seq;
    return new Promise<T>((resolve, reject) => {
      this.replies.set(seq, { resolve: resolve as (value: never) => void, reject });
      this.worker!.postMessage({ type: "request", seq, request });
    });
  }

  /** A photo's normalized vector (for blending it with words). */
  vector(id: number): Promise<Float32Array> {
    return this.request({ method: "vector", id });
  }

  /** A photo's base-layer neighbours and how similar each is to `target`. */
  neighbors(id: number, target: number): Promise<Neighborhood> {
    return this.request({ method: "neighbors", id, target });
  }

  /** Why `label` is or isn't among the results of `query` at this beam width (honoring the filter). */
  explain(query: Query, label: number, ef: number, k = 12): Promise<Explanation> {
    return this.request({ method: "explain", query, label, k, ef });
  }

  similarity(a: number, b: number): Promise<number> {
    return this.request({ method: "similarity", a, b });
  }

  /** Measures recall@10 and single-thread QPS of the current graph on held-out queries. */
  async evaluate(ef = 64, samples = 300): Promise<Evaluation> {
    this.evaluating = true;
    try {
      const result = await this.request<Evaluation>({ method: "evaluate", ef, samples });
      this.evaluation = result;
      return result;
    } finally {
      this.evaluating = false;
    }
  }

  /** Searches use only some of the vectors' dimensions while the rest download. */
  get sharpening(): boolean {
    return this.status === "ready" && this.dims < this.dim;
  }

  /** Recall@10 an exact search reaches at the current dimension count (vs. float vectors). */
  get stageRecall(): number {
    return this.stages.find((s) => s.dims === this.dims)?.recallAt10VsFloat ?? this.quantizationRecall;
  }

  /** Rebuilds the whole index with new graph parameters, on every core. */
  rebuild(M: number, efConstruction: number): Promise<void> {
    if (!this.worker || this.status !== "ready" || this.rebuilding || this.sharpening) return Promise.resolve();
    this.rebuilding = true;
    this.rebuildProgress = 0;
    this.error = "";
    return new Promise((resolve) => {
      this.rebuilt = resolve;
      this.worker!.postMessage({ type: "rebuild", M, efConstruction });
    });
  }

  randomId(): number {
    const ids = this.filter?.ids;
    if (ids?.length) return ids[Math.floor(Math.random() * ids.length)];
    return Math.floor(Math.random() * this.count);
  }

  private categoryList: Category[] | null = null;

  /** Every label in the dataset with how many photos carry it, most common first. */
  categories(): Category[] {
    if (this.source === "mine") return [];
    if (this.categoryList || !this.meta) return this.categoryList ?? [];
    const counts = new Map<string, number>();
    const names = this.meta.labelNames;
    for (const labels of this.meta.labels) for (const label of labels) counts.set(names[label], (counts.get(names[label]) ?? 0) + 1);
    this.categoryList = [...counts].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count);
    return this.categoryList;
  }

  /** Restricts every following search to photos with any of `labels` (none: all photos). */
  async setFilter(labels: string[]): Promise<void> {
    if (!this.meta) return;
    if (labels.length === 0) {
      await this.request<FilterInfo>({ method: "filter", ids: null });
      this.filter = null;
      return;
    }
    const wanted = new Set(labels.map((name) => this.meta!.labelNames.indexOf(name)));
    const ids: number[] = [];
    this.meta.labels.forEach((photo, id) => {
      if (photo.some((label) => wanted.has(label))) ids.push(id);
    });
    const allowed = Uint32Array.from(ids);
    const info = await this.request<FilterInfo>({ method: "filter", ids: allowed });
    this.filter = { ...info, labels, ids: allowed };
  }

  // ---- Photo metadata helpers ----

  // Every photo from live.staticflickr.com (the same files as the old per-farm cN hosts, which
  // the path's first part names): one host means one connection for a whole page of results,
  // instead of a DNS lookup and TLS handshake for each of eight hosts.
  thumbnail(id: number, size: "q" | "n" | "z" = "n"): string {
    if (this.source === "mine") return this.own?.photos[id]?.thumb ?? "";
    if (!this.meta) return "";
    const path = this.meta!.path[id];
    return `https://live.staticflickr.com/${path.slice(path.indexOf("/") + 1)}_${size}.jpg`;
  }

  caption(id: number): string {
    if (this.source === "mine") return this.own?.photos[id]?.name ?? "";
    if (!this.meta) return "";
    const m = this.meta!;
    return m.labels[id].length ? m.labels[id].map((l) => m.labelNames[l]).join(" · ") : decodeEntities(m.title[id]) || "Untitled";
  }

  author(id: number): string {
    if (this.source === "mine") return "you";
    return this.meta?.author[id] || "Unknown";
  }

  /** Where to see the photo in full: its page on Flickr, or the visitor's own file ("" if neither). */
  original(id: number): string {
    if (this.source === "mine") {
      const file = this.own?.photos[id]?.file;
      if (!file) return "";
      if (!originals.has(file)) originals.set(file, URL.createObjectURL(file));
      return originals.get(file)!;
    }
    const m = this.meta;
    if (!m) return "";
    const owner = m.owner[id];
    if (owner.startsWith("http")) return owner;
    const photo = m.path[id].slice(m.path[id].lastIndexOf("/") + 1).split("_")[0];
    return `https://www.flickr.com/photos/${owner}/${photo}/`;
  }
}

// Object URLs of the visitor's originals, made on first use (opening a photo in full).
const originals = new WeakMap<File, string>();

export const engine = new Engine();
