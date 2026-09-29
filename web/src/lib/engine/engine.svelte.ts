// The site-wide engine: dataset, the WebAssembly index (in a worker) and search. One instance for
// the whole app, so the graph is built once and survives navigation between pages.
import type { Evaluation, FromWorker, GraphInfo, Neighborhood, Query, Request, Trace } from "./protocol";

export interface Meta {
  id: string[];
  host: string[];
  path: string[];
  labels: string[][];
  title: string[];
  author: string[];
  landing: string[];
  license: number[];
  licenses: string[];
}

interface Manifest {
  count: number;
  dim: number;
  quantization: { recallAt10VsFloat: number };
}

export type EngineStatus = "idle" | "loading" | "building" | "ready" | "error";

export interface SearchOutcome {
  query: Query;
  trace: Trace;
  ef: number;
}

const DATA = `${import.meta.env.BASE_URL}data/openimages/`;

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

  private worker: Worker | null = null;
  private pending = new Map<number, { resolve: (o: SearchOutcome) => void; reject: (e: Error) => void; ef: number }>();
  private replies = new Map<number, { resolve: (value: never) => void; reject: (e: Error) => void }>();
  private rebuilt: (() => void) | null = null;
  private seq = 0;
  private started: Promise<void> | null = null;

  /** Loads the dataset and builds the index (idempotent). */
  start(): Promise<void> {
    this.started ??= this.load();
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
    return this.start();
  }

  private async load() {
    this.status = "loading";
    this.stage = "Loading the dataset";
    try {
      const [manifest, meta, layout] = await Promise.all([
        fetch(`${DATA}manifest.json`).then((r) => r.json() as Promise<Manifest>),
        fetch(`${DATA}meta.json`).then((r) => r.json() as Promise<Meta>),
        fetch(`${DATA}layout.u16`).then((r) => r.arrayBuffer()),
      ]);
      this.count = manifest.count;
      this.dim = manifest.dim;
      this.quantizationRecall = manifest.quantization.recallAt10VsFloat;
      this.meta = meta;
      this.layout = Float32Array.from(new Uint16Array(layout), (v) => (v / 65535) * 2 - 1);
    } catch (error) {
      this.fail(`Could not load the dataset: ${error}`);
      return;
    }

    await new Promise<void>((resolve) => {
      this.worker = new Worker(new URL("./collection.worker.ts", import.meta.url), { type: "module" });
      this.worker.onmessage = (event: MessageEvent<FromWorker>) => {
        const message = event.data;
        switch (message.type) {
          case "progress":
            if (message.stage === "download") {
              this.stage = `Downloading ${(message.total / 1e6).toFixed(0)} MB of CLIP vectors`;
              this.progress = 0.35 * (message.done / message.total);
            } else if (this.rebuilding) {
              this.rebuildProgress = message.done / message.total;
            } else {
              this.status = "building";
              this.stage = `Building the HNSW graph · ${message.done.toLocaleString()} photos`;
              this.progress = 0.35 + 0.65 * (message.done / message.total);
              this.built = message.done;
            }
            break;
          case "built":
            this.graph = message.graph;
            this.buildMs = message.graph.buildMs;
            this.params = { M: message.graph.M, efConstruction: message.graph.efConstruction };
            this.built = this.count;
            this.progress = 1;
            this.stage = "Ready";
            this.status = "ready";
            this.evaluation = null;
            this.rebuilding = false;
            this.rebuilt?.();
            this.rebuilt = null;
            resolve();
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
      this.worker.onerror = (event) => {
        this.fail(event.message || "The engine worker failed to start.");
        resolve();
      };
      this.worker.postMessage({ type: "load", base: DATA, count: this.count, dim: this.dim, M: 16, efConstruction: 128 });
    });
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
      this.worker!.postMessage({ type: "search", seq, query, k, ef });
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

  /** Rebuilds the whole index with new graph parameters, on every core. */
  rebuild(M: number, efConstruction: number): Promise<void> {
    if (!this.worker || this.status !== "ready" || this.rebuilding) return Promise.resolve();
    this.rebuilding = true;
    this.rebuildProgress = 0;
    this.error = "";
    return new Promise((resolve) => {
      this.rebuilt = resolve;
      this.worker!.postMessage({ type: "rebuild", M, efConstruction });
    });
  }

  randomId(): number {
    return Math.floor(Math.random() * this.count);
  }

  // ---- Photo metadata helpers ----

  thumbnail(id: number, size: "q" | "n" | "z" = "n"): string {
    const m = this.meta!;
    return `https://c${m.host[id]}.staticflickr.com/${m.path[id]}_${size}.jpg`;
  }

  caption(id: number): string {
    const m = this.meta!;
    return m.labels[id].length ? m.labels[id].join(" · ") : m.title[id] || "Untitled";
  }

  author(id: number): string {
    return this.meta?.author[id] || "Unknown";
  }

  source(id: number): string {
    return this.meta?.landing[id] ?? "";
  }
}

export const engine = new Engine();
