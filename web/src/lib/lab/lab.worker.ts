// Runs the Epzero engine (C++ compiled to WebAssembly) off the main thread, so the page stays at
// 60 fps while the engine saturates every core.
import { UNSUPPORTED_BROWSER, wasmSimdSupported } from "../support";
import type { FromWorker, KernelRow, LabConfig, SweepRow, ToWorker } from "./protocol";
import { norms, readVectors, VectorFileError, type Vectors } from "./vectors";

interface Lab {
  generate(n: number, queries: number, dim: number, clusters: number, seed: number): number;
  baseBuffer(n: number, dim: number): Float32Array;
  queryBuffer(n: number): Float32Array;
  useMetric(metric: string): number;
  computeTruth(k: number, threads: number): number;
  beginBuild(M: number, efConstruction: number): void;
  buildStep(count: number, threads: number): number;
  sweep(ef: number, k: number): SweepRow;
  bruteForceQps(k: number, maxQueries: number): number;
  kernelBench(dim: number): KernelRow;
  info(): { simd: string; buildMs: number; memoryBytes: number; maxLevel: number };
  delete(): void;
}

interface EpzeroModule {
  Lab: new () => Lab;
}

const post = (message: FromWorker) => (self as unknown as Worker).postMessage(message);

const variant: "mt" | "st" = self.crossOriginIsolated ? "mt" : "st";
const threads = variant === "mt" ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;

async function loadEngine(): Promise<EpzeroModule> {
  const url = `${import.meta.env.BASE_URL}wasm/${variant}/epzero.js`;
  const { default: createEpzero } = await import(/* @vite-ignore */ url);
  return createEpzero();
}

const enginePromise = loadEngine();
enginePromise.then(
  (engine) => {
    const lab = new engine.Lab();
    post({ type: "ready", info: { simd: lab.info().simd, threads, variant, isolated: self.crossOriginIsolated } });
    lab.delete();
  },
  (error) =>
    post({
      type: "error",
      message: wasmSimdSupported() ? `Could not load the engine: ${error}` : UNSUPPORTED_BROWSER,
    }),
);

/** Queries to measure with: 100 to 1,000, fewer for bigger data (see loadOwn). */
function queryBudget(n: number, dim: number): number {
  return Math.max(100, Math.min(1000, Math.floor(4e10 / (n * dim))));
}

// The visitor's files, read when chosen.
const files: { base: Vectors | null; queries: Vectors | null } = { base: null, queries: null };

async function inspect(role: "base" | "queries", file: File) {
  try {
    const vectors = await readVectors(file);
    if (role === "queries" && files.base && vectors.dim !== files.base.dim)
      throw new VectorFileError(`The queries have ${vectors.dim} dimensions, the vectors ${files.base.dim}.`);
    files[role] = vectors;
    if (role === "base" && files.queries && files.queries.dim !== vectors.dim) files.queries = null;
    const { mean, unit } = norms(vectors);
    post({
      type: "inspected",
      role,
      info: { name: file.name, bytes: file.size, n: vectors.n, dim: vectors.dim, format: vectors.format, meanNorm: mean, unit },
    });
  } catch (error) {
    files[role] = null;
    post({
      type: "inspect-error",
      role,
      message: error instanceof VectorFileError ? error.message : `Couldn't read ${file.name}: ${error instanceof Error ? error.message : error}`,
    });
  }
}

// Copies the visitor's vectors into the engine: the base, and the queries (their own file, or a
// random sample held out of the base so no query is also an answer).
function loadOwn(lab: Lab, config: LabConfig): { n: number; queries: number; dim: number } {
  const base = files.base!;
  const { dim } = base;
  // Exact ground truth costs queries x vectors x dimensions: keep it to ~10-20 s in a browser.
  const budget = queryBudget(base.n, dim);
  let queryRows: number[] = [];
  let baseRows: number[] | null = null;
  if (!files.queries) {
    const count = Math.max(1, Math.min(config.holdout ?? budget, budget, Math.floor(base.n / 10)));
    const order = Array.from({ length: base.n }, (_, i) => i);
    let seed = 42;
    for (let i = order.length - 1; i > 0; i--) {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      const j = seed % (i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }
    queryRows = order.slice(0, count).sort((a, b) => a - b);
    const held = new Uint8Array(base.n);
    for (const i of queryRows) held[i] = 1;
    baseRows = order.filter((i) => !held[i]).sort((a, b) => a - b);
  }
  const n = baseRows ? baseRows.length : base.n;
  const view = lab.baseBuffer(n, dim);
  if (baseRows) baseRows.forEach((row, i) => view.set(base.data.subarray(row * dim, (row + 1) * dim), i * dim));
  else view.set(base.data);
  const queries = files.queries;
  const nq = queries ? Math.min(queries.n, budget) : queryRows.length;
  const qview = lab.queryBuffer(nq);
  if (queries) qview.set(queries.data.subarray(0, nq * dim));
  else queryRows.forEach((row, i) => qview.set(base.data.subarray(row * dim, (row + 1) * dim), i * dim));
  lab.useMetric(config.metric ?? "l2");
  return { n, queries: nq, dim };
}

async function run(config: LabConfig) {
  const engine = await enginePromise;
  const lab = new engine.Lab();
  try {
    if (config.source === "own") {
      if (!files.base) throw new Error("Choose a file of vectors first.");
      post({ type: "stage", stage: "load", state: "running" });
      const start = performance.now();
      const loaded = loadOwn(lab, config);
      config.n = loaded.n;
      post({ type: "loaded", ...loaded });
      post({ type: "stage", stage: "load", state: "done", ms: performance.now() - start });
    } else {
      post({ type: "stage", stage: "generate", state: "running" });
      const genMs = lab.generate(config.n, config.queries, config.dim, config.clusters, 42);
      post({ type: "stage", stage: "generate", state: "done", ms: genMs });
    }

    post({ type: "stage", stage: "truth", state: "running" });
    const truthMs = lab.computeTruth(config.k, threads);
    post({ type: "stage", stage: "truth", state: "done", ms: truthMs });

    // Build in slices so progress streams to the page while every core inserts.
    post({ type: "stage", stage: "build", state: "running" });
    lab.beginBuild(config.M, config.efConstruction);
    const slice = Math.max(1000, Math.round(config.n / 40));
    const start = performance.now();
    for (let built = 0; built < config.n; ) {
      built = lab.buildStep(slice, threads);
      const elapsed = (performance.now() - start) / 1000;
      post({ type: "progress", built, total: config.n, insertsPerSec: built / elapsed });
    }
    const info = lab.info();
    post({ type: "stage", stage: "build", state: "done", ms: info.buildMs });
    post({ type: "summary", buildMs: info.buildMs, memoryBytes: info.memoryBytes, layers: info.maxLevel + 1 });

    post({ type: "stage", stage: "baseline", state: "running" });
    const bfQps = lab.bruteForceQps(config.k, 60);
    post({ type: "baseline", qps: bfQps });
    post({ type: "stage", stage: "baseline", state: "done" });

    post({ type: "stage", stage: "sweep", state: "running" });
    for (const ef of config.efs) {
      const row = lab.sweep(ef, config.k);
      if (Number.isFinite(row.qps) && row.qps > 0) post({ type: "sweep", row });
    }
    post({ type: "stage", stage: "sweep", state: "done" });

    post({ type: "stage", stage: "kernel", state: "running" });
    for (const dim of [64, 128, 384, 768]) post({ type: "kernel", row: lab.kernelBench(dim) });
    post({ type: "stage", stage: "kernel", state: "done" });
    post({ type: "done" });
  } catch (error) {
    post({ type: "error", message: String(error) });
  } finally {
    lab.delete();
  }
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
  const message = event.data;
  if (message.type === "run") void run(message.config);
  else if (message.type === "inspect") void inspect(message.role, message.file);
  else files[message.role] = null;
};
