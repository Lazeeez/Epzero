// Runs the Oryx engine (C++ compiled to WebAssembly) off the main thread, so the page stays at
// 60 fps while the engine saturates every core.
import type { FromWorker, KernelRow, LabConfig, SweepRow, ToWorker } from "./protocol";

interface Lab {
  generate(n: number, queries: number, dim: number, clusters: number, seed: number): number;
  computeTruth(k: number, threads: number): number;
  beginBuild(M: number, efConstruction: number): void;
  buildStep(count: number, threads: number): number;
  sweep(ef: number, k: number): SweepRow;
  bruteForceQps(k: number, maxQueries: number): number;
  kernelBench(dim: number): KernelRow;
  info(): { simd: string; buildMs: number; memoryBytes: number; maxLevel: number };
  delete(): void;
}

interface OryxModule {
  Lab: new () => Lab;
}

const post = (message: FromWorker) => (self as unknown as Worker).postMessage(message);

const variant: "mt" | "st" = self.crossOriginIsolated ? "mt" : "st";
const threads = variant === "mt" ? Math.min(navigator.hardwareConcurrency || 4, 16) : 1;

async function loadEngine(): Promise<OryxModule> {
  const url = `${import.meta.env.BASE_URL}wasm/${variant}/oryx.js`;
  const { default: createOryx } = await import(/* @vite-ignore */ url);
  return createOryx();
}

const enginePromise = loadEngine();
enginePromise.then(
  (engine) => {
    const lab = new engine.Lab();
    post({ type: "ready", info: { simd: lab.info().simd, threads, variant, isolated: self.crossOriginIsolated } });
    lab.delete();
  },
  (error) => post({ type: "error", message: `Could not load the engine: ${error}` }),
);

async function run(config: LabConfig) {
  const engine = await enginePromise;
  const lab = new engine.Lab();
  try {
    post({ type: "stage", stage: "generate", state: "running" });
    const genMs = lab.generate(config.n, config.queries, config.dim, config.clusters, 42);
    post({ type: "stage", stage: "generate", state: "done", ms: genMs });

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
      post({ type: "sweep", row: lab.sweep(ef, config.k) });
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
  if (event.data.type === "run") void run(event.data.config);
};
