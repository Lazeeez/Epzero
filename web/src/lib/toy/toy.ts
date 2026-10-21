// The landing page's teaching index: a few hundred 2-D points in the real Epzero engine (the
// single-threaded WebAssembly build, loaded on the main thread; it builds in a few milliseconds).
import type { Trace } from "../engine/protocol";

export interface ToyIndex {
  build(points: Float32Array, dim: number, M: number, efConstruction: number, seed: number): void;
  levels(): Uint8Array;
  edges(layer: number): Uint32Array;
  maxLevel(): number;
  trace(query: Float32Array, k: number, ef: number): Trace;
}

export interface ToyGraph {
  index: ToyIndex;
  /** Interleaved x, y in [0, 1]. */
  points: Float32Array;
  levels: Uint8Array;
  /** edges[layer] = flat [a0, b0, a1, b1, ...] */
  edges: Uint32Array[];
  maxLevel: number;
}

let module: Promise<{ Toy: new () => ToyIndex }> | null = null;

function engine() {
  module ??= import(/* @vite-ignore */ `${import.meta.env.BASE_URL}wasm/st/epzero.js`).then((m) => m.default());
  return module;
}

// Deterministic PRNG so the demo looks the same on every visit.
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Clustered points, like real embeddings: dense groups with sparse space between them. */
export function makePoints(count: number, seed = 7): Float32Array {
  const random = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
  const clusters = Array.from({ length: 8 }, () => ({
    x: 0.14 + random() * 0.72,
    y: 0.14 + random() * 0.72,
    spread: 0.035 + random() * 0.06,
  }));
  const out = new Float32Array(count * 2);
  for (let i = 0; i < count; i++) {
    let x: number;
    let y: number;
    if (random() < 0.14) {
      x = random();
      y = random();
    } else {
      const c = clusters[Math.floor(random() * clusters.length)];
      x = c.x + gauss() * c.spread;
      y = c.y + gauss() * c.spread;
    }
    out[i * 2] = Math.min(0.97, Math.max(0.03, x));
    out[i * 2 + 1] = Math.min(0.97, Math.max(0.03, y));
  }
  return out;
}

export async function buildToy(count = 640, M = 6, efConstruction = 64): Promise<ToyGraph> {
  const { Toy } = await engine();
  const index = new Toy();
  const points = makePoints(count);
  index.build(points, 2, M, efConstruction, 42);
  const maxLevel = index.maxLevel();
  const edges: Uint32Array[] = [];
  for (let layer = 0; layer <= maxLevel; layer++) edges.push(index.edges(layer));
  return { index, points, levels: index.levels(), edges, maxLevel };
}
