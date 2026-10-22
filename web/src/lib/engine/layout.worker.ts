// Lays out the visitor's own photos on a 2-D map for the 3D view: UMAP over their CLIP vectors,
// as pipeline/build_dataset.py does for the demo set, then centred and scaled to [-1, 1].
import { UMAP } from "umap-js";

export type ToLayout = { vectors: Float32Array; count: number; dim: number };
export type FromLayout = { type: "progress"; done: number } | { type: "layout"; xy: Float32Array } | { type: "error"; message: string };

const post = (message: FromLayout, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(message, transfer);

// A seeded generator, so the same photos always land in the same places.
function random(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

self.onmessage = async (event: MessageEvent<ToLayout>) => {
  const { vectors, count, dim } = event.data;
  try {
    const xy = new Float32Array(count * 2);
    if (count < 8) {
      // Too few for a neighbour graph: a circle reads fine.
      for (let i = 0; i < count; i++) {
        const angle = (2 * Math.PI * i) / Math.max(count, 1);
        xy[i * 2] = count > 1 ? Math.cos(angle) : 0;
        xy[i * 2 + 1] = count > 1 ? Math.sin(angle) : 0;
      }
    } else {
      const rows = Array.from({ length: count }, (_, i) => Array.from(vectors.subarray(i * dim, (i + 1) * dim)));
      // The vectors are unit length, so Euclidean distance ranks pairs as cosine does.
      const umap = new UMAP({ nComponents: 2, nNeighbors: Math.min(30, Math.max(4, Math.round(count / 20)), count - 1), minDist: 0.08, random: random(42) });
      const epochs = umap.initializeFit(rows);
      for (let epoch = 0; epoch < epochs; epoch++) {
        umap.step();
        if (epoch % 10 === 0) post({ type: "progress", done: epoch / epochs });
      }
      const embedding = umap.getEmbedding();
      let mx = 0;
      let my = 0;
      for (const [x, y] of embedding) {
        mx += x / count;
        my += y / count;
      }
      let scale = 0;
      for (const [x, y] of embedding) scale = Math.max(scale, Math.abs(x - mx), Math.abs(y - my));
      scale ||= 1;
      embedding.forEach(([x, y], i) => {
        xy[i * 2] = (x - mx) / scale;
        xy[i * 2 + 1] = (y - my) / scale;
      });
    }
    post({ type: "layout", xy }, [xy.buffer]);
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
