// Messages between the app and its engine worker.

export interface Trace {
  entryLabel: number;
  entryLayer: number;
  layer: Uint8Array;
  from: Uint32Array;
  to: Uint32Array;
  distance: Float32Array;
  accepted: Uint8Array;
  results: Uint32Array;
  resultDistances: Float32Array;
  distanceComputations: number;
  micros: number;
  /** The same query answered by an exact scan of every vector, for recall and speedup. */
  exact: Uint32Array;
  exactMicros: number;
}

export interface GraphInfo {
  levels: Uint8Array;
  /** edges[layer] = flat [a0, b0, a1, b1, ...]; layer 0 is omitted (too dense to draw whole). */
  edges: Uint32Array[];
  maxLevel: number;
  buildMs: number;
  memoryBytes: number;
  threads: number;
  variant: "mt" | "st";
}

export type Query =
  | { kind: "id"; id: number }
  | {
      kind: "vector";
      vector: Float32Array;
      source: "text" | "photo" | "camera";
      text: string; // the description, or a short label for image queries
      preview?: string; // object URL of the query image
    };

export type ToWorker =
  | { type: "load"; base: string; count: number; dim: number; M: number; efConstruction: number }
  | { type: "search"; seq: number; query: Query; k: number; ef: number };

export type FromWorker =
  | { type: "progress"; stage: "download" | "build"; done: number; total: number }
  | { type: "built"; graph: GraphInfo }
  | { type: "result"; seq: number; query: Query; trace: Trace }
  | { type: "error"; seq?: number; message: string };
