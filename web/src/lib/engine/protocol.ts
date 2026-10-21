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
  /** With a filter: how the engine honored it, and how many photos it allowed. */
  strategy?: FilterStrategy;
  allowed?: number;
}

/** How a filtered search ran: scan the allowed photos, or walk the graph (keeping only allowed
 * results, or stepping past excluded neighbours to theirs). */
export type FilterStrategy = "exact" | "inline" | "two-hop";

/** Why a photo is or isn't among a search's results (see HnswIndex::explain). */
export interface Explanation {
  cause: "returned" | "not-in-index" | "filtered-out" | "not-near-enough" | "search-missed";
  /** 1-based rank by exact distance among the photos the search could return. */
  exactRank: number;
  distance: number;
  kthDistance: number;
  /** search-missed: the walk scored it but kept closer ones (otherwise it never reached it). */
  scored: boolean;
  /** search-missed: the smallest beam width that returns it (0: none up to 4096). */
  efNeeded: number;
}

export interface FilterInfo {
  allowed: number;
  selectivity: number;
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
  M: number;
  efConstruction: number;
  /** Loaded from the prebuilt graph file (buildMs is then the load time) rather than built here. */
  prebuilt: boolean;
}

export type Query =
  | { kind: "id"; id: number }
  | {
      kind: "vector";
      vector: Float32Array;
      source: "text" | "photo" | "camera" | "blend";
      text: string; // the description, or a short label for image queries
      preview?: string; // object URL of the query image
      blend?: { id?: number; words: string; mix: number }; // a photo + words mix
    };

export interface Evaluation {
  recall: number;
  qps: number;
  distancesPerQuery: number;
  ef: number;
}

/** Small request/reply calls answered from the loaded vectors and graph. */
export type Request =
  | { method: "vector"; id: number } // -> Float32Array (normalized)
  | { method: "neighbors"; id: number; target: number } // -> Neighborhood on the base layer
  | { method: "similarity"; a: number; b: number } // -> number (cosine)
  | { method: "evaluate"; ef: number; samples: number } // -> Evaluation
  | { method: "filter"; ids: Uint32Array | null } // -> FilterInfo; restricts later searches to these photos
  | { method: "explain"; query: Query; label: number; k: number; ef: number }; // -> Explanation

export interface Neighborhood {
  ids: Uint32Array;
  /** Cosine similarity of each neighbour to the target. */
  similarities: Float32Array;
}

/**
 * Where each stage of the progressive vector file (pipeline/progressive_vectors.py) ends, in bytes:
 * the file holds blocks of dimensions, each block's scales (float32), rotation rows (float16) and
 * the photos' codes (int8), so stage s is searchable once ends[s] bytes have arrived.
 */
export function stageEnds(count: number, dim: number, stages: number[]): number[] {
  let end = 0;
  let lo = 0;
  return stages.map((dims) => {
    const w = dims - lo;
    end += 4 * w + 2 * w * dim + count * w;
    lo = dims;
    return end;
  });
}

export type ToWorker =
  | {
      type: "load";
      base: string;
      /** The progressive vector file, and the dimension counts it can be searched at. */
      vectors: string;
      stages: number[];
      count: number;
      dim: number;
      M: number;
      efConstruction: number;
      /** A prebuilt graph to load instead of building (falls back to building if it fails). */
      graph?: string;
    }
  | { type: "rebuild"; M: number; efConstruction: number }
  | { type: "search"; seq: number; query: Query; k: number; ef: number }
  | { type: "request"; seq: number; request: Request };

export type FromWorker =
  | { type: "progress"; stage: "download" | "build"; done: number; total: number }
  | { type: "built"; graph: GraphInfo }
  /** More of the vectors arrived: searches now use this many dimensions (of `dim`). */
  | { type: "precision"; dims: number }
  | { type: "result"; seq: number; query: Query; trace: Trace }
  | { type: "reply"; seq: number; value: unknown }
  | { type: "error"; seq?: number; message: string };
