// Messages between the page and the engine worker.

export type Metric = "cosine" | "l2" | "ip";

export interface LabConfig {
  /** synthetic: clustered vectors generated in WebAssembly; own: the visitor's file (inspected first). */
  source: "synthetic" | "own";
  /** own: the distance, and how many vectors to hold out as queries when no query file is given. */
  metric?: Metric;
  holdout?: number;
  n: number;
  dim: number;
  queries: number;
  clusters: number;
  M: number;
  efConstruction: number;
  k: number;
  efs: number[];
}

export interface EngineInfo {
  simd: string;
  threads: number;
  variant: "mt" | "st";
  isolated: boolean;
}

export interface SweepRow {
  ef: number;
  recall: number;
  qps: number;
  p50Us: number;
  p99Us: number;
  distancesPerQuery: number;
}

export interface KernelRow {
  dim: number;
  scalarNs: number;
  simdNs: number;
}

/** A vector file the worker has read (and keeps until it's replaced or run). */
export interface VectorInfo {
  name: string;
  bytes: number;
  n: number;
  dim: number;
  format: string;
  meanNorm: number;
  /** Every vector is (nearly) unit length: cosine is the natural metric. */
  unit: boolean;
}

export type StageId = "generate" | "load" | "truth" | "build" | "sweep" | "baseline" | "kernel";

export type ToWorker =
  | { type: "run"; config: LabConfig }
  | { type: "inspect"; role: "base" | "queries"; file: File }
  | { type: "drop"; role: "base" | "queries" };

export type FromWorker =
  | { type: "ready"; info: EngineInfo }
  | { type: "stage"; stage: StageId; state: "running" | "done"; ms?: number; detail?: string }
  | { type: "progress"; built: number; total: number; insertsPerSec: number }
  | { type: "sweep"; row: SweepRow }
  | { type: "baseline"; qps: number }
  | { type: "kernel"; row: KernelRow }
  | { type: "summary"; buildMs: number; memoryBytes: number; layers: number }
  | { type: "inspected"; role: "base" | "queries"; info: VectorInfo }
  | { type: "inspect-error"; role: "base" | "queries"; message: string }
  /** own: what the run indexes (after holding out queries). */
  | { type: "loaded"; n: number; queries: number; dim: number }
  | { type: "done" }
  | { type: "error"; message: string };
