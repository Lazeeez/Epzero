// Messages between the page and the engine worker.

export interface LabConfig {
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

export type StageId = "generate" | "truth" | "build" | "sweep" | "baseline" | "kernel";

export type ToWorker = { type: "run"; config: LabConfig };

export type FromWorker =
  | { type: "ready"; info: EngineInfo }
  | { type: "stage"; stage: StageId; state: "running" | "done"; ms?: number; detail?: string }
  | { type: "progress"; built: number; total: number; insertsPerSec: number }
  | { type: "sweep"; row: SweepRow }
  | { type: "baseline"; qps: number }
  | { type: "kernel"; row: KernelRow }
  | { type: "summary"; buildMs: number; memoryBytes: number; layers: number }
  | { type: "done" }
  | { type: "error"; message: string };
