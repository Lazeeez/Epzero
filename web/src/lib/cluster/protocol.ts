// Messages between the cluster page and the simulator worker, and the simulator's JSON shapes
// (cluster/cmd/epzero-sim/main.go, cluster/router, cluster/sim).

export type Role = "primary" | "replica" | "none";

export interface ShardCopy {
  shard: number;
  role: Role;
  epoch: number;
  lsn: number;
  vectors: number;
  in_sync: boolean;
}

export interface SimNode {
  name: string;
  up: boolean;
  slow_ms: number;
  shards: ShardCopy[] | null;
}

export interface SimLink {
  from: string;
  to: string;
  cut: boolean;
  loss: number;
  latency_ms: number;
  sent: number;
}

export interface ShardMap {
  id: number;
  epoch: number;
  primary: string;
  replica: string;
  vectors: number;
  primary_lsn: number;
  replica_lsn: number;
  replica_in_sync: boolean;
}

export interface RouterNode {
  name: string;
  up: boolean;
  rtt_ms: number;
}

export interface ClusterEvent {
  time: string;
  kind: string;
  message: string;
}

export interface Second {
  t: number;
  searches: number;
  partial: number;
  failed: number;
  hedged: number;
  /** Mean recall@10 of complete answers against the exact top 10; -1 when there were none. */
  recall: number;
  p50: number;
  p99: number;
  writes: number;
  unreplicated: number;
  write_errors: number;
}

export interface Snapshot {
  ready: boolean;
  nodes: SimNode[] | null;
  links: SimLink[] | null;
  cluster: { nodes: RouterNode[] | null; shards: ShardMap[] | null; events: ClusterEvent[] | null };
  seconds: Second[] | null;
}

export interface CheckResult {
  operations: number;
  ambiguous: number;
  result: "ok" | "illegal" | "unknown";
  millis: number;
}

export type Fault =
  | { kind: "kill" | "restart" | "isolate"; node: string }
  | { kind: "slow"; node: string; ms: number }
  | { kind: "lossy"; node: string; loss: number }
  | { kind: "split"; side: string[] }
  | { kind: "heal" };

export interface SimOptions {
  nodes: number;
  vectors: number;
  dim: number;
  seed: number;
  latency_ms: number;
}

export type Consistency = "strong" | "eventual";

export type ToSim =
  | { type: "start"; options: SimOptions }
  | { type: "consistency"; consistency: Consistency }
  | { type: "fault"; fault: Fault }
  | { type: "rates"; searches: number; keyOps: number }
  | { type: "check" };

export type FromSim =
  | { type: "booted" }
  | { type: "started" }
  | { type: "snapshot"; snapshot: Snapshot }
  | { type: "checked"; result: CheckResult }
  | { type: "error"; message: string };
