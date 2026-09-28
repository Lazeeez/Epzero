// Derived views of a search trace: the beam's state after every event (for the step-through
// debugger) and the set of nodes a search touched (for the brute-force race).
import type { Trace } from "./protocol";

export interface EventInfo {
  layer: number;
  from: number;
  to: number;
  distance: number;
  accepted: boolean;
  /** Layer 0: candidates kept so far (at most ef). Upper layers: always 1 (greedy). */
  beamSize: number;
  /** The distance a newcomer had to beat after this event (beam's worst, or greedy's current best). */
  bound: number;
  /** Where this event sits in the search, for a readable description. */
  phase: "descent" | "beam";
}

/** Replays the search's decisions to recover the beam after each event. */
export function analyze(trace: Trace, ef: number): EventInfo[] {
  const n = trace.to.length;
  const out: EventInfo[] = new Array(n);
  // Max-heap of the kept distances on layer 0 (the beam), mirroring the engine's `best` queue.
  const heap: number[] = [];
  const push = (d: number) => {
    heap.push(d);
    for (let i = heap.length - 1; i > 0; ) {
      const parent = (i - 1) >> 1;
      if (heap[parent] >= heap[i]) break;
      [heap[parent], heap[i]] = [heap[i], heap[parent]];
      i = parent;
    }
  };
  const pop = () => {
    const last = heap.pop()!;
    if (heap.length === 0) return;
    heap[0] = last;
    for (let i = 0; ; ) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < heap.length && heap[l] > heap[m]) m = l;
      if (r < heap.length && heap[r] > heap[m]) m = r;
      if (m === i) break;
      [heap[m], heap[i]] = [heap[i], heap[m]];
      i = m;
    }
  };

  let greedyBest = Infinity;
  let beamStarted = false;
  for (let i = 0; i < n; i++) {
    const layer = trace.layer[i];
    const accepted = trace.accepted[i] === 1;
    const distance = trace.distance[i];
    if (layer > 0) {
      if (accepted) greedyBest = Math.min(greedyBest, distance);
      out[i] = { layer, from: trace.from[i], to: trace.to[i], distance, accepted, beamSize: 1, bound: greedyBest, phase: "descent" };
      continue;
    }
    if (!beamStarted) {
      beamStarted = true;
      push(greedyBest); // the entry point on layer 0 seeds the beam
    }
    if (accepted) {
      push(distance);
      if (heap.length > ef) pop();
    }
    out[i] = {
      layer,
      from: trace.from[i],
      to: trace.to[i],
      distance,
      accepted,
      beamSize: heap.length,
      bound: heap.length ? heap[0] : Infinity,
      phase: "beam",
    };
  }
  return out;
}

/** Every distinct node the search scored, in the order it first did so. */
export function touched(trace: Trace): number[] {
  const seen = new Set<number>([trace.entryLabel]);
  const order = [trace.entryLabel];
  for (const node of trace.to) {
    if (!seen.has(node)) {
      seen.add(node);
      order.push(node);
    }
  }
  return order;
}

/** How many of the HNSW results the exact scan also returned. */
export function agreement(trace: Trace): number {
  const exact = new Set(trace.exact);
  let hits = 0;
  for (const id of trace.results) if (exact.has(id)) hits++;
  return hits;
}
