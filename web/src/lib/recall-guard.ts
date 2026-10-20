// The playground's recall guard: the policy of the engine's RecallGuard (src/recall.cpp), fed by
// the exact check every search here already runs, so every query is a sample rather than 1%.
//   widen  as soon as measured recall falls below the target (x1.5)
//   narrow only while the confidence interval's lower end clears the target by a margin (x7/8)
//   hold   otherwise; after any change, measure afresh

export interface Estimate {
  samples: number;
  recall: number;
  lower: number;
  upper: number;
}

/** Wilson score interval for a mean of per-query recalls (one observation per query). */
export function wilson(values: readonly number[], z = 1.96): Estimate {
  const n = values.length;
  if (n === 0) return { samples: 0, recall: 1, lower: 0, upper: 1 };
  const p = values.reduce((a, b) => a + b, 0) / n;
  const z2 = z * z;
  const center = (p + z2 / (2 * n)) / (1 + z2 / n);
  const half = (z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / (1 + z2 / n);
  return { samples: n, recall: p, lower: Math.max(0, center - half), upper: Math.min(1, center + half) };
}

export interface GuardOptions {
  target: number;
  minSamples: number;
  margin: number;
  minEf: number;
  maxEf: number;
}

export interface Decision {
  ef: number;
  reason: string;
}

/** The next beam width for a window of per-query recalls at `ef`, or null to hold. */
export function decide(window: readonly number[], ef: number, o: GuardOptions): Decision | null {
  if (window.length < o.minSamples) return null;
  const e = wilson(window);
  const step = (x: number) => Math.round(x / 16) * 16;
  if (e.recall < o.target) {
    const next = Math.min(o.maxEf, Math.max(ef + 16, step(ef * 1.5)));
    if (next === ef) return null;
    return { ef: next, reason: `recall ${e.recall.toFixed(3)} is under ${o.target}: widened` };
  }
  if (e.lower >= o.target + o.margin) {
    const next = Math.max(o.minEf, step((ef * 7) / 8));
    if (next === ef) return null;
    return { ef: next, reason: `recall is at least ${e.lower.toFixed(3)} with 95% confidence: narrowed to save work` };
  }
  return null;
}
