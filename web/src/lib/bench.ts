// Headline numbers derived from the committed benchmark results, so every page quotes the same
// figures as the comparison run on a dedicated server node (bench/results/pace/gold6226/). There,
// the kernel gives every library 2 MiB huge pages, so the comparison is on equal footing.
import compare from "../../../bench/results/pace/gold6226/sift1m-compare.json";
import drift from "../../../bench/results/sift1m-guard-laptop.json";

export { compare };

/** Where the comparison ran. */
export const MACHINE = "2 × Intel Xeon Gold 6226, clock fixed at 2.7 GHz";

type Library = keyof typeof compare.libraries;
interface Point {
  ef: number;
  recall: number;
  qps: number;
}

/** A library's QPS at exactly `recall`, interpolated log-linearly between its measured points. */
function qpsAtRecall(sweep: Point[], recall: number): number {
  const points = [...sweep].sort((a, b) => a.recall - b.recall);
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1];
    const b = points[i];
    if (recall >= a.recall && recall <= b.recall) {
      const t = (recall - a.recall) / (b.recall - a.recall);
      return Math.exp(Math.log(a.qps) + t * (Math.log(b.qps) - Math.log(a.qps)));
    }
  }
  return NaN;
}

/** Recall under drift (engine/bench/bench_guard.cpp): three policies through four kinds of drift. */
export { drift };

export const DRIFT_POLICIES = [
  { policy: "guard", name: "Recall guard", color: "var(--series-1)" },
  { policy: "fixed-tuned", name: "Fixed, tuned once", color: "var(--series-2)" },
  { policy: "fixed-worst", name: "Fixed, worst case", color: "var(--series-3)" },
];

/** Per policy: phases at the target once adapted (each phase's last pass), and mean cost. */
export const driftSummary = DRIFT_POLICIES.map(({ policy, name }) => {
  const rows = drift.batches.filter((b) => b.policy === policy);
  const last = rows.filter((b) => b.pass === drift.passes - 1);
  return {
    policy,
    name,
    adapted: last.filter((b) => b.recall >= drift.target).length,
    phases: last.length,
    cost: rows.reduce((sum, b) => sum + b.distances, 0) / rows.length,
  };
});

/** How far the guard's own estimates were from the true recall, and how often its interval held it. */
export const driftEstimates = (() => {
  const checked = drift.batches.filter((b) => b.policy === "guard" && "lower" in b && (b as { lower: number }).lower > 0) as {
    recall: number;
    estimate: number;
    lower: number;
    upper: number;
  }[];
  return {
    error: checked.reduce((sum, b) => sum + Math.abs(b.estimate - b.recall), 0) / checked.length,
    covered: checked.filter((b) => b.lower <= b.recall && b.recall <= b.upper).length,
    total: checked.length,
  };
})();

/** The operating point quoted everywhere: recall@10 = 0.95, each library interpolated to it. */
export const OPERATING_RECALL = 0.95;
export const operating = { recall: OPERATING_RECALL, qps: qpsAtRecall(compare.libraries.oryx.sweep, OPERATING_RECALL) };

/** Oryx's QPS divided by `other`'s, at equal recall (the operating point). */
export function speedup(other: Exclude<Library, "oryx">): number {
  return operating.qps / qpsAtRecall(compare.libraries[other].sweep, operating.recall);
}
