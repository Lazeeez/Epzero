// Headline numbers derived from the committed benchmark results, so every page quotes the same
// figures as bench/results/sift1m-compare.json.
import compare from "../../../bench/results/sift1m-compare.json";

export { compare };

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

/** The operating point quoted everywhere: Oryx at ef = 48 (recall ≈ 0.943). */
export const OPERATING_EF = 48;
export const operating = compare.libraries.oryx.sweep.find((p) => p.ef === OPERATING_EF)!;

/** Oryx's QPS divided by `other`'s, at equal recall (Oryx's recall at the operating point). */
export function speedup(other: Exclude<Library, "oryx">): number {
  return operating.qps / qpsAtRecall(compare.libraries[other].sweep, operating.recall);
}
