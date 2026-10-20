// Headline numbers derived from the committed benchmark results, so every page quotes the same
// figures as the comparison run on a dedicated server node (bench/results/pace/gold6226/). There,
// the kernel gives every library 2 MiB huge pages, so the comparison is on equal footing.
import compare from "../../../bench/results/pace/gold6226/sift1m-compare.json";

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

/** The operating point quoted everywhere: recall@10 = 0.95, each library interpolated to it. */
export const OPERATING_RECALL = 0.95;
export const operating = { recall: OPERATING_RECALL, qps: qpsAtRecall(compare.libraries.oryx.sweep, OPERATING_RECALL) };

/** Oryx's QPS divided by `other`'s, at equal recall (the operating point). */
export function speedup(other: Exclude<Library, "oryx">): number {
  return operating.qps / qpsAtRecall(compare.libraries[other].sweep, operating.recall);
}
