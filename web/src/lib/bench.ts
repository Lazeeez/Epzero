// Headline numbers derived from the committed benchmark results, so every page quotes the same
// figures as the comparison run on a dedicated server node (bench/results/pace/gold6226/). There,
// the kernel gives every library 2 MiB huge pages, so the comparison is on equal footing.
import compare from "../../../bench/results/pace/gold6226/sift1m-compare.json";
import drift from "../../../bench/results/pace/gold6226/sift1m-guard-0.95.json";
import filterRun from "../../../bench/results/pace/gold6226/sift1m-filter.json";
import churn from "../../../bench/results/pace/gold6226/sift1m-churn.json";
import gist from "../../../bench/results/pace/gold6226/gist-960-euclidean-compete.json";
import glove from "../../../bench/results/pace/gold6226/glove-100-angular-compete.json";
import dbpedia from "../../../bench/results/pace/gold6226/dbpedia-openai-1000k-angular-compete.json";
import deep from "../../../bench/results/pace/gold6226/deep-image-96-angular-compete.json";

export { compare };

/** Where the comparison ran. */
export const MACHINE = "2 × Intel Xeon Gold 6226, clock fixed at 2.7 GHz";

type Library = keyof typeof compare.libraries;
interface Point {
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

/** Search under writes (engine/bench/bench_churn.cpp): mixed read/write load and delete/re-insert cycles. */
export { churn };

/** Filtered search (engine/bench/bench_filter.cpp): each strategy's best throughput at a recall target. */
export const FILTER_RECALL = 0.95;
export const FILTER_STRATEGIES = [
  { strategy: "exact", name: "Exact scan" },
  { strategy: "inline", name: "Inline walk" },
  { strategy: "two-hop", name: "Two-hop walk" },
] as const;

export const filterTable = [...new Set(filterRun.results.map((r) => r.selectivity))]
  .sort((a, b) => a - b)
  .map((selectivity) => {
    const rows = filterRun.results.filter((r) => r.selectivity === selectivity);
    const best = Object.fromEntries(
      FILTER_STRATEGIES.map(({ strategy }) => {
        const ok = rows.filter((r) => r.strategy === strategy && r.recall >= FILTER_RECALL);
        return [strategy, ok.length ? ok.reduce((a, b) => (b.qps > a.qps ? b : a)) : null];
      }),
    ) as Record<(typeof FILTER_STRATEGIES)[number]["strategy"], (typeof rows)[number] | null>;
    const winner = FILTER_STRATEGIES.map((s) => s.strategy).reduce((a, b) => ((best[b]?.qps ?? 0) > (best[a]?.qps ?? 0) ? b : a));
    return { selectivity, allowed: rows[0].allowed, best, winner };
  });

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
export const operating = { recall: OPERATING_RECALL, qps: qpsAtRecall(compare.libraries.epzero.sweep, OPERATING_RECALL) };

/** Epzero's QPS divided by `other`'s, at equal recall (the operating point). */
export function speedup(other: Exclude<Library, "epzero">): number {
  return operating.qps / qpsAtRecall(compare.libraries[other].sweep, operating.recall);
}

/** The competitive benchmark (bench/compete.py): Epzero and five other libraries on four ann-benchmarks datasets,
 *  each library's QPS at the same recall@10, one search thread, same graph degree and build beam. */
const FIELD_NAMES: Record<string, string> = {
  epzero: "Epzero",
  "epzero-sq8": "Epzero SQ8",
  hnswlib: "hnswlib",
  "faiss-hnsw": "FAISS HNSW",
  "faiss-ivfpq": "FAISS IVF-PQ",
  usearch: "USearch",
  vsag: "VSAG",
  glass: "Glass",
};
interface CompeteRun {
  n: number;
  dim: number;
  metric: string;
  queries: number;
  libraries: Record<string, { build_seconds: number; param: string; sweep: Point[] }>;
}
const FIELD_RUNS = [
  { key: "gist", name: "GIST1M", what: "image descriptors", recall: 0.95, run: gist },
  // On GloVe no graph library reaches 0.95 by ef = 800 at M = 16 (VSAG just does), so it's compared at 0.90.
  { key: "glove", name: "GloVe-100", what: "word embeddings", recall: 0.9, run: glove },
  { key: "dbpedia", name: "DBpedia OpenAI", what: "OpenAI text embeddings", recall: 0.95, run: dbpedia },
  { key: "deep", name: "DEEP10M", what: "image embeddings, 10M", recall: 0.95, run: deep },
] as const;

export interface FieldEntry {
  id: string;
  name: string;
  qps: number | null; // null: never reaches the recall target
  build: number;
  best: number; // highest recall measured
  epzero: boolean;
}
export const field = FIELD_RUNS.map(({ key, name, what, recall, run }) => {
  const r = run as CompeteRun;
  const entries: FieldEntry[] = Object.entries(r.libraries).map(([id, lib]) => {
    const qps = qpsAtRecall(lib.sweep, recall);
    return {
      id,
      name: FIELD_NAMES[id] ?? id,
      qps: Number.isFinite(qps) ? qps : null,
      build: lib.build_seconds,
      best: Math.max(...lib.sweep.map((p) => p.recall)),
      epzero: id.startsWith("epzero"),
    };
  });
  entries.sort((a, b) => (b.qps ?? -1) - (a.qps ?? -1));
  return { key, name, what, recall, n: r.n, dim: r.dim, metric: r.metric, queries: r.queries, entries };
});

// Round 8 (bench/results/pace/<cpu>/<dataset>-speed-round8.json): VIBE's modern embeddings, some with
// queries from another modality, epzero.build with no settings against Glass on two CPUs.
interface Round8Run {
  n: number;
  dim: number;
  libraries: Record<string, { build_seconds: number; note?: string; sweep: (Point & { ef: number })[] }>;
}
const round8Runs = import.meta.glob<Round8Run>("../../../bench/results/pace/*/*-speed-round8.json", { eager: true, import: "default" });
export const MODERN_RECALLS = [0.9, 0.95, 0.98] as const;
export const MODERN_CPUS = [
  { key: "threadripper", name: "Threadripper PRO 9985WX" },
  { key: "gold6226", name: "Xeon Gold 6226" },
] as const;
const MODERN_SETS = [
  { key: "agnews-mxbai-1024-euclidean", name: "AG News", model: "mxbai", shifted: false },
  { key: "arxiv-nomic-768-normalized", name: "arXiv", model: "Nomic", shifted: false },
  { key: "imagenet-clip-512-normalized", name: "ImageNet", model: "CLIP", shifted: false },
  { key: "laion-clip-512-normalized", name: "LAION", model: "CLIP", shifted: true },
  { key: "imagenet-align-640-normalized", name: "ImageNet", model: "ALIGN", shifted: true },
  { key: "yandex-200-cosine", name: "Yandex Text-to-Image", model: "", shifted: true },
] as const;

/** Recall at the beam width calibration chose (from the run's note), interpolated in log ef. */
function calibratedRecall(lib: Round8Run["libraries"][string] | undefined): number | null {
  const ef = Number(lib?.note?.match(/calibrated ef (\d+)/)?.[1]);
  if (!lib || !ef) return null;
  const points = [...lib.sweep].sort((a, b) => a.ef - b.ef);
  for (let i = 1; i < points.length; i++) {
    const [a, b] = [points[i - 1], points[i]];
    if (ef >= a.ef && ef <= b.ef) return a.recall + ((Math.log(ef) - Math.log(a.ef)) / (Math.log(b.ef) - Math.log(a.ef))) * (b.recall - a.recall);
  }
  return null;
}

export const modern = MODERN_SETS.map((set) => {
  const cpus = MODERN_CPUS.flatMap((cpu) => {
    const run = Object.entries(round8Runs).find(([path]) => path.includes(`/${cpu.key}/${set.key}-`))?.[1];
    if (!run) return [];
    const glass = run.libraries.glass;
    const epzero = run.libraries["epzero-auto"];
    const glassBest = Math.max(...glass.sweep.map((p) => p.recall));
    const ratios = MODERN_RECALLS.map((recall) => {
      const ratio = qpsAtRecall(epzero.sweep, recall) / qpsAtRecall(glass.sweep, recall);
      return Number.isFinite(ratio) ? ratio : null; // null: Glass never reaches this recall
    });
    return [
      {
        cpu: cpu.key,
        cpuName: cpu.name,
        ratios,
        glassBest,
        epzeroBest: Math.max(...epzero.sweep.map((p) => p.recall)),
        build: { epzero: epzero.build_seconds, glass: glass.build_seconds },
        // Recall at the calibrated beam width: calibrated on rows of the data, and (shifted sets) on sample queries.
        calibrated: calibratedRecall(epzero),
        calibratedOnQueries: calibratedRecall(run.libraries["epzero-auto-ood"]),
      },
    ];
  });
  const run = Object.entries(round8Runs).find(([path]) => path.endsWith(`/${set.key}-speed-round8.json`))?.[1];
  return { ...set, n: run?.n ?? 0, dim: run?.dim ?? 0, cpus };
});
