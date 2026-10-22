// Each page's title, description and heading, for the browser tab, search results and link
// previews. One list: the build writes it into every page's static HTML (vite.config.ts), so
// crawlers see it without running the app, and the app keeps the live document in step with it.

export const SITE = "https://epzero.dev/";

export interface Page {
  /** Path under the site root ("" for the home page). */
  path: string;
  /** About 60 characters at most, so search results show it whole. */
  title: string;
  /** About 155 characters at most, for the same reason. */
  description: string;
  /** The page's main heading, also the no-JavaScript fallback's. */
  heading: string;
}

export const PAGES = {
  home: {
    path: "",
    title: "Epzero · A vector database, running in your browser",
    description:
      "Epzero is a vector database for fast similarity search, built from scratch in C++20 on HNSW. Try semantic photo search and live benchmarks in your browser.",
    heading: "A vector database, built from scratch.",
  },
  try: {
    path: "try",
    title: "Semantic photo search in your browser · Epzero",
    description:
      "Search 29,825 photos, or your own privately, by words, image or webcam with a vector database running in your browser, and step through every HNSW hop.",
    heading: "Search 29,825 photos, locally.",
  },
  play: {
    path: "play",
    title: "Can you out-navigate HNSW? · Epzero",
    description:
      "A game: hop from photo to photo along the real HNSW graph to reach a target, then see the route Epzero's search took through its express layers.",
    heading: "Can you out-navigate Epzero?",
  },
  cluster: {
    path: "cluster",
    title: "Break a vector database cluster · Epzero",
    description:
      "Kill, isolate or slow the nodes of a replicated vector database in your browser and watch searches keep their exact answers, audited for linearizability.",
    heading: "Break the cluster. Watch it keep its answers.",
  },
  benchmarks: {
    path: "benchmarks",
    title: "Vector search benchmarks: Epzero vs FAISS, hnswlib, Glass",
    description:
      "Recall vs queries per second for Epzero, hnswlib, FAISS, USearch, VSAG and Glass on SIFT1M, GIST1M, GloVe, OpenAI embeddings and DEEP10M, plus a lab.",
    heading: "Faster than hnswlib and FAISS. Measured against the field.",
  },
  blog: {
    path: "blog",
    title: "Vector search is a memory problem · Epzero",
    description:
      "How an HNSW vector index got 31% faster from memory work, which obvious optimizations did nothing, and where it stands against FAISS, hnswlib and Glass.",
    heading: "Vector search is a memory problem",
  },
  notfound: {
    path: "",
    title: "Page not found · Epzero",
    description: "This page doesn't exist. Epzero is an HNSW vector database you can run and search in your browser.",
    heading: "This page drifted out of the graph.",
  },
} satisfies Record<string, Page>;

export type PageName = keyof typeof PAGES;
