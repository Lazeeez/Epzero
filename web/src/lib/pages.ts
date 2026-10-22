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
      "Epzero is an HNSW vector database built from scratch in C++20. It runs here as WebAssembly, searching 29,825 photos on your own CPU.",
    heading: "A vector database, built from scratch.",
  },
  try: {
    path: "try",
    title: "Search 29,825 photos in your browser · Epzero",
    description:
      "Search photos by words, image or webcam with an HNSW index running in your browser, then step through every hop the graph search takes.",
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
    title: "Benchmarks: Epzero vs hnswlib, FAISS and Glass",
    description:
      "Epzero against hnswlib, FAISS, USearch, VSAG and Glass on SIFT1M, GIST1M, GloVe, OpenAI embeddings and DEEP10M, with recall under drift and filters.",
    heading: "Faster than hnswlib and FAISS. Measured against the field.",
  },
  notfound: {
    path: "",
    title: "Page not found · Epzero",
    description: "This page doesn't exist. Epzero is an HNSW vector database you can run and search in your browser.",
    heading: "This page drifted out of the graph.",
  },
} satisfies Record<string, Page>;

export type PageName = keyof typeof PAGES;
