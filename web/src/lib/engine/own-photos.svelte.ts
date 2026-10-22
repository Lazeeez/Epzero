// The visitor's own photos: embedded with CLIP in this tab (a small pool of encoder workers), laid
// out on a map (UMAP, in a worker) and handed to the engine, which indexes them in place of the demo
// set. Nothing leaves the device. A collection can be kept in this browser (IndexedDB) for next time.
import type { Embedded, FromImage } from "./image.worker";
import type { FromLayout } from "./layout.worker";

export interface OwnPhoto {
  /** The file name, for captions and alt text. */
  name: string;
  /** Object URL of a small JPEG thumbnail. */
  thumb: string;
  width: number;
  height: number;
  /** The original, while this page is open (a collection loaded from storage keeps only thumbnails). */
  file?: File;
}

export interface OwnCollection {
  photos: OwnPhoto[];
  /** Unit CLIP vectors, photos.length x dim. */
  vectors: Float32Array;
  dim: number;
  /** Map positions in [-1, 1]^2, two per photo. */
  layout: Float32Array;
}

export type IndexPhase = "idle" | "model" | "embedding" | "layout" | "saving" | "done" | "error";

const IMAGE = /\.(jpe?g|png|webp|avif|gif|bmp|heic|heif)$/i;
const THUMB = 360; // px, longest side: sharp in the results grid and the cards
const BATCH = 4;
const coarse = () => matchMedia("(pointer: coarse)").matches;
const deviceMemory = () => (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8;

/** Most photos one collection may hold on this device. */
export function photoLimit(): number {
  return coarse() || deviceMemory() < 4 ? 1000 : 3000;
}

/** Encoder workers to run side by side: each holds its own copy of the model (~200 MB). */
export function poolSize(): number {
  if (coarse() || deviceMemory() < 4) return 1;
  return Math.max(1, Math.min(deviceMemory() >= 8 ? 4 : 2, Math.floor((navigator.hardwareConcurrency || 4) / 2)));
}

/** The image files among `files` (folders bring other files along), without exact duplicates. */
export function imageFiles(files: Iterable<File>): File[] {
  const seen = new Set<string>();
  const out: File[] = [];
  for (const file of files) {
    if (!(file.type.startsWith("image/") || IMAGE.test(file.name))) continue;
    const key = `${file.name}|${file.size}|${file.lastModified}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(file);
  }
  return out;
}

/** "IMG_2041.jpeg" -> "IMG_2041" */
export function photoName(file: string): string {
  return file.replace(/\.[^.]+$/, "");
}

class PhotoIndexer {
  phase = $state<IndexPhase>("idle");
  total = $state(0);
  done = $state(0);
  /** Files the browser couldn't decode (HEIC outside Safari, corrupt files). */
  skipped = $state(0);
  /** Photos per second, smoothed. */
  rate = $state(0);
  /** Seconds left, estimated from the rate. */
  eta = $state(0);
  modelProgress = $state(0);
  layoutProgress = $state(0);
  workers = $state(0);
  /** The latest thumbnails, newest last, for the progress mosaic. */
  recent = $state.raw<string[]>([]);
  error = $state("");

  private pool: Worker[] = [];
  private layoutWorker: Worker | null = null;
  private cancelled = false;

  get busy(): boolean {
    return this.phase === "model" || this.phase === "embedding" || this.phase === "layout" || this.phase === "saving";
  }

  /** Embeds `files` (after `existing`'s photos, if given) and lays the whole collection out.
   *  Resolves null when cancelled. */
  async run(files: File[], existing: OwnCollection | null = null): Promise<OwnCollection | null> {
    this.cancelled = false;
    this.error = "";
    this.total = files.length;
    this.done = this.skipped = 0;
    this.rate = this.eta = 0;
    this.modelProgress = this.layoutProgress = 0;
    this.recent = [];
    this.phase = "model";
    try {
      const embedded = await this.embed(files);
      if (!embedded) return null;
      const before = existing?.photos.length ?? 0;
      const dim = existing?.dim ?? embedded.dim;
      const photos = [...(existing?.photos ?? []), ...embedded.photos];
      const vectors = new Float32Array(photos.length * dim);
      if (existing) vectors.set(existing.vectors);
      vectors.set(embedded.vectors, before * dim);
      if (photos.length === 0) throw new Error("None of these files could be read as photos in this browser.");
      this.phase = "layout";
      const layout = await this.layOut(vectors, photos.length, dim);
      if (!layout) return null;
      this.phase = "done";
      return { photos, vectors, dim, layout };
    } catch (error) {
      if (this.cancelled) return null;
      this.error = error instanceof Error ? error.message : String(error);
      this.phase = "error";
      return null;
    } finally {
      this.stop();
    }
  }

  cancel() {
    this.cancelled = true;
    this.stop();
    for (const url of this.recent) URL.revokeObjectURL(url);
    this.recent = [];
    this.phase = "idle";
  }

  reset() {
    if (!this.busy) this.phase = "idle";
  }

  private stop() {
    for (const worker of this.pool) worker.terminate();
    this.pool = [];
    this.layoutWorker?.terminate();
    this.layoutWorker = null;
  }

  private async embed(files: File[]): Promise<{ photos: OwnPhoto[]; vectors: Float32Array; dim: number } | null> {
    const size = Math.min(poolSize(), Math.max(1, Math.ceil(files.length / BATCH)));
    this.workers = size;
    const results: (Embedded & { url?: string })[] = new Array(files.length);
    const queue: number[][] = [];
    for (let i = 0; i < files.length; i += BATCH) queue.push(Array.from({ length: Math.min(BATCH, files.length - i) }, (_, j) => i + j));

    let seq = 0;
    const samples: [number, number][] = []; // (time, photos done), for the rate
    const work = (worker: Worker, first: boolean) =>
      new Promise<void>((resolve, reject) => {
        let batch: number[] = [];
        const next = () => {
          if (this.cancelled) return resolve();
          const taken = queue.shift();
          if (!taken) return resolve();
          batch = taken;
          worker.postMessage({ type: "embed", seq: ++seq, files: batch.map((i) => files[i]), thumb: THUMB });
        };
        worker.onmessage = (event: MessageEvent<FromImage>) => {
          const message = event.data;
          if (message.type === "progress" && first) this.modelProgress = Math.min(message.loaded / message.total, 1);
          else if (message.type === "error" && message.seq === undefined) reject(new Error(message.message));
          else if (message.type === "error") {
            // A batch the model choked on: count its photos as unreadable and go on.
            this.skipped += batch.length;
            this.done += batch.length;
            next();
          } else if (message.type === "embedded") {
            if (this.phase === "model") this.phase = "embedding";
            const fresh: string[] = [];
            message.items.forEach((item, k) => {
              const url = item.vector && item.thumb ? URL.createObjectURL(item.thumb) : undefined;
              results[batch[k]] = { ...item, url };
              if (url) fresh.push(url);
              else this.skipped++;
            });
            this.done += message.items.length;
            this.recent = [...this.recent, ...fresh].slice(-36);
            const now = performance.now();
            samples.push([now, this.done]);
            while (samples.length > 2 && now - samples[0][0] > 6000) samples.shift();
            if (samples.length > 1) {
              const [t0, d0] = samples[0];
              this.rate = ((this.done - d0) / (now - t0)) * 1000;
              this.eta = this.rate > 0 ? (this.total - this.done) / this.rate : 0;
            }
            next();
          }
        };
        worker.onerror = (event) => reject(new Error(event.message || "The image encoder failed to start."));
        next();
      });

    // The first worker loads the model (from cache after the first visit); the others start once
    // it's in, so they read it from cache rather than downloading it again side by side.
    const first = new Worker(new URL("./image.worker.ts", import.meta.url), { type: "module" });
    this.pool.push(first);
    const firstDone = work(first, true);
    while (this.phase === "model" && !this.cancelled && queue.length) await new Promise((r) => setTimeout(r, 100));
    const rest = Array.from({ length: size - 1 }, () => {
      const worker = new Worker(new URL("./image.worker.ts", import.meta.url), { type: "module" });
      this.pool.push(worker);
      return work(worker, false);
    });
    await Promise.all([firstDone, ...rest]);
    if (this.cancelled) {
      for (const r of results) if (r?.url) URL.revokeObjectURL(r.url);
      return null;
    }

    const photos: OwnPhoto[] = [];
    let dim = 0;
    const kept: Float32Array[] = [];
    results.forEach((r, i) => {
      if (!r?.vector || !r.url) return;
      dim = r.vector.length;
      kept.push(r.vector);
      photos.push({ name: photoName(files[i].name), thumb: r.url, width: r.width, height: r.height, file: files[i] });
    });
    const vectors = new Float32Array(kept.length * dim);
    kept.forEach((v, i) => vectors.set(v, i * dim));
    return { photos, vectors, dim: dim || 512 };
  }

  private layOut(vectors: Float32Array, count: number, dim: number): Promise<Float32Array | null> {
    return new Promise((resolve, reject) => {
      const worker = new Worker(new URL("./layout.worker.ts", import.meta.url), { type: "module" });
      this.layoutWorker = worker;
      worker.onmessage = (event: MessageEvent<FromLayout>) => {
        const message = event.data;
        if (message.type === "progress") this.layoutProgress = message.done;
        else if (message.type === "layout") resolve(message.xy);
        else reject(new Error(message.message));
      };
      worker.onerror = (event) => reject(new Error(event.message || "The layout worker failed to start."));
      // A copy: the engine still needs the vectors.
      worker.postMessage({ vectors: vectors.slice(), count, dim });
    }).then((xy) => (this.cancelled ? null : (xy as Float32Array)));
  }
}

export const photoIndexer = new PhotoIndexer();

// ---- Keeping a collection in this browser (IndexedDB) ----

const DB = "epzero-photos";
const STORE = "collections";
const KEY = "mine";

interface Stored {
  version: 1;
  dim: number;
  vectors: ArrayBuffer;
  layout: ArrayBuffer;
  names: string[];
  sizes: number[];
  thumbs: Blob[];
  savedAt: number;
}

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function transact<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await open();
  try {
    return await new Promise<T>((resolve, reject) => {
      const request = run(db.transaction(STORE, mode).objectStore(STORE));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  } finally {
    db.close();
  }
}

/** Keeps `collection` in this browser, replacing any kept before. */
export async function keepCollection(collection: OwnCollection): Promise<void> {
  const thumbs = await Promise.all(collection.photos.map((p) => fetch(p.thumb).then((r) => r.blob())));
  const stored: Stored = {
    version: 1,
    dim: collection.dim,
    vectors: collection.vectors.slice().buffer,
    layout: collection.layout.slice().buffer,
    names: collection.photos.map((p) => p.name),
    sizes: collection.photos.flatMap((p) => [p.width, p.height]),
    thumbs,
    savedAt: Date.now(),
  };
  await transact("readwrite", (store) => store.put(stored, KEY));
}

/** The kept collection's size and date, without loading it (null: none). */
export async function keptInfo(): Promise<{ count: number; savedAt: number } | null> {
  try {
    const keys = await transact("readonly", (store) => store.getAllKeys());
    if (!keys.includes(KEY)) return null;
    const stored = await transact<Stored | undefined>("readonly", (store) => store.get(KEY));
    return stored ? { count: stored.names.length, savedAt: stored.savedAt } : null;
  } catch {
    return null; // private mode, or storage blocked: nothing kept
  }
}

export async function loadKept(): Promise<OwnCollection | null> {
  const stored = await transact<Stored | undefined>("readonly", (store) => store.get(KEY));
  if (!stored || stored.version !== 1) return null;
  return {
    dim: stored.dim,
    vectors: new Float32Array(stored.vectors),
    layout: new Float32Array(stored.layout),
    photos: stored.names.map((name, i) => ({
      name,
      thumb: URL.createObjectURL(stored.thumbs[i]),
      width: stored.sizes[i * 2],
      height: stored.sizes[i * 2 + 1],
    })),
  };
}

export async function forgetKept(): Promise<void> {
  try {
    await transact("readwrite", (store) => store.delete(KEY));
  } catch {
    /* nothing kept */
  }
}

/** Frees a collection's thumbnails (object URLs). */
export function release(collection: OwnCollection | null) {
  for (const photo of collection?.photos ?? []) URL.revokeObjectURL(photo.thumb);
}
