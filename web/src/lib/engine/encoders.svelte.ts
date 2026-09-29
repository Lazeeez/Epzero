// CLIP encoders that turn a description or an image into a query vector. Each runs in its own
// worker and only downloads its model on first use.
import type { FromImage } from "./image.worker";
import type { FromText } from "./text.worker";

export type EncoderStatus = "idle" | "loading" | "ready" | "error";

class ImageEncoder {
  status = $state<EncoderStatus>("idle");
  progress = $state(0);
  error = $state("");
  lastMs = $state(0);

  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (v: Float32Array) => void; reject: (e: Error) => void }>();

  start() {
    if (this.worker) return;
    this.status = "loading";
    this.worker = new Worker(new URL("./image.worker.ts", import.meta.url), { type: "module" });
    this.worker.onerror = (event) => this.fail(event.message || "The encoder worker failed to start.");
    this.worker.onmessage = (event: MessageEvent<FromImage>) => {
      const message = event.data;
      switch (message.type) {
        case "progress":
          this.progress = Math.min(message.loaded / message.total, 1);
          break;
        case "ready":
          this.status = "ready";
          this.progress = 1;
          break;
        case "embedding": {
          this.lastMs = message.ms;
          const waiter = this.pending.get(message.seq);
          this.pending.delete(message.seq);
          waiter?.resolve(message.vector);
          break;
        }
        case "error":
          if (message.seq !== undefined) {
            this.pending.get(message.seq)?.reject(new Error(message.message));
            this.pending.delete(message.seq);
          } else {
            this.fail(message.message);
          }
          break;
      }
    };
  }

  private fail(message: string) {
    this.status = "error";
    this.error = message;
    for (const waiter of this.pending.values()) waiter.reject(new Error(message));
    this.pending.clear();
  }

  encode(image: Blob): Promise<Float32Array> {
    this.start();
    if (this.status === "error") return Promise.reject(new Error(this.error));
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(seq, { resolve, reject });
      this.worker!.postMessage({ type: "encode", seq, image });
    });
  }
}

export const imageEncoder = new ImageEncoder();

class TextEncoder {
  status = $state<EncoderStatus>("idle");
  progress = $state(0);
  error = $state("");
  lastMs = $state(0);

  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (v: Float32Array) => void; reject: (e: Error) => void }>();

  /** Starts downloading the model (idempotent). */
  start() {
    if (this.worker) return;
    this.status = "loading";
    this.worker = new Worker(new URL("./text.worker.ts", import.meta.url), { type: "module" });
    this.worker.onerror = (event) => this.fail(event.message || "The encoder worker failed to start.");
    this.worker.onmessage = (event: MessageEvent<FromText>) => {
      const message = event.data;
      switch (message.type) {
        case "progress":
          if (message.total > 0) this.progress = Math.min(message.loaded / message.total, 1);
          break;
        case "ready":
          this.status = "ready";
          this.progress = 1;
          break;
        case "embedding": {
          this.lastMs = message.ms;
          const waiter = this.pending.get(message.seq);
          this.pending.delete(message.seq);
          waiter?.resolve(message.vector);
          break;
        }
        case "error":
          if (message.seq !== undefined) {
            this.pending.get(message.seq)?.reject(new Error(message.message));
            this.pending.delete(message.seq);
          } else {
            this.fail(message.message);
          }
          break;
      }
    };
  }

  private fail(message: string) {
    this.status = "error";
    this.error = message;
    for (const waiter of this.pending.values()) waiter.reject(new Error(message));
    this.pending.clear();
  }

  encode(text: string): Promise<Float32Array> {
    this.start();
    if (this.status === "error") return Promise.reject(new Error(this.error));
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(seq, { resolve, reject });
      this.worker!.postMessage({ type: "encode", seq, text });
    });
  }
}

export const textEncoder = new TextEncoder();

/**
 * Starts the CLIP encoders in the background once the page is idle, so a first text (and, on
 * capable desktops, photo or webcam) search doesn't wait for a model download. Skipped when the
 * visitor asked to save data or is on a slow connection.
 */
export function warmUpEncoders() {
  const connection = (navigator as Navigator & { connection?: { saveData?: boolean; effectiveType?: string } }).connection;
  if (connection?.saveData || /(^|-)(2g|3g)$/.test(connection?.effectiveType ?? "")) return;
  const idle = (task: () => void) =>
    "requestIdleCallback" in window ? requestIdleCallback(task, { timeout: 5000 }) : setTimeout(task, 2000);
  idle(() => {
    textEncoder.start();
    // The image model (89 MB, heavy to initialize) only warms up on capable desktops: on phones,
    // or on machines with few cores, it would compete with the 3D view and make it stutter.
    const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8;
    if (matchMedia("(pointer: coarse)").matches || (navigator.hardwareConcurrency || 4) < 8 || memory < 8) return;
    // The image model after the text one, so the two downloads don't compete.
    const waitForText = () => {
      if (textEncoder.status === "ready" || textEncoder.status === "error") idle(() => imageEncoder.start());
      else setTimeout(waitForText, 1000);
    };
    waitForText();
  });
}
