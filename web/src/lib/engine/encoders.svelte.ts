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
            this.status = "error";
            this.error = message.message;
            for (const waiter of this.pending.values()) waiter.reject(new Error(message.message));
            this.pending.clear();
          }
          break;
      }
    };
  }

  encode(image: Blob): Promise<Float32Array> {
    this.start();
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
          if (this.status !== "ready") {
            this.status = "error";
            this.error = message.message;
          }
          for (const waiter of this.pending.values()) waiter.reject(new Error(message.message));
          this.pending.clear();
          break;
      }
    };
  }

  encode(text: string): Promise<Float32Array> {
    this.start();
    const seq = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(seq, { resolve, reject });
      this.worker!.postMessage({ type: "encode", seq, text });
    });
  }
}

export const textEncoder = new TextEncoder();
