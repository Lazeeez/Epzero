// Runs the cluster simulator (cluster/cmd/oryx-sim: the real Go router against simulated nodes,
// compiled to WebAssembly) off the page's thread, and relays its state a few times a second.
import type { FromSim, ToSim } from "./protocol";

interface OryxSim {
  start(options: string): Promise<void>;
  fault(command: string): string;
  rates(searches: number, keyOps: number): void;
  consistency(level: string): void;
  snapshot(): string;
  check(): Promise<string>;
}

declare const Go: new () => { importObject: WebAssembly.Imports; run(instance: WebAssembly.Instance): Promise<void> };

const post = (message: FromSim) => (self as unknown as Worker).postMessage(message);
const base = import.meta.env.BASE_URL;
let sim: OryxSim | null = null;

async function boot(options: Extract<ToSim, { type: "start" }>["options"]) {
  await import(/* @vite-ignore */ `${base}sim/wasm_exec.js`); // defines globalThis.Go
  const go = new Go();
  const { instance } = await WebAssembly.instantiateStreaming(fetch(`${base}sim/oryx-sim.wasm`), go.importObject);
  void go.run(instance);
  sim = (globalThis as unknown as { oryxSim: OryxSim }).oryxSim;
  post({ type: "booted" });
  await sim.start(JSON.stringify(options));
  post({ type: "started" });
  const tick = () => sim && post({ type: "snapshot", snapshot: JSON.parse(sim.snapshot()) });
  tick();
  setInterval(tick, 250);
}

self.onmessage = async (event: MessageEvent<ToSim>) => {
  const message = event.data;
  try {
    switch (message.type) {
      case "start":
        await boot(message.options);
        break;
      case "fault": {
        const error = sim?.fault(JSON.stringify(message.fault));
        if (error) post({ type: "error", message: error });
        break;
      }
      case "consistency":
        sim?.consistency(message.consistency);
        break;
      case "rates":
        sim?.rates(message.searches, message.keyOps);
        break;
      case "check":
        if (sim) post({ type: "checked", result: JSON.parse(await sim.check()) });
        break;
    }
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
