// The cluster page's simulator: starts the worker on first use and keeps its latest state.
import type { CheckResult, Consistency, Fault, FromSim, SimOptions, Snapshot, ToSim } from "./protocol";

export const SIM_OPTIONS: SimOptions = { nodes: 4, vectors: 12_000, dim: 32, seed: 7, latency_ms: 2 };

class ClusterSim {
  status = $state<"idle" | "loading" | "starting" | "running" | "error">("idle");
  error = $state("");
  snapshot = $state.raw<Snapshot | null>(null);
  check = $state.raw<CheckResult | null>(null);
  checking = $state(false);
  /** Searches and key reads/writes per second. */
  rates = $state({ searches: 40, keyOps: 20 });
  consistency = $state<Consistency>("eventual");

  private worker: Worker | null = null;

  start() {
    if (this.worker) return;
    this.status = "loading";
    this.worker = new Worker(new URL("./sim.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent<FromSim>) => {
      const message = event.data;
      switch (message.type) {
        case "booted":
          this.status = "starting";
          break;
        case "started":
          this.status = "running";
          break;
        case "snapshot":
          this.snapshot = message.snapshot;
          break;
        case "checked":
          this.check = message.result;
          this.checking = false;
          break;
        case "error":
          this.checking = false;
          if (this.status !== "running") this.status = "error";
          this.error = message.message;
          break;
      }
    };
    this.worker.onerror = (event) => {
      this.status = "error";
      this.error = event.message || "The simulator failed to start.";
    };
    this.send({ type: "start", options: SIM_OPTIONS });
  }

  /** Stops the simulation (leaving the page): nothing keeps running in the background. */
  stop() {
    this.worker?.terminate();
    this.worker = null;
    this.status = "idle";
    this.snapshot = null;
    this.check = null;
    this.checking = false;
    this.consistency = "eventual";
  }

  fault(fault: Fault) {
    this.send({ type: "fault", fault });
  }

  setRates(searches: number, keyOps: number) {
    this.rates = { searches, keyOps };
    this.send({ type: "rates", searches, keyOps });
  }

  setConsistency(level: Consistency) {
    this.consistency = level;
    this.check = null; // the recorded history restarts with the new level
    this.send({ type: "consistency", consistency: level });
  }

  verify() {
    if (this.checking) return;
    this.checking = true;
    this.check = null;
    this.send({ type: "check" });
  }

  private send(message: ToSim) {
    this.worker?.postMessage(message);
  }
}

export const clusterSim = new ClusterSim();
