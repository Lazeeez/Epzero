import "../shared/base.css";
import "./lab.css";
import { initTheme } from "../shared/theme";
import { formatQps, formatUs, RecallChart } from "./chart";
import type { EngineInfo, FromWorker, KernelRow, LabConfig, StageId, SweepRow } from "./protocol";

const $ = <T extends Element>(selector: string) => document.querySelector<T>(selector)!;

initTheme($<HTMLButtonElement>("button.theme-toggle"));

// ---------- State ----------
const form = $<HTMLFormElement>("#config");
const runButton = $<HTMLButtonElement>("button.run");
const chart = new RecallChart($("#chart"));
let engine: EngineInfo | null = null;
let config: LabConfig | null = null;
let rows: SweepRow[] = [];
let baselineQps: number | null = null;
let build: { ms: number; memoryBytes: number; layers: number } | null = null;
const kernels: KernelRow[] = [];

const worker = new Worker(new URL("./lab.worker.ts", import.meta.url), { type: "module" });
worker.onmessage = (event: MessageEvent<FromWorker>) => handle(event.data);
worker.onerror = (event) => handle({ type: "error", message: event.message || "The engine worker failed to start." });

function readConfig(): LabConfig {
  const data = new FormData(form);
  const num = (key: string) => Number(data.get(key));
  return {
    n: num("n"),
    dim: num("dim"),
    M: num("M"),
    efConstruction: num("efConstruction"),
    queries: 1000,
    clusters: 100,
    k: 10,
    efs: [10, 16, 24, 32, 48, 64, 96, 128, 192, 256],
  };
}

function setRunning(running: boolean) {
  runButton.disabled = running || !engine;
  runButton.querySelector(".run-label")!.textContent = running ? "Running…" : "Run benchmark";
  form.querySelectorAll("input").forEach((input) => (input.disabled = running));
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!engine) return;
  config = readConfig();
  rows = [];
  baselineQps = null;
  build = null;
  kernels.length = 0;
  $<HTMLElement>(".pipeline").hidden = false;
  document.querySelectorAll<HTMLElement>(".stages li").forEach((li) => {
    delete li.dataset.state;
    li.querySelector(".detail")!.textContent = "";
    li.querySelector(".ms")!.textContent = "";
  });
  setBuildProgress(0);
  for (const id of ["#tiles", "#chart-card", "#kernel-card"]) $<HTMLElement>(id).hidden = true;
  $("#sweep-table tbody").replaceChildren();
  $("#kernels").replaceChildren();
  $("#pipeline-note").textContent =
    `${config.n.toLocaleString()} × ${config.dim}-d vectors · ${config.queries.toLocaleString()} queries · ` +
    `M=${config.M}, efConstruction=${config.efConstruction}`;
  setRunning(true);
  worker.postMessage({ type: "run", config });
});

// ---------- Worker messages ----------
function handle(message: FromWorker) {
  switch (message.type) {
    case "ready":
      engine = message.info;
      renderEnvironment(message.info);
      setRunning(false);
      break;
    case "stage":
      setStage(message.stage, message.state, message.ms);
      break;
    case "progress": {
      setBuildProgress(message.built / message.total);
      setDetail(
        "build",
        `${message.built.toLocaleString()} / ${message.total.toLocaleString()} · ` +
          `${formatQps(message.insertsPerSec)} inserts/s`,
      );
      break;
    }
    case "summary":
      build = { ms: message.buildMs, memoryBytes: message.memoryBytes, layers: message.layers };
      setDetail("build", `${formatQps(config!.n / (message.buildMs / 1000))} inserts/s on ${engine!.threads} threads`);
      break;
    case "baseline":
      baselineQps = message.qps;
      setDetail("baseline", `${formatQps(message.qps)} QPS, recall 1.000`);
      break;
    case "sweep":
      rows.push(message.row);
      setDetail("sweep", `ef ${message.row.ef} · recall ${message.row.recall.toFixed(3)}`);
      $<HTMLElement>("#chart-card").hidden = false;
      chart.update(rows, baselineQps);
      renderTable();
      renderTiles();
      break;
    case "kernel": {
      kernels.push(message.row);
      renderKernels();
      const k = message.row;
      setDetail("kernel", `${(k.scalarNs / k.simdNs).toFixed(1)}× faster at ${k.dim}-d`);
      break;
    }
    case "done":
      setRunning(false);
      break;
    case "error":
      $("#pipeline-note").textContent = message.message;
      renderEnvironment(engine, message.message);
      setRunning(false);
      break;
  }
}

// ---------- Rendering ----------
function renderEnvironment(info: EngineInfo | null, error?: string) {
  const chips: [string, "ok" | "warn" | "pending"][] = [];
  if (error && !info) {
    chips.push([error, "warn"]);
  } else if (info) {
    chips.push([`SIMD: ${info.simd === "wasm" ? "WebAssembly SIMD128" : info.simd}`, info.simd === "wasm" ? "ok" : "warn"]);
    chips.push(
      info.variant === "mt"
        ? [`${info.threads} threads (cross-origin isolated)`, "ok"]
        : ["Single-threaded fallback (page not cross-origin isolated)", "warn"],
    );
    chips.push(["C++20 → Emscripten → WebAssembly", "ok"]);
  }
  $("#env").replaceChildren(
    ...chips.map(([text, state]) => {
      const li = document.createElement("li");
      li.className = "chip";
      li.dataset.state = state;
      li.textContent = text;
      return li;
    }),
  );
}

const stageRow = (stage: StageId) => $<HTMLElement>(`.stages li[data-stage="${stage}"]`);

function setStage(stage: StageId, state: "running" | "done", ms?: number) {
  const li = stageRow(stage);
  li.dataset.state = state;
  if (ms !== undefined) li.querySelector(".ms")!.textContent = ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`;
  if (stage === "build" && state === "done") setBuildProgress(1);
}

function setDetail(stage: StageId, text: string) {
  stageRow(stage).querySelector(".detail")!.textContent = text;
}

function setBuildProgress(fraction: number) {
  const bar = $<HTMLElement>('.stages li[data-stage="build"] .bar');
  bar.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
  bar.querySelector("i")!.style.width = `${(fraction * 100).toFixed(1)}%`;
}

// The operating point a production system would pick: the fastest setting with recall >= 0.95.
function operatingPoint(): SweepRow | undefined {
  const good = rows.filter((r) => r.recall >= 0.95);
  if (good.length) return good.reduce((a, b) => (b.qps > a.qps ? b : a));
  return rows.reduce<SweepRow | undefined>((a, b) => (!a || b.recall > a.recall ? b : a), undefined);
}

function tile(label: string, value: string, unit: string, sub: string, hero = false) {
  const div = document.createElement("div");
  div.className = hero ? "tile hero-tile" : "tile";
  div.innerHTML = `<p class="tile-label"></p><p class="tile-value"></p><p class="tile-sub"></p>`;
  div.querySelector(".tile-label")!.textContent = label;
  const valueEl = div.querySelector(".tile-value")!;
  valueEl.textContent = value;
  if (unit) {
    const small = document.createElement("small");
    small.textContent = unit;
    valueEl.appendChild(small);
  }
  div.querySelector(".tile-sub")!.textContent = sub;
  return div;
}

function renderTiles() {
  const op = operatingPoint();
  if (!op || !build || !config) return;
  const tiles = $<HTMLElement>("#tiles");
  const seconds = build.ms / 1000;
  tiles.replaceChildren(
    tile(`Queries / s at recall ${op.recall >= 0.95 ? "≥ 0.95" : op.recall.toFixed(2)}`, formatQps(op.qps), "QPS",
      `ef ${op.ef} · recall ${op.recall.toFixed(3)} · p50 ${formatUs(op.p50Us)}`, true),
    tile("Faster than exact search", baselineQps ? `${Math.round(op.qps / baselineQps).toLocaleString()}×` : "–", "",
      `${op.distancesPerQuery.toFixed(0)} of ${config.n.toLocaleString()} vectors scored per query`),
    tile("Graph build", seconds.toFixed(1), "s",
      `${formatQps(config.n / seconds)} inserts/s on ${engine?.threads ?? 1} threads`),
    tile("Index", (build.memoryBytes / 1048576).toFixed(0), "MiB", `${build.layers} layers · M=${config.M}`),
  );
  tiles.hidden = false;
}

function renderTable() {
  const body = $("#sweep-table tbody");
  body.replaceChildren(
    ...rows.map((r) => {
      const tr = document.createElement("tr");
      const cells = [
        String(r.ef),
        r.recall.toFixed(4),
        Math.round(r.qps).toLocaleString(),
        formatUs(r.p50Us),
        formatUs(r.p99Us),
        Math.round(r.distancesPerQuery).toLocaleString(),
        baselineQps ? `${Math.round(r.qps / baselineQps).toLocaleString()}×` : "–",
      ];
      for (const text of cells) {
        const td = document.createElement("td");
        td.textContent = text;
        tr.appendChild(td);
      }
      return tr;
    }),
  );
}

function renderKernels() {
  $<HTMLElement>("#kernel-card").hidden = false;
  const maxSpeedup = Math.max(...kernels.map((k) => k.scalarNs / k.simdNs), 1);
  $("#kernels").replaceChildren(
    ...kernels.map((k) => {
      const speedup = k.scalarNs / k.simdNs;
      const row = document.createElement("div");
      row.className = "kernel-row";
      row.innerHTML = `<span class="dim"></span><div class="track"><div class="fill"></div></div><span class="value"></span>`;
      row.querySelector(".dim")!.textContent = `${k.dim}-d`;
      (row.querySelector(".fill") as HTMLElement).style.width = `${(speedup / maxSpeedup) * 100}%`;
      const value = row.querySelector(".value")!;
      value.textContent = `${speedup.toFixed(1)}× faster `;
      const detail = document.createElement("span");
      detail.textContent = `${k.scalarNs.toFixed(1)} → ${k.simdNs.toFixed(1)} ns`;
      value.appendChild(detail);
      return row;
    }),
  );
}
