import "../shared/base.css";
import "./explorer.css";
import { initTheme, onThemeChange } from "../shared/theme";
import type { FromWorker, Query, Trace } from "./protocol";
import { GraphScene } from "./scene";
import type { FromText } from "./text.worker";

interface Manifest {
  count: number;
  dim: number;
  thumbnail: string;
}

interface Meta {
  id: string[];
  host: string[];
  path: string[];
  labels: string[][];
  title: string[];
  author: string[];
  landing: string[];
  license: number[];
  licenses: string[];
}

const $ = <T extends HTMLElement>(selector: string) => document.querySelector<T>(selector)!;
const DATA = `${import.meta.env.BASE_URL}data/openimages/`;
const K = 10;

initTheme($<HTMLButtonElement>("button.theme-toggle"));
const scene = new GraphScene($("#stage"));
onThemeChange(() => scene.themeChanged());

// Keep the graph centered in the space the panel leaves free (left side on desktop, bottom sheet
// on phones).
const panel = $("aside.panel");
const updateInsets = () => {
  const box = panel.getBoundingClientRect();
  if (innerWidth > 760) scene.setInsets(box.right, 0);
  else scene.setInsets(0, innerHeight - box.top);
};
new ResizeObserver(updateInsets).observe(panel);
window.addEventListener("resize", updateInsets);

let meta: Meta | null = null;
let count = 0;
let lastQuery: Query | null = null;
let lastTrace: Trace | null = null;
let seq = 0;

const worker = new Worker(new URL("./explorer.worker.ts", import.meta.url), { type: "module" });

// ---------- Loading ----------
function setLoading(text: string, fraction: number | null) {
  $("#loading-text").textContent = text;
  $("#loading-pct").textContent = fraction === null ? "" : `${Math.round(fraction * 100)}%`;
  $("#loading-bar").style.width = `${((fraction ?? 0) * 100).toFixed(1)}%`;
}

async function start() {
  const [manifest, metaJson, layout] = await Promise.all([
    fetch(`${DATA}manifest.json`).then((r) => r.json() as Promise<Manifest>),
    fetch(`${DATA}meta.json`).then((r) => r.json() as Promise<Meta>),
    fetch(`${DATA}layout.u16`).then((r) => r.arrayBuffer()),
  ]);
  meta = metaJson;
  count = manifest.count;
  $("#dataset-count").textContent = count.toLocaleString();
  const raw = new Uint16Array(layout);
  scene.setLayout(Float32Array.from(raw, (v) => (v / 65535) * 2 - 1));
  scene.setBuilt(0);
  worker.postMessage({ type: "load", base: DATA, count, dim: manifest.dim, M: 16, efConstruction: 128 });
}

worker.onmessage = (event: MessageEvent<FromWorker>) => {
  const message = event.data;
  switch (message.type) {
    case "progress":
      if (message.stage === "download") {
        setLoading(`Downloading ${(message.total / 1e6).toFixed(1)} MB of CLIP vectors`, message.done / message.total);
      } else {
        setLoading(`Building the HNSW graph · ${message.done.toLocaleString()} photos`, message.done / message.total);
        scene.setBuilt(message.done);
      }
      break;
    case "built": {
      const g = message.graph;
      scene.setGraph(g);
      setLoading(
        `Graph built in ${(g.buildMs / 1000).toFixed(1)} s on ${g.threads} thread${g.threads > 1 ? "s" : ""} · ` +
          `${g.maxLevel + 1} layers`,
        1,
      );
      setTimeout(() => ($("#loading").hidden = true), 2500);
      for (const id of ["#surprise", "#replay"]) $<HTMLButtonElement>(id).disabled = false;
      surprise(); // bring the page to life with a first query
      startTextSearch(); // after the build, so the graph gets every core first
      break;
    }
    case "result":
      if (message.seq !== seq) return; // a newer search superseded this one
      lastTrace = message.trace;
      scene.play(message.trace);
      renderStats(message.trace, message.query);
      renderResults(message.trace, message.query);
      break;
    case "error":
      setLoading(`Something went wrong: ${message.message}`, null);
      $("#loading").hidden = false;
      break;
  }
};

// ---------- Search ----------
function search(query: Query) {
  lastQuery = query;
  seq += 1;
  const ef = Number($<HTMLInputElement>("#ef").value);
  worker.postMessage({ type: "search", seq, query, k: K, ef });
}

function surprise() {
  if (count) search({ kind: "id", id: Math.floor(Math.random() * count) });
}

// ---------- Text search ----------
const queryInput = $<HTMLInputElement>("#query");
const searchButton = $<HTMLButtonElement>("#search button");
let textWorker: Worker | null = null;
let textSeq = 0;
let encodeMs: number | null = null;

function startTextSearch() {
  $("#search").hidden = false;
  textWorker = new Worker(new URL("./text.worker.ts", import.meta.url), { type: "module" });
  textWorker.onmessage = (event: MessageEvent<FromText>) => {
    const message = event.data;
    switch (message.type) {
      case "progress":
        if (message.total > 0) {
          queryInput.placeholder = `Loading the CLIP text model… ${Math.round((message.loaded / message.total) * 100)}%`;
        }
        break;
      case "ready":
        queryInput.disabled = false;
        searchButton.disabled = false;
        queryInput.placeholder = "Describe a photo, e.g. “a dog on the beach”";
        $("#examples").hidden = false;
        break;
      case "embedding":
        if (message.seq !== textSeq) return; // superseded by a newer query
        encodeMs = message.ms;
        search({ kind: "vector", vector: message.vector, text: message.text });
        break;
      case "error":
        console.warn(message.message);
        queryInput.placeholder = "Text search is unavailable in this browser";
        break;
    }
  };
}

function searchText(text: string) {
  const trimmed = text.trim();
  if (!trimmed || !textWorker) return;
  textSeq += 1;
  textWorker.postMessage({ type: "encode", seq: textSeq, text: trimmed });
}

$("#search").addEventListener("submit", (event) => {
  event.preventDefault();
  searchText(queryInput.value);
});
for (const chip of document.querySelectorAll<HTMLButtonElement>("#examples button")) {
  chip.addEventListener("click", () => {
    queryInput.value = chip.textContent ?? "";
    searchText(queryInput.value);
  });
}
window.addEventListener("keydown", (event) => {
  if (event.key === "/" && document.activeElement !== queryInput && !queryInput.disabled) {
    event.preventDefault();
    queryInput.focus();
  }
});

$("#surprise").addEventListener("click", surprise);
$("#replay").addEventListener("click", () => lastTrace && scene.play(lastTrace));
let efTimer = 0;
$("#ef").addEventListener("input", (event) => {
  $("#ef-value").textContent = (event.target as HTMLInputElement).value;
  clearTimeout(efTimer);
  efTimer = window.setTimeout(() => lastQuery && search(lastQuery), 160);
});
scene.onClick((id) => search({ kind: "id", id }));

// ---------- Rendering ----------
function thumbnail(id: number, size: "n" | "q" | "z" = "n") {
  return `https://c${meta!.host[id]}.staticflickr.com/${meta!.path[id]}_${size}.jpg`;
}

function caption(id: number) {
  const labels = meta!.labels[id];
  return labels.length ? labels.join(" · ") : meta!.title[id] || "Untitled";
}

function image(id: number, alt: string) {
  const img = document.createElement("img");
  img.crossOrigin = "anonymous"; // Flickr sends CORS headers, which cross-origin isolation requires
  img.referrerPolicy = "no-referrer";
  img.loading = "lazy";
  img.decoding = "async";
  img.alt = alt;
  img.src = thumbnail(id);
  return img;
}

const escapeHtml = (text: string) =>
  text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

const formatTime = (micros: number) => (micros >= 1000 ? `${(micros / 1000).toFixed(1)} ms` : `${micros.toFixed(0)} µs`);

function renderStats(trace: Trace, query: Query) {
  const exact = new Set(trace.exact);
  const hits = Array.from(trace.results).filter((id) => exact.has(id)).length;
  const speedup = trace.exactMicros / Math.max(trace.micros, 1);
  const scored = trace.distanceComputations;
  const stat = (value: string, label: string) => `<div class="stat"><b>${value}</b><span>${label}</span></div>`;
  const el = $("#stats");
  el.innerHTML =
    stat(formatTime(trace.micros), "HNSW search") +
    stat(`${hits}/${trace.results.length}`, "same as exact scan") +
    stat(`${speedup.toFixed(0)}×`, "faster than exact") +
    `<div class="stat wide"><span>Scored <strong>${scored.toLocaleString()}</strong> of ${count.toLocaleString()} photos ` +
    `(${((scored / count) * 100).toFixed(1)}%) across <strong>${trace.entryLayer + 1}</strong> layers</span>` +
    `<span>exact scan ${formatTime(trace.exactMicros)}</span></div>` +
    (query.kind === "vector" && encodeMs !== null
      ? `<div class="stat wide"><span>“${escapeHtml(query.text)}” → CLIP text vector</span>` +
        `<span>${encodeMs.toFixed(0)} ms</span></div>`
      : "");
  el.hidden = false;
}

function renderResults(trace: Trace, query: Query) {
  const list = $("#results");
  list.replaceChildren(
    ...Array.from(trace.results).map((id, rank) => {
      const li = document.createElement("li");
      li.style.animationDelay = `${rank * 30}ms`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "result";
      const label = caption(id);
      button.setAttribute("aria-label", `Result ${rank + 1}: ${label}. Search for similar photos.`);
      const similarity = 1 - trace.resultDistances[rank];
      const isQuery = query.kind === "id" && query.id === id;
      button.append(
        image(id, label),
        Object.assign(document.createElement("span"), { className: "rank", textContent: isQuery ? "query" : `#${rank + 1}` }),
        Object.assign(document.createElement("span"), { className: "score", textContent: similarity.toFixed(2) }),
        Object.assign(document.createElement("span"), { className: "caption", textContent: label }),
      );
      button.addEventListener("click", () => search({ kind: "id", id }));
      li.append(button);
      return li;
    }),
  );
}

// ---------- Hover card ----------
const card = $("#hovercard");
scene.onHover((hover) => {
  if (!hover || !meta) {
    card.hidden = true;
    return;
  }
  const id = hover.id;
  const body = document.createElement("div");
  body.append(
    Object.assign(document.createElement("strong"), { textContent: caption(id) }),
    Object.assign(document.createElement("span"), { textContent: `© ${meta.author[id] || "Unknown"} · CC BY 2.0 · click to search` }),
  );
  card.replaceChildren(image(id, caption(id)), body);
  card.style.left = `${Math.min(Math.max(hover.x, 120), innerWidth - 120)}px`;
  card.style.top = `${Math.max(hover.y, 200)}px`;
  card.hidden = false;
});

start().catch((error) => setLoading(`Could not load the dataset: ${error}`, null));
