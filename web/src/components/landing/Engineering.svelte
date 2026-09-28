<script lang="ts">
  import { speedup } from "../../lib/bench";
  import { router } from "../../lib/router.svelte";
  import { reveal } from "../../lib/reveal";

  // SIFT1M, equal recall, one thread; relative to hnswlib.
  const vsHnswlib = speedup("hnswlib");
  const vsFaiss = speedup("faiss");
  const libraries = [
    { name: "Oryx", value: vsHnswlib, accent: true },
    { name: "FAISS", value: vsHnswlib / vsFaiss },
    { name: "hnswlib", value: 1 },
  ];
  const steps = [
    { name: "Baseline", qps: 7706 },
    { name: "Lock-free reads", qps: 7879 },
    { name: "Batched prefetch", qps: 9052 },
    { name: "2 MiB huge pages", qps: 10189 },
  ];
</script>

<section class="engineering section" aria-labelledby="engineering-title">
  <div class="container">
    <header use:reveal>
      <p class="eyebrow">Under the hood</p>
      <h2 id="engineering-title" class="section-title">Engineered down to the cache line.</h2>
      <p class="section-lede">
        Every optimization was kept only if an interleaved A/B test on SIFT1M showed a gain, and the negative results are
        written up too.
      </p>
    </header>

    <div class="bento">
      <article class="card tile wide" use:reveal>
        <div class="copy">
          <span class="kicker">Throughput at equal recall</span>
          <h3><span class="big num">{vsHnswlib.toFixed(2)}×</span> hnswlib, {vsFaiss.toFixed(2)}× FAISS</h3>
          <p>
            Same algorithm, same graph quality. The difference is memory-system engineering: search is bound by DRAM latency, at
            ~63 ns per distance, so the wins come from keeping more cache misses in flight.
          </p>
        </div>
        <div class="bars" role="img" aria-label="Relative queries per second: {libraries.map((l) => `${l.name} ${l.value.toFixed(2)}`).join(', ')}">
          {#each libraries as lib, i (lib.name)}
            <div class="bar-row" style:--w="{(lib.value / vsHnswlib) * 100}%" style:--d="{0.15 + i * 0.12}s">
              <span>{lib.name}</span>
              <div class="track"><i class:accent={lib.accent}></i></div>
              <b class="num">{lib.value.toFixed(2)}×</b>
            </div>
          {/each}
        </div>
      </article>

      <article class="card tile" use:reveal={0.08}>
        <span class="kicker">+31% from profiling</span>
        <h3>Memory latency, attacked</h3>
        <ol class="ladder num">
          {#each steps as step, i (step.name)}
            <li style:--w="{(step.qps / 10189) * 100}%" style:--d="{0.2 + i * 0.1}s">
              <span>{step.name}</span><b>{step.qps.toLocaleString()}</b>
            </li>
          {/each}
        </ol>
        <p class="small">QPS on SIFT1M, one thread. Prefetch the next 32 neighbours at once; back layer 0 with huge pages.</p>
      </article>

      <article class="card tile" use:reveal={0.16}>
        <span class="kicker">SQ8 quantization</span>
        <h3><span class="big num">+48%</span> QPS, same recall</h3>
        <p>
          Per-dimension 8-bit codes scored with an asymmetric distance kernel, then the top candidates re-ranked exactly.
          Recall stays at 0.9744; vectors shrink 4×.
        </p>
      </article>

      <article class="card tile" use:reveal>
        <span class="kicker">Concurrency</span>
        <h3>Lock-free reads</h3>
        <p>
          Neighbour slots are <code>std::atomic_ref</code> with release/acquire ordering, so a search never waits behind an
          insert. Verified with ThreadSanitizer under concurrent load.
        </p>
      </article>

      <article class="card tile" use:reveal={0.08}>
        <span class="kicker">Durability</span>
        <h3>Crash-safe write-ahead log</h3>
        <p>
          CRC32C-framed records, group commit, atomic checkpoints via rename, and recovery that truncates a torn tail. Deletes are
          tombstones, compacted on rebuild.
        </p>
      </article>

      <article class="card tile" use:reveal={0.16}>
        <span class="kicker">SIMD</span>
        <h3>One kernel, four ISAs</h3>
        <div class="chips">
          <span>AVX-512</span><span>AVX2 + FMA</span><span>NEON</span><span>WASM SIMD128</span>
        </div>
        <p class="small">Selected at runtime by CPU detection. The build you're running uses SIMD128.</p>
      </article>

      <article class="card tile wide cluster" use:reveal>
        <div class="copy">
          <span class="kicker">Distributed</span>
          <h3>Replicated, sharded, and it fails over</h3>
          <p>
            C++ gRPC nodes replicate synchronously from primary to backup, with epochs as fencing tokens. A Go router shards
            writes, fans searches out with hedged requests, merges the top-k and promotes a backup when a primary dies. CI kills
            a primary on every push and checks nothing is lost.
          </p>
        </div>
        <svg class="topology" viewBox="0 0 300 170" aria-hidden="true">
          <g class="links">
            <path class="cut" d="M150 34 L66 118" /><path d="M150 34 L150 118" /><path d="M150 34 L234 118" />
            <path class="repl" d="M66 118 C 66 150, 150 150, 150 118" />
            <path class="repl" d="M150 118 C 150 150, 234 150, 234 118" />
          </g>
          <g class="flow">
            <path d="M150 34 L150 118" /><path d="M150 34 L234 118" />
          </g>
          <rect x="112" y="14" width="76" height="30" rx="9" class="router" />
          <text x="150" y="33">router</text>
          {#each [66, 150, 234] as x, i (x)}
            <rect x={x - 39} y="104" width="78" height="28" rx="8" class="node" class:down={i === 0} />
            <text {x} y="122">{["old primary", "promoted", "replica"][i]}</text>
          {/each}
        </svg>
      </article>
    </div>

    <p class="more" use:reveal>
      <a href={router.href("benchmarks")}>Methodology, charts and the ideas that didn't work <span aria-hidden="true">→</span></a>
    </p>
  </div>
</section>

<style>
  .bento {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 56px;
  }
  .tile {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 26px;
    background: color-mix(in srgb, var(--surface-solid) 72%, transparent);
    transition: border-color 0.3s, transform 0.4s var(--ease);
  }
  .tile:hover {
    border-color: var(--border-strong);
  }
  .wide {
    grid-column: span 2;
    display: grid;
    grid-template-columns: 1.1fr 1fr;
    align-items: center;
    gap: 28px;
  }
  .copy {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .kicker {
    color: var(--accent-strong);
    font: 600 12px var(--mono);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  h3 {
    font-size: 22px;
    line-height: 1.2;
  }
  .big {
    display: block;
    margin-bottom: 4px;
    font-size: 52px;
    font-weight: 700;
    letter-spacing: -0.045em;
    background: linear-gradient(90deg, var(--accent-strong), var(--accent-2));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .tile p {
    margin: 0;
    color: var(--text-2);
    font-size: 14.5px;
    line-height: 1.65;
  }
  .tile p.small {
    color: var(--muted);
    font-size: 13px;
  }
  .bars {
    display: grid;
    gap: 14px;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 70px 1fr 52px;
    align-items: center;
    gap: 12px;
    font-size: 13.5px;
    color: var(--text-2);
  }
  .bar-row b {
    color: var(--text);
    font: 600 13.5px var(--mono);
    text-align: right;
  }
  .track {
    height: 26px;
    border-radius: 8px;
    background: var(--surface-3);
    overflow: hidden;
  }
  .track i {
    display: block;
    width: 0;
    height: 100%;
    border-radius: 8px;
    background: var(--axis);
    transition: width 1.4s var(--ease) var(--d);
  }
  .track i.accent {
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    box-shadow: 0 0 24px var(--accent-glow);
  }
  :global(.revealed) .track i {
    width: var(--w);
  }
  .ladder {
    display: grid;
    gap: 8px;
    margin: 4px 0 0;
    padding: 0;
    list-style: none;
  }
  .ladder li {
    position: relative;
    display: flex;
    justify-content: space-between;
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 12.5px;
    color: var(--text-2);
    overflow: hidden;
    z-index: 0;
  }
  .ladder li::before {
    content: "";
    position: absolute;
    inset: 0 auto 0 0;
    z-index: -1;
    width: 0;
    background: var(--accent-wash);
    border-right: 2px solid var(--accent);
    transition: width 1.2s var(--ease) var(--d);
  }
  :global(.revealed) .ladder li::before {
    width: var(--w);
  }
  .ladder b {
    color: var(--text);
    font: 600 12.5px var(--mono);
  }
  .chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .chips span {
    padding: 6px 11px;
    border: 1px solid var(--border-strong);
    border-radius: 99px;
    font: 12.5px var(--mono);
    color: var(--text-2);
  }
  .chips span:last-child {
    border-color: var(--accent);
    color: var(--accent-strong);
    background: var(--accent-wash);
  }
  .topology {
    width: 100%;
    max-height: 200px;
  }
  .topology .links path {
    fill: none;
    stroke: var(--border-strong);
    stroke-width: 1.5;
  }
  .topology .links .cut {
    stroke: var(--warn);
    stroke-dasharray: 2 5;
    opacity: 0.6;
  }
  .topology .links .repl {
    stroke: var(--accent-2);
    stroke-dasharray: 3 4;
    opacity: 0.7;
  }
  .topology .flow path {
    fill: none;
    stroke: var(--accent);
    stroke-width: 2;
    stroke-dasharray: 6 110;
    animation: flow 2.2s linear infinite;
  }
  .topology rect {
    fill: var(--surface-2);
    stroke: var(--border-strong);
  }
  .topology rect.router {
    stroke: var(--accent);
  }
  .topology rect.down {
    stroke: var(--warn);
    stroke-dasharray: 3 3;
    fill: transparent;
  }
  .topology text {
    fill: var(--text-2);
    font: 10.5px var(--mono);
    text-anchor: middle;
  }
  .more {
    margin: 36px 0 0;
    text-align: center;
  }
  .more a {
    font-weight: 550;
  }
  @keyframes flow {
    from {
      stroke-dashoffset: 116;
    }
    to {
      stroke-dashoffset: 0;
    }
  }
  @media (max-width: 1000px) {
    .bento {
      grid-template-columns: 1fr 1fr;
    }
  }
  @media (max-width: 700px) {
    .bento {
      grid-template-columns: 1fr;
    }
    .wide {
      grid-column: auto;
      grid-template-columns: 1fr;
    }
    .big {
      font-size: 44px;
    }
  }
</style>
