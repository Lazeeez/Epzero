<script lang="ts">
  import { reveal } from "../../lib/reveal";

  const browser = [
    { title: "Your query", detail: "text · photo · webcam" },
    { title: "CLIP encoder", detail: "ONNX Runtime · WASM · 8-bit" },
    { title: "Epzero engine", detail: "C++20 → WASM · threads + SIMD128" },
    { title: "Results + trace", detail: "every hop, rendered in WebGL" },
  ];
  const server = [
    { title: "Client", detail: "HTTP / gRPC" },
    { title: "Go router", detail: "sharding · hedged fan-out · top-k merge" },
    { title: "epzero-node × N", detail: "primary → backup, sync replication" },
    { title: "WAL + snapshots", detail: "CRC32C · group commit · checkpoints" },
  ];
</script>

<section class="architecture section" aria-labelledby="architecture-title">
  <div class="container">
    <header use:reveal>
      <p class="eyebrow">Architecture</p>
      <h2 id="architecture-title" class="section-title">One engine, two homes.</h2>
      <p class="section-lede">
        The same C++ core compiles to a native server and to the WebAssembly running on this page. Nothing here calls a
        backend: the model, the index and the search all run on your device.
      </p>
    </header>

    {#each [{ name: "In this tab", items: browser, tone: "accent" }, { name: "As a cluster", items: server, tone: "violet" }] as lane, l (lane.name)}
      <div class="lane {lane.tone}" use:reveal={l * 0.1}>
        <span class="lane-name">{lane.name}</span>
        <ol>
          {#each lane.items as item, i (item.title)}
            <li style:--i={i}>
              <strong>{item.title}</strong>
              <span>{item.detail}</span>
            </li>
          {/each}
        </ol>
      </div>
    {/each}
  </div>
</section>

<style>
  .lane {
    --tone: var(--accent);
    display: grid;
    grid-template-columns: 130px 1fr;
    align-items: center;
    gap: 20px;
    margin-top: 48px;
  }
  .lane.violet {
    --tone: var(--accent-2);
  }
  .lane + .lane {
    margin-top: 20px;
  }
  .lane-name {
    color: var(--tone);
    font: 600 12.5px var(--mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }
  ol {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 36px;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  li {
    position: relative;
    display: grid;
    gap: 4px;
    padding: 18px;
    border: 1px solid var(--border);
    border-radius: 14px;
    background: color-mix(in srgb, var(--surface-solid) 72%, transparent);
  }
  li:first-child {
    border-color: color-mix(in srgb, var(--tone) 45%, var(--border));
  }
  strong {
    font-size: 15px;
  }
  li span {
    color: var(--muted);
    font-size: 12.5px;
    line-height: 1.45;
  }
  /* Connectors: a line with a pulse travelling along it, staggered down the pipeline. */
  li:not(:last-child)::after {
    content: "";
    position: absolute;
    top: 50%;
    left: 100%;
    width: 36px;
    height: 2px;
    background:
      linear-gradient(90deg, transparent, var(--tone), transparent) no-repeat,
      var(--border-strong);
    background-size: 40% 100%, 100% 100%;
    animation: pulse 1.8s linear infinite;
    animation-delay: calc(var(--i) * 0.45s);
  }
  @keyframes pulse {
    from {
      background-position: -60% 0, 0 0;
    }
    to {
      background-position: 160% 0, 0 0;
    }
  }
  @media (max-width: 900px) {
    .lane {
      grid-template-columns: 1fr;
      gap: 12px;
    }
    ol {
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    li::after {
      display: none;
    }
  }
</style>
