<script lang="ts">
  import { router } from "../../lib/router.svelte";
  import { reveal } from "../../lib/reveal";
  import { spotlight } from "../../lib/spotlight";

  const tiles = [
    {
      kicker: "Configures itself",
      since: "0.4",
      title: "No settings to tune",
      text: "epzero.build chooses compression, the graph build and the search effort for your data and recall target, times memory prefetching on your CPU, and says why for each choice.",
      figure: "Within 1–5% of hand-tuned settings, on three CPUs",
    },
    {
      kicker: "Keeps itself accurate",
      since: "0.5",
      title: "Fixes in place, not rebuilds",
      text: "health() says when an index has drifted from its calibration, and recalibrate, requantize and in-place growth fix it without starting over. check_queries spots queries unlike the data.",
      figure: "Text-to-image recall 0.74–0.80 → 0.945–0.960, recalibrated",
    },
    {
      kicker: "Plugs in",
      since: "0.5",
      title: "LangChain, LlamaIndex, Haystack",
      text: "A vector store for each framework, passing its own test suite, with metadata filters and hybrid keyword search run in the engine. epzero.ir scores retrieval as trec_eval does.",
      figure: "Hybrid nDCG@10 0.713 on BEIR SciFact",
    },
  ];
</script>

<section class="product section" aria-labelledby="product-title">
  <div class="container">
    <header use:reveal>
      <p class="eyebrow">Built to be used</p>
      <h2 id="product-title" class="section-title">Fast is the start. It also runs itself.</h2>
      <p class="section-lede">
        The same engine is a Python package: it configures itself for your data and machine, keeps itself accurate as the data changes, and
        drops into the frameworks you already use.
      </p>
    </header>

    <div class="grid">
      {#each tiles as tile, i (tile.kicker)}
        <article class="card tile" use:reveal={i * 0.08} use:spotlight>
          <span class="kicker">{tile.kicker} <em class:next={tile.since === "0.5"}>{tile.since}</em></span>
          <h3>{tile.title}</h3>
          <p>{tile.text}</p>
          <p class="figure num">{tile.figure}</p>
        </article>
      {/each}
      <article class="card tile install" use:reveal={0.24} use:spotlight>
        <div>
          <span class="kicker">Linux · macOS · Windows from 0.5 · Python 3.10–3.13</span>
          <code><span aria-hidden="true">$ </span>pip install epzero</code>
        </div>
        <a class="btn btn-primary" href={router.href("download")}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 19h14" /></svg>
          Download and examples
        </a>
      </article>
    </div>
  </div>
</section>

<style>
  .grid {
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
    transition: border-color 0.3s;
  }
  .tile:hover {
    border-color: var(--border-strong);
  }
  .kicker {
    color: var(--accent-strong);
    font: 600 12px var(--mono);
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .kicker em {
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: 999px;
    font-style: normal;
    color: var(--text-2);
    background: color-mix(in srgb, var(--text-2) 12%, transparent);
  }
  .kicker em.next {
    color: var(--accent-strong);
    background: color-mix(in srgb, var(--accent) 16%, transparent);
  }
  h3 {
    font-size: 22px;
    line-height: 1.2;
  }
  .tile p {
    margin: 0;
    color: var(--text-2);
    font-size: 14.5px;
    line-height: 1.65;
  }
  .tile .figure {
    margin-top: auto;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    color: var(--text);
    font: 500 12.5px/1.5 var(--mono);
  }
  .install {
    grid-column: 1 / -1;
    flex-direction: row;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
  }
  .install div {
    display: grid;
    gap: 10px;
  }
  .install code {
    padding: 0;
    border: 0;
    background: none;
    color: var(--text);
    font: 500 clamp(20px, 2.4vw, 26px) var(--mono);
  }
  .install code span {
    color: var(--muted);
  }
  .install .btn svg {
    width: 17px;
    height: 17px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2.2;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  @media (max-width: 1000px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }
</style>
