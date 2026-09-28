<script lang="ts">
  // From docs/performance.md (SQ8 section): same machine and methodology as the other benchmarks.
  const datasets = [
    {
      name: "SIFT1M",
      detail: "1M × 128-D · L2 · ef = 80",
      memoryLabel: "Snapshot",
      rows: [
        { mode: "float32", recall: 0.9744, qps: 10394, memory: 658, unit: "MB" },
        { mode: "SQ8 + re-rank", recall: 0.9744, qps: 15365, memory: 791, unit: "MB", best: true },
        { mode: "SQ8, codes only", recall: 0.9666, qps: 17015, memory: 279, unit: "MB" },
      ],
    },
    {
      name: "CLIP ViT-B/32",
      detail: "28,825 × 512-D · cosine · ef = 64",
      memoryLabel: "Index",
      rows: [
        { mode: "float32", recall: 0.9979, qps: 9229, memory: 64, unit: "MiB" },
        { mode: "SQ8 + re-rank", recall: 0.9981, qps: 11570, memory: 78, unit: "MiB", best: true },
        { mode: "SQ8, codes only", recall: 0.9879, qps: 13364, memory: 22, unit: "MiB" },
      ],
    },
  ];
  let active = $state(0);
  const set = $derived(datasets[active]);
  const maxQps = $derived(Math.max(...set.rows.map((r) => r.qps)));
  const maxMemory = $derived(Math.max(...set.rows.map((r) => r.memory)));
</script>

<div class="quant">
  <div class="tabs" role="tablist" aria-label="Dataset">
    {#each datasets as d, i (d.name)}
      <button role="tab" aria-selected={active === i} class:on={active === i} onclick={() => (active = i)}>
        {d.name}<span>{d.detail}</span>
      </button>
    {/each}
  </div>

  <div class="table" role="table" aria-label="{set.name}: recall, throughput and memory by storage mode">
    <div class="head" role="row">
      <span role="columnheader">Mode</span>
      <span role="columnheader">Recall@10</span>
      <span role="columnheader">Queries / s</span>
      <span role="columnheader">{set.memoryLabel}</span>
    </div>
    {#each set.rows as row (set.name + row.mode)}
      <div class="line" class:best={row.best} role="row">
        <span class="mode" role="cell">{row.mode}</span>
        <span class="num recall" role="cell">{row.recall.toFixed(4)}</span>
        <span class="cell" role="cell">
          <i class="qps" style:width="{(row.qps / maxQps) * 100}%"></i>
          <b class="num">
            {row.qps.toLocaleString()}
            {#if row.mode !== "float32"}<em>+{Math.round((row.qps / set.rows[0].qps - 1) * 100)}%</em>{/if}
          </b>
        </span>
        <span class="cell" role="cell">
          <i class="mem" style:width="{(row.memory / maxMemory) * 100}%"></i>
          <b class="num">{row.memory} {row.unit}</b>
        </span>
      </div>
    {/each}
  </div>
  <p class="note">
    {#if active === 0}
      SIFT's components are small integers, so 8-bit codes are nearly lossless on it, which flatters SQ8. The CLIP tab is the
      honest check.
    {:else}
      Real float embeddings with 1,000 held-out queries. With re-ranking SQ8 is strictly better; codes-only trades a recall
      ceiling of ~0.989 for ~3× less memory.
    {/if}
  </p>
</div>

<style>
  .tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 18px;
  }
  .tabs button {
    display: grid;
    gap: 2px;
    padding: 10px 16px;
    border: 1px solid var(--border);
    border-radius: 12px;
    background: transparent;
    color: var(--text-2);
    font-weight: 600;
    text-align: left;
    cursor: pointer;
  }
  .tabs button span {
    color: var(--muted);
    font: 400 11.5px var(--mono);
  }
  .tabs button.on {
    border-color: var(--accent);
    background: var(--accent-wash);
    color: var(--text);
  }
  .table {
    display: grid;
    gap: 6px;
  }
  .head,
  .line {
    display: grid;
    grid-template-columns: 150px 90px 1fr 1fr;
    align-items: center;
    gap: 16px;
  }
  .head {
    color: var(--muted);
    font: 12px var(--mono);
    padding: 0 12px;
  }
  .line {
    padding: 10px 12px;
    border: 1px solid transparent;
    border-radius: 12px;
  }
  .line.best {
    border-color: color-mix(in srgb, var(--accent) 40%, transparent);
    background: var(--accent-wash);
  }
  .mode {
    font-weight: 550;
  }
  .recall {
    font: 13.5px var(--mono);
  }
  .cell {
    position: relative;
    display: flex;
    align-items: center;
    height: 28px;
  }
  .cell i {
    position: absolute;
    inset: 0 auto 0 0;
    border-radius: 7px;
    transition: width 0.7s var(--ease);
  }
  .qps {
    background: color-mix(in srgb, var(--accent) 45%, transparent);
  }
  .mem {
    background: color-mix(in srgb, var(--accent-2) 35%, transparent);
  }
  .cell b {
    position: relative;
    padding-left: 10px;
    font: 600 13px var(--mono);
    white-space: nowrap;
  }
  em {
    margin-left: 6px;
    color: var(--good);
    font-style: normal;
  }
  .note {
    margin: 14px 0 0;
    color: var(--muted);
    font-size: 13px;
  }
  @media (max-width: 700px) {
    .head {
      display: none;
    }
    .line {
      grid-template-columns: 1fr auto;
      gap: 8px;
    }
    .cell {
      grid-column: 1 / -1;
    }
  }
</style>
