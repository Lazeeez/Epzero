<script lang="ts">
  import article from "virtual:article";
  import { router } from "../lib/router.svelte";

  // The engineering write-up, from docs/blog (converted at build time: see scripts/article.ts).
  const minutes = Math.max(1, Math.round(article.words / 230));
</script>

<article class="article container" aria-labelledby="article-title">
  <header>
    <p class="eyebrow">Engineering notes</p>
    <h1 id="article-title">{article.title}</h1>
    <p class="lede">{article.subtitle}</p>
    <p class="byline">By Lajat · {minutes} min read</p>
  </header>

  <div class="prose">
    <!-- Converted from the repository's Markdown at build time; no user input reaches it. -->
    {@html article.html}
  </div>

  <aside class="next card" aria-label="Try it">
    <div>
      <strong>See the engine run</strong>
      <p>The same C++ engine, compiled to WebAssembly: search 29,825 photos or your own, or benchmark it on your own embeddings.</p>
    </div>
    <div class="actions">
      <a class="btn btn-primary" href={router.href("try")}>Search photos</a>
      <a class="btn btn-ghost" href={router.href("benchmarks", "#lab")}>Benchmark your data</a>
    </div>
  </aside>
</article>

<style>
  .article {
    max-width: 760px;
    padding-top: calc(var(--nav-h) + 72px);
    padding-bottom: 96px;
  }
  header {
    margin-bottom: 40px;
  }
  h1 {
    margin: 12px 0 0;
    font-size: clamp(34px, 5vw, 52px);
    letter-spacing: -0.035em;
    line-height: 1.05;
    text-wrap: balance;
  }
  .lede {
    margin: 18px 0 0;
    color: var(--text-2);
    font-size: 19px;
    line-height: 1.5;
    text-wrap: pretty;
  }
  .byline {
    margin: 16px 0 0;
    color: var(--muted);
    font: 13px var(--mono);
  }
  .prose {
    color: var(--text-2);
    font-size: 17px;
    line-height: 1.75;
  }
  .prose :global(p),
  .prose :global(ul),
  .prose :global(ol) {
    margin: 0 0 20px;
  }
  .prose :global(h2) {
    margin: 52px 0 16px;
    color: var(--text);
    font-size: 27px;
    letter-spacing: -0.02em;
    line-height: 1.2;
  }
  .prose :global(h3) {
    margin: 34px 0 12px;
    color: var(--text);
    font-size: 20px;
    letter-spacing: -0.01em;
  }
  .prose :global(strong) {
    color: var(--text);
  }
  .prose :global(a) {
    color: var(--accent-strong);
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .prose :global(li) {
    margin-bottom: 8px;
  }
  .prose :global(code) {
    padding: 2px 6px;
    border-radius: 6px;
    background: var(--surface-3);
    color: var(--text);
    font: 0.88em var(--mono);
  }
  .prose :global(.table-wrap) {
    --table-bg: var(--page);
    margin: 0 0 24px;
    border: 1px solid var(--border);
    border-radius: 12px;
  }
  .prose :global(table) {
    width: 100%;
    border-collapse: collapse;
    font-size: 14.5px;
    line-height: 1.5;
  }
  .prose :global(th),
  .prose :global(td) {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
  }
  .prose :global(th) {
    background: var(--surface);
    color: var(--text);
    font-weight: 600;
    white-space: nowrap;
  }
  .prose :global(tr:last-child td) {
    border-bottom: 0;
  }
  /* Phones: a table of three or more columns would squeeze its prose into a tall, narrow column.
     Each row becomes a card instead, every cell under its column's name. */
  @media (max-width: 640px) {
    .prose :global(.table-wrap.stack) {
      overflow: visible;
      border: 0;
      border-radius: 0;
      background: none;
    }
    .prose :global(.stack thead) {
      display: none;
    }
    .prose :global(.stack :is(table, tbody, tr, td)) {
      display: block;
    }
    .prose :global(.stack tr) {
      margin-bottom: 12px;
      padding: 4px 16px;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--surface);
    }
    .prose :global(.stack td) {
      position: static;
      padding: 10px 0;
      background: none;
    }
    .prose :global(.stack td:not(:last-child)) {
      border-bottom: 1px solid var(--border);
    }
    .prose :global(.stack td::before) {
      content: attr(data-label);
      display: block;
      margin-bottom: 2px;
      color: var(--muted);
      font: 600 11px var(--mono);
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .prose :global(.stack td:first-child) {
      color: var(--text);
      font-weight: 600;
    }
    .prose :global(.stack td:first-child::before) {
      display: none; /* the row's name heads its card */
    }
  }
  .next {
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: center;
    gap: 20px;
    margin-top: 56px;
    padding: 24px;
  }
  .next strong {
    font-size: 17px;
  }
  .next p {
    margin: 6px 0 0;
    color: var(--text-2);
    font-size: 14px;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
  }
  @media (max-width: 700px) {
    .article {
      padding-top: calc(var(--nav-h) + 40px);
    }
    .prose {
      font-size: 16px;
    }
    .next {
      grid-template-columns: 1fr;
    }
  }
</style>
