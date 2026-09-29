<script lang="ts">
  import { router } from "../lib/router.svelte";
  import Logo from "./Logo.svelte";
</script>

<footer class="footer">
  <div class="container inner">
    <div class="about">
      <a class="brand" href={router.href("home")}><Logo size={20} /><span>Oryx</span></a>
      <p>An HNSW vector database written from scratch in C++20, compiled to WebAssembly for this site. Nothing you search leaves your browser.</p>
    </div>
    <nav aria-label="Footer">
      <a href={router.href("home", "#how")}>How it works</a>
      <a href={router.href("try")}>Try it</a>
      <a href={router.href("play")}>Play</a>
      <a href={router.href("benchmarks")}>Benchmarks</a>
    </nav>
    <p class="credits">
      Photos: <a href="https://storage.googleapis.com/openimages/web/index.html" rel="noopener">Open Images</a>, CC BY 2.0 via Flickr
      (authors credited on each photo) · Labels CC BY 4.0 · Built by Lajat
    </p>
  </div>
</footer>

<style>
  .footer {
    position: relative;
    z-index: 1;
    border-top: 1px solid var(--border);
    background: var(--page);
  }
  .inner {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 24px 48px;
    padding: 48px 0 40px;
  }
  .brand {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    color: var(--text);
    font-weight: 650;
  }
  .about p {
    max-width: 440px;
    margin: 12px 0 0;
    color: var(--text-2);
    font-size: 14px;
  }
  nav {
    display: flex;
    flex-direction: column;
    gap: 10px;
    font-size: 14px;
  }
  nav a {
    color: var(--text-2);
  }
  nav a:hover {
    color: var(--text);
  }
  .credits {
    grid-column: 1 / -1;
    margin: 0;
    padding-top: 24px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 12.5px;
  }
  .credits a {
    color: var(--text-2);
  }
  @media (max-width: 640px) {
    .inner {
      grid-template-columns: 1fr;
    }
    nav {
      flex-direction: row;
      gap: 20px;
    }
  }
</style>
