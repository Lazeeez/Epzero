<script lang="ts">
  import { router } from "../lib/router.svelte";
</script>

<section class="notfound container" aria-labelledby="notfound-title">
  <p class="eyebrow">404</p>
  <h1 id="notfound-title">This page drifted out of the graph.</h1>
  <p class="lede">
    There's nothing at <code>{location.pathname}</code>. The link may be mistyped, or the page may have moved.
  </p>
  <div class="actions">
    <a class="btn btn-primary" href={router.href("home")}>Go to the home page</a>
    <a class="btn btn-ghost" href={router.href("try")}>Search photos</a>
  </div>
</section>

<style>
  .notfound {
    min-height: 70vh;
    padding: calc(var(--nav-h) + 96px) 0 64px;
  }
  h1 {
    max-width: 720px;
    font-size: clamp(34px, 5vw, 56px);
    letter-spacing: -0.04em;
  }
  .lede {
    max-width: 620px;
    margin: 18px 0 0;
    color: var(--text-2);
    font-size: 17px;
    overflow-wrap: anywhere;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 30px;
  }
</style>
