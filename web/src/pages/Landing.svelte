<script lang="ts">
  import Architecture from "../components/landing/Architecture.svelte";
  import Engineering from "../components/landing/Engineering.svelte";
  import Hero from "../components/landing/Hero.svelte";
  import HowItWorks from "../components/landing/HowItWorks.svelte";
  import Problem from "../components/landing/Problem.svelte";
  import { engine } from "../lib/engine/engine.svelte";
  import { reveal } from "../lib/reveal";
  import { router } from "../lib/router.svelte";
</script>

<Hero />
<Problem />
<HowItWorks />
<Engineering />
<Architecture />

<section class="cta section" aria-labelledby="cta-title">
  <div class="glow" aria-hidden="true"></div>
  <div class="container" use:reveal>
    <h2 id="cta-title" class="section-title">Now search {engine.count ? engine.count.toLocaleString() : "29,825"} photos yourself.</h2>
    <p class="section-lede">
      Describe something, drop in a photo or point your camera. Then step through the search hop by hop, or race it against an
      exact scan.
    </p>
    <div class="actions">
      <a class="btn btn-primary" href={router.href("try")}>Open the playground <span aria-hidden="true">→</span></a>
      <a class="btn btn-ghost" href={router.href("benchmarks")}>See the benchmarks</a>
    </div>
  </div>
</section>

<style>
  .cta {
    overflow: hidden;
    text-align: center;
  }
  .cta .section-title,
  .cta .section-lede {
    margin-inline: auto;
  }
  .glow {
    position: absolute;
    left: 50%;
    top: 50%;
    width: min(900px, 120vw);
    aspect-ratio: 2;
    border-radius: 50%;
    background: radial-gradient(closest-side, var(--accent-glow), transparent);
    opacity: 0.55;
    filter: blur(20px);
    transform: translate(-50%, -50%);
    pointer-events: none;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 12px;
    margin-top: 32px;
  }
  .actions .btn {
    height: 50px;
    padding: 0 24px;
  }
</style>
