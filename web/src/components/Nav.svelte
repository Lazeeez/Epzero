<script lang="ts">
  import { router } from "../lib/router.svelte";
  import { isDark, toggleTheme } from "../lib/theme";
  import Logo from "./Logo.svelte";

  let scrolled = $state(false);
  let dark = $state(isDark());

  $effect(() => {
    const onScroll = () => (scrolled = scrollY > 12);
    onScroll();
    addEventListener("scroll", onScroll, { passive: true });
    return () => removeEventListener("scroll", onScroll);
  });

  function theme() {
    toggleTheme();
    dark = isDark();
  }
</script>

<header class="nav" class:scrolled class:overlay={router.route === "try"}>
  <div class="inner">
    <a class="brand" href={router.href("home")} aria-label="Oryx, home">
      <Logo size={24} />
      <span>Oryx</span>
    </a>
    <nav aria-label="Main">
      <a href={router.href("try")} class="link" aria-current={router.route === "try" ? "page" : undefined}>Try it</a>
      <a href={router.href("play")} class="link" aria-current={router.route === "play" ? "page" : undefined}>Play</a>
      <a href={router.href("benchmarks")} class="link" aria-current={router.route === "benchmarks" ? "page" : undefined}>Benchmarks</a>
    </nav>
    <div class="right">
      <button class="icon" type="button" onclick={theme} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}>
        {#if dark}
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4.2" /><path d="M12 2.5v2.2M12 19.3v2.2M4.6 4.6l1.6 1.6M17.8 17.8l1.6 1.6M2.5 12h2.2M19.3 12h2.2M4.6 19.4l1.6-1.6M17.8 6.2l1.6-1.6" /></svg>
        {:else}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14.3A8.5 8.5 0 0 1 9.7 3.5a8.5 8.5 0 1 0 10.8 10.8Z" /></svg>
        {/if}
      </button>
      {#if router.route !== "try"}
        <a class="cta" href={router.href("try")}>Try it <span aria-hidden="true">→</span></a>
      {/if}
    </div>
  </div>
</header>

<style>
  .nav {
    position: fixed;
    inset: 0 0 auto 0;
    z-index: 50;
    height: var(--nav-h);
    padding-inline: env(safe-area-inset-left, 0px) env(safe-area-inset-right, 0px); /* landscape notches */
    transition: background 0.3s, border-color 0.3s, backdrop-filter 0.3s;
    border-bottom: 1px solid transparent;
  }
  .nav.scrolled,
  .nav.overlay {
    background: var(--glass);
    border-bottom-color: var(--border);
    backdrop-filter: blur(16px) saturate(1.4);
  }
  .inner {
    display: flex;
    align-items: center;
    gap: 24px;
    width: min(var(--content), 100% - 48px);
    height: 100%;
    margin-inline: auto;
  }
  .overlay .inner {
    width: calc(100% - 48px);
  }
  .brand {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    color: var(--text);
    font-weight: 650;
    font-size: 17px;
    letter-spacing: -0.02em;
  }
  nav {
    display: flex;
    gap: 4px;
    margin-left: 12px;
  }
  .link {
    padding: 7px 12px;
    white-space: nowrap;
    border-radius: 9px;
    color: var(--text-2);
    font-size: 14.5px;
    transition: color 0.2s, background 0.2s;
  }
  .link:hover {
    color: var(--text);
    background: var(--surface);
  }
  .link[aria-current="page"] {
    color: var(--text);
    background: var(--surface);
    box-shadow: inset 0 0 0 1px var(--border);
  }
  .right {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-left: auto;
  }
  .icon {
    display: grid;
    place-items: center;
    width: 36px;
    height: 36px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--text-2);
    cursor: pointer;
    transition: color 0.2s, border-color 0.2s;
  }
  .icon:hover {
    color: var(--text);
    border-color: var(--border-strong);
  }
  .icon svg {
    width: 17px;
    height: 17px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
  }
  .cta {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 36px;
    padding: 0 14px;
    border-radius: 10px;
    background: var(--text);
    color: var(--page);
    font-size: 14px;
    font-weight: 600;
    transition: transform 0.2s var(--ease);
  }
  .cta:hover {
    transform: translateY(-1px);
  }
  @media (max-width: 720px) {
    nav {
      margin-left: 0;
    }
    .cta {
      display: none;
    }
    .inner,
    .overlay .inner {
      width: calc(100% - 32px);
      gap: 12px;
    }
  }
  /* Three links, the logo and the theme toggle share a phone's width: the logo drops its name. */
  @media (max-width: 400px) {
    .brand span {
      display: none;
    }
  }
  @media (max-width: 359px) {
    .link {
      padding: 7px 8px;
    }
    .inner,
    .overlay .inner {
      gap: 6px;
    }
  }
</style>
