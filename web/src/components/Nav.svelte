<script lang="ts">
  import { FREE_LOOK_TOGGLE, freeLook } from "../lib/free-look.svelte";
  import { untrack } from "svelte";
  import { router } from "../lib/router.svelte";
  import { isDark, toggleTheme } from "../lib/theme";
  import Logo from "./Logo.svelte";

  let scrolled = $state(false);
  let dark = $state(isDark());

  // The active tab's highlight is one pill that glides to the new tab on navigation.
  let links: HTMLElement | undefined = $state();
  let pill = $state({ x: 0, w: 0, on: false });
  $effect(() => {
    void router.route;
    if (!links) return;
    const place = () => {
      const active = links!.querySelector<HTMLElement>('[aria-current="page"]');
      // Without an active tab the pill fades out where it is (read untracked: this effect writes it).
      pill = active ? { x: active.offsetLeft, w: active.offsetWidth, on: true } : { ...untrack(() => pill), on: false };
    };
    place();
    const observer = new ResizeObserver(place); // fonts loading, breakpoints
    observer.observe(links);
    return () => observer.disconnect();
  });

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
    <nav aria-label="Main" bind:this={links}>
      <span class="pill" class:on={pill.on} style:transform="translateX({pill.x}px)" style:width="{pill.w}px" aria-hidden="true"></span>
      <a href={router.href("try")} class="link" aria-current={router.route === "try" ? "page" : undefined}>Try it</a>
      <a href={router.href("play")} class="link" aria-current={router.route === "play" ? "page" : undefined}>Play</a>
      <a href={router.href("cluster")} class="link" aria-current={router.route === "cluster" ? "page" : undefined}>Cluster</a>
      <a href={router.href("benchmarks")} class="link" aria-current={router.route === "benchmarks" ? "page" : undefined}>Benchmarks</a>
    </nav>
    <div class="right">
      {#if router.route === "try"}
        <!-- Free look hides everything on the playground but this bar, so the way back lives here. -->
        <button
          id={FREE_LOOK_TOGGLE}
          class="look"
          class:on={freeLook.on}
          type="button"
          onclick={() => (freeLook.on = !freeLook.on)}
          aria-pressed={freeLook.on}
          aria-label={freeLook.on ? "Show the panels" : "Free look"}
          title={freeLook.on ? "Show the panels again (H or Esc)" : "Free look: hide everything and explore the graph (H)"}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            {#if freeLook.on}<path d="M4 5h16v14H4zM10 5v14" />{:else}<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />{/if}
          </svg>
          <span>{freeLook.on ? "Show panels" : "Free look"}</span>
        </button>
      {/if}
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
    position: relative;
    display: flex;
    gap: 4px;
    margin-left: 12px;
  }
  .link {
    position: relative; /* above the pill */
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
  }
  .link[aria-current="page"]:hover {
    background: none;
  }
  .pill {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    border-radius: 9px;
    background: var(--surface);
    box-shadow: inset 0 0 0 1px var(--border-strong), 0 6px 20px -10px var(--accent-glow);
    opacity: 0;
    transition:
      transform 0.45s var(--ease),
      width 0.45s var(--ease),
      opacity 0.3s;
    pointer-events: none;
  }
  .pill.on {
    opacity: 1;
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
  .look {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 36px;
    padding: 0 12px 0 10px;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--surface);
    color: var(--text-2);
    font: 550 13.5px var(--font);
    white-space: nowrap;
    cursor: pointer;
    transition:
      color 0.2s,
      border-color 0.2s,
      background 0.2s;
  }
  .look:hover {
    color: var(--text);
    border-color: var(--border-strong);
  }
  /* While everything else is hidden, the way back is the one bright thing in the bar. */
  .look.on {
    border-color: var(--accent);
    background: var(--accent-wash);
    color: var(--accent-strong);
  }
  .look svg {
    width: 16px;
    height: 16px;
    fill: none;
    stroke: currentColor;
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
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
  /* Narrow bars: the free-look toggle is just its icon. */
  @media (max-width: 720px) {
    .look {
      width: 36px;
      padding: 0;
      justify-content: center;
    }
    .look span {
      display: none;
    }
  }
  /* Phones have no room for one more button: free look starts from the panel's own Hide button,
     and while it's on, the way back takes the theme toggle's place. */
  @media (max-width: 520px) {
    .look:not(.on),
    .look.on + .icon {
      display: none;
    }
  }
  /* Four links, the logo and the theme toggle share a phone's width: the links tighten and the
     logo drops its name. */
  @media (max-width: 520px) {
    nav {
      gap: 0;
    }
    .link {
      padding: 7px 8px;
      font-size: 13.5px;
    }
  }
  @media (max-width: 400px) {
    .brand span {
      display: none;
    }
  }
  @media (max-width: 359px) {
    .link {
      padding: 7px 5px;
      font-size: 13px;
    }
    .inner,
    .overlay .inner {
      gap: 6px;
    }
  }
</style>
