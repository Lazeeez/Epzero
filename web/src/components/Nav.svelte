<script lang="ts">
  import { FREE_LOOK_TOGGLE, freeLook, setFreeLook } from "../lib/free-look.svelte";
  import { tick, untrack } from "svelte";
  import { router, type Route } from "../lib/router.svelte";
  import { isDark, toggleTheme } from "../lib/theme";
  import Logo from "./Logo.svelte";

  const PAGES: [Route, string][] = [
    ["try", "Try it"],
    ["play", "Play"],
    ["cluster", "Cluster"],
    ["benchmarks", "Benchmarks"],
    ["blog", "Blog"],
  ];

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

  // Phones: the pages live in a menu behind the top-right button.
  let menuOpen = $state(false);
  let menuButton: HTMLButtonElement | undefined = $state();
  let menu: HTMLElement | undefined = $state();
  async function openMenu() {
    menuOpen = true;
    await tick();
    menu?.querySelector<HTMLElement>("a")?.focus();
  }
  function closeMenu(returnFocus = false) {
    menuOpen = false;
    if (returnFocus) menuButton?.focus();
  }
  // A page change closes it (the link was followed), and so does growing past the phone layout.
  $effect(() => {
    void router.route;
    untrack(() => closeMenu());
  });
  $effect(() => {
    if (!menuOpen) return;
    const wide = matchMedia("(min-width: 721px)");
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && closeMenu(true);
    const onWide = () => wide.matches && closeMenu();
    addEventListener("keydown", onKey);
    wide.addEventListener("change", onWide);
    return () => {
      removeEventListener("keydown", onKey);
      wide.removeEventListener("change", onWide);
    };
  });
</script>

<header class="nav" class:scrolled class:overlay={router.route === "try"} class:open={menuOpen}>
  <div class="inner">
    <a class="brand" href={router.href("home")} aria-label="Epzero, home">
      <Logo size={24} />
      <span>Epzero</span>
    </a>
    <nav class="tabs" aria-label="Main" bind:this={links}>
      <span class="pill" class:on={pill.on} style:transform="translateX({pill.x}px)" style:width="{pill.w}px" aria-hidden="true"></span>
      {#each PAGES as [route, label] (route)}
        <a href={router.href(route)} class="link" aria-current={router.route === route ? "page" : undefined}>{label}</a>
      {/each}
    </nav>
    <div class="right">
      {#if router.route === "try" && freeLook.on}
        <!-- Free look hides everything on the playground but this bar, so the way back lives here. -->
        <button
          id={FREE_LOOK_TOGGLE}
          class="look on"
          type="button"
          onclick={() => setFreeLook(false)}
          aria-label="Show the panels"
          title="Show the panels again (H or Esc)"
        >
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16v14H4zM10 5v14" /></svg>
          <span>Show panels</span>
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
      <button
        class="icon menu-button"
        type="button"
        bind:this={menuButton}
        onclick={() => (menuOpen ? closeMenu() : openMenu())}
        aria-label={menuOpen ? "Close the menu" : "Open the menu"}
        aria-expanded={menuOpen}
        aria-controls="site-menu"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path class="bar top" d="M4 7h16" />
          <path class="bar middle" d="M4 12h16" />
          <path class="bar bottom" d="M4 17h16" />
        </svg>
      </button>
    </div>
  </div>
</header>

{#if menuOpen}
  <!-- A tap anywhere outside the menu closes it. -->
  <button class="scrim" type="button" tabindex="-1" aria-hidden="true" onclick={() => closeMenu()}></button>
  <nav id="site-menu" class="menu card" aria-label="Menu" bind:this={menu}>
    {#each PAGES as [route, label] (route)}
      <a href={router.href(route)} aria-current={router.route === route ? "page" : undefined}>
        {label}
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 6l6 6-6 6" /></svg>
      </a>
    {/each}
    {#if router.route !== "try"}
      <a class="menu-cta" href={router.href("try")}>Search photos in your browser <span aria-hidden="true">→</span></a>
    {/if}
  </nav>
{/if}

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
  .nav.overlay,
  .nav.open {
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
  .tabs {
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

  /* The menu button and its menu exist on phones only. */
  .menu-button {
    display: none;
  }
  .menu-button .bar {
    transform-box: fill-box;
    transform-origin: center;
    transition:
      transform 0.3s var(--ease),
      opacity 0.2s;
  }
  .open .menu-button .top {
    transform: translateY(5px) rotate(45deg);
  }
  .open .menu-button .middle {
    opacity: 0;
  }
  .open .menu-button .bottom {
    transform: translateY(-5px) rotate(-45deg);
  }
  .open .menu-button {
    border-color: var(--border-strong);
    color: var(--text);
  }
  .scrim {
    position: fixed;
    inset: 0;
    z-index: 48;
    border: 0;
    background: rgba(0, 0, 0, 0.35);
    animation: fade 0.2s var(--ease);
  }
  .menu {
    position: fixed;
    z-index: 49;
    top: calc(var(--nav-h) + 8px);
    right: max(12px, env(safe-area-inset-right, 0px));
    width: min(300px, calc(100% - 24px));
    display: grid;
    gap: 2px;
    padding: 8px;
    background: var(--surface-solid);
    transform-origin: top right;
    animation: drop 0.25s var(--ease);
  }
  .menu a {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 48px; /* a comfortable tap target */
    padding: 0 14px;
    border-radius: 10px;
    color: var(--text-2);
    font-size: 16px;
    font-weight: 500;
  }
  .menu a:hover,
  .menu a:focus-visible {
    background: var(--surface-2);
    color: var(--text);
  }
  .menu a[aria-current="page"] {
    background: var(--surface-3);
    color: var(--text);
  }
  .menu svg {
    width: 16px;
    height: 16px;
    fill: none;
    stroke: var(--muted);
    stroke-width: 2;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
  .menu a[aria-current="page"] svg {
    stroke: var(--accent-strong);
  }
  .menu .menu-cta {
    justify-content: center;
    gap: 6px;
    margin-top: 6px;
    background: var(--text);
    color: var(--page);
    font-size: 15px;
    font-weight: 600;
  }
  .menu .menu-cta:hover,
  .menu .menu-cta:focus-visible {
    background: var(--text);
    color: var(--page);
  }
  @keyframes drop {
    from {
      opacity: 0;
      transform: translateY(-6px) scale(0.97);
    }
  }
  @keyframes fade {
    from {
      opacity: 0;
    }
  }

  @media (max-width: 720px) {
    .tabs,
    .cta {
      display: none;
    }
    .menu-button {
      display: grid;
    }
    .inner,
    .overlay .inner {
      width: calc(100% - 32px);
      gap: 12px;
    }
    /* Narrow bars: the free-look toggle is just its icon, and while it's on it takes the theme
       toggle's place. */
    .look {
      width: 36px;
      padding: 0;
      justify-content: center;
    }
    .look span {
      display: none;
    }
    .look.on + .icon {
      display: none;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .menu,
    .scrim {
      animation: none;
    }
  }
</style>
