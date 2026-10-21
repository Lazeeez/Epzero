<script lang="ts">
  // The backdrop of the pages without the 3D graph: two slow aurora glows and a faint dot grid that
  // fades toward the edges. Pure CSS: the glows are gradients (no blur filter) moved by transforms,
  // so the compositor animates them without repainting, and they stop for reduced motion.
</script>
<div class="ambient" aria-hidden="true">
  <i class="glow a"></i>
  <i class="glow b"></i>
  <i class="grid"></i>
</div>

<style>
  .ambient {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
    contain: strict;
  }
  .glow {
    position: absolute;
    width: 70vmax;
    height: 70vmax;
    border-radius: 50%;
    opacity: var(--ambient-strength, 0.5);
    will-change: transform;
  }
  .a {
    top: -38vmax;
    right: -18vmax;
    background: radial-gradient(closest-side, var(--accent-glow), transparent);
    animation: drift-a 38s ease-in-out infinite alternate;
  }
  .b {
    bottom: -44vmax;
    left: -24vmax;
    background: radial-gradient(closest-side, color-mix(in srgb, var(--accent-2) 30%, transparent), transparent);
    animation: drift-b 46s ease-in-out infinite alternate;
  }
  .grid {
    position: absolute;
    inset: 0;
    background-image: radial-gradient(var(--grid-dot) 1px, transparent 1.2px);
    background-size: 28px 28px;
    mask-image: radial-gradient(ellipse 70% 55% at 70% 0%, #000 0%, transparent 70%);
  }
  @keyframes drift-a {
    to {
      transform: translate(-12vmax, 10vmax) scale(1.15);
    }
  }
  @keyframes drift-b {
    to {
      transform: translate(14vmax, -8vmax) scale(0.9);
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .glow {
      animation: none;
    }
  }
</style>
