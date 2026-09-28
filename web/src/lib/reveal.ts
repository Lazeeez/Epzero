// Scroll-reveal action: fades an element in the first time it enters the viewport. One shared
// observer for the whole page; elements are unobserved as soon as they have been revealed.
let observer: IntersectionObserver | null = null;

function shared(): IntersectionObserver {
  observer ??= new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("revealed");
        observer!.unobserve(entry.target);
      }
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.12 },
  );
  return observer;
}

/** `use:reveal` or `use:reveal={0.1}` (a delay in seconds, for staggering siblings). */
export function reveal(node: HTMLElement, delay = 0) {
  node.dataset.reveal = "";
  if (delay) node.style.setProperty("--reveal-delay", `${delay}s`);
  shared().observe(node);
  return {
    destroy() {
      observer?.unobserve(node);
    },
  };
}
