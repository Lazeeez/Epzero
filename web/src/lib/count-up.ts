// Count-up action for static headline numbers: when one first scrolls into view, its digits run up
// from zero to the value written in the element ("1.13×", "−6%", "4,049"), keeping the format
// (decimals, thousands separators, units), and the exact original text is put back at the end.
// Assistive technology reads the final value throughout (aria-label). Skipped for reduced motion.
const reduced = typeof matchMedia !== "undefined" && matchMedia("(prefers-reduced-motion: reduce)").matches;
const NUMBER = /\d[\d,]*(?:\.\d+)?/g;

let observer: IntersectionObserver | null = null;
const starts = new WeakMap<Element, () => void>();

function shared() {
  observer ??= new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        observer!.unobserve(entry.target);
        starts.get(entry.target)?.();
      }
    },
    { threshold: 0.6 },
  );
  return observer;
}

function scaled(text: string, t: number): string {
  return text.replace(NUMBER, (match) => {
    const decimals = match.includes(".") ? match.split(".")[1].length : 0;
    const value = Number(match.replace(/,/g, "")) * t;
    return match.includes(",")
      ? value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
      : value.toFixed(decimals);
  });
}

function textNodes(node: Node): Text[] {
  const out: Text[] = [];
  const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) out.push(walker.currentNode as Text);
  return out;
}

export function countUp(node: HTMLElement, duration = 1100) {
  if (reduced || typeof IntersectionObserver === "undefined") return;
  let frame = 0;
  let restore = () => {};
  starts.set(node, () => {
    const nodes = textNodes(node).filter((t) => /\d/.test(t.data));
    if (!nodes.length) return;
    const finals = nodes.map((t) => t.data);
    node.setAttribute("aria-label", node.textContent ?? "");
    restore = () => {
      nodes.forEach((t, i) => (t.data = finals[i]));
      node.removeAttribute("aria-label");
    };
    const begin = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - begin) / duration);
      if (t >= 1) return restore();
      const eased = 1 - Math.pow(1 - t, 4);
      nodes.forEach((n, i) => (n.data = scaled(finals[i], eased)));
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
  });
  shared().observe(node);
  return {
    destroy() {
      cancelAnimationFrame(frame);
      restore();
      observer?.unobserve(node);
    },
  };
}
