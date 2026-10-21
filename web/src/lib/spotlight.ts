// Spotlight action: a soft glow that follows the pointer across a card, and lights its border
// where the pointer is nearest. Only for fine pointers (a mouse or trackpad); the glow is drawn by
// CSS (.spot in global.css) from two custom properties, so a move costs one style write.
const fine = typeof matchMedia !== "undefined" && matchMedia("(hover: hover) and (pointer: fine)").matches;

export function spotlight(node: HTMLElement) {
  node.classList.add("spot");
  if (!fine) return;
  let frame = 0;
  let x = 0;
  let y = 0;
  const move = (event: PointerEvent) => {
    x = event.clientX;
    y = event.clientY;
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      const box = node.getBoundingClientRect();
      node.style.setProperty("--mx", `${x - box.left}px`);
      node.style.setProperty("--my", `${y - box.top}px`);
    });
  };
  node.addEventListener("pointermove", move, { passive: true });
  return {
    destroy() {
      cancelAnimationFrame(frame);
      node.removeEventListener("pointermove", move);
    },
  };
}
