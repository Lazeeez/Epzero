// The one 3D scene the whole site shares. It lives in a fixed layer behind the pages: the landing
// hero shows it as a cinematic backdrop, the playground makes it interactive, and pages that don't
// use it switch it off entirely. It follows the engine: nodes appear as the graph is built.
import { engine } from "../engine/engine.svelte";
import { onThemeChange } from "../theme";
import type { GraphInfo } from "../engine/protocol";
import type { GraphScene, Hover, Tap } from "./graph-scene";

export type SceneMode = "hidden" | "hero" | "explore";

class SceneController {
  mode = $state<SceneMode>("hidden");
  hover = $state<Hover | null>(null);
  /** Debugger playhead: event index into the current trace, total events, whether it's playing. */
  playhead = $state({ event: -1, total: 0, playing: false });
  ready = $state(false);
  /** A photo picked by a tap (touch screens have no hover): shown as a card with actions. */
  selected = $state<{ id: number; x: number; y: number } | null>(null);
  autoRotate = $state(true);
  /** -1: all layers lit; otherwise only this layer. */
  layerFocus = $state(-1);
  private clickTimer = 0;
  /** False while the page has scrolled the scene out of sight: rendering pauses. */
  private visible = true;

  scene: GraphScene | null = null;
  private clickHandler: ((id: number) => void) | null = null;
  private laidOut = false;
  /** The graph the scene currently shows; a rebuild replaces it with a new object. */
  private shownGraph: GraphInfo | null = null;
  private creating: Promise<void> | null = null;

  /** Creates the WebGL scene in `host` (lazily loads three.js). */
  attach(host: HTMLElement): Promise<void> {
    this.creating ??= (async () => {
      const { GraphScene } = await import("./graph-scene");
      const scene = new GraphScene(host);
      host.addEventListener("dblclick", () => this.cancelClick(), true); // a double-click zooms, not searches
      scene.onHover((hover) => (this.hover = hover));
      scene.onTap((tap) => this.tap(tap));
      scene.onAutoRotateChange((on) => (this.autoRotate = on));
      scene.onPlayhead((event, total, playing) => (this.playhead = { event, total, playing }));
      onThemeChange(() => scene.themeChanged());
      this.scene = scene;
      this.apply();
      this.ready = true;
    })();
    return this.creating;
  }

  onNodeClick(handler: ((id: number) => void) | null) {
    this.clickHandler = handler;
  }

  // Mouse: a click on a photo searches for similar ones, after a beat so a double-click can zoom
  // instead. Touch: a tap selects the photo and shows its card (search from there); a tap on empty
  // space clears the selection.
  private tap(tap: Tap) {
    clearTimeout(this.clickTimer);
    if (tap.pointerType === "mouse") {
      this.selected = null;
      if (tap.id >= 0) this.clickTimer = window.setTimeout(() => this.clickHandler?.(tap.id), 260);
      return;
    }
    this.selected = tap.id >= 0 ? { id: tap.id, x: tap.x, y: tap.y } : null;
    if (tap.id < 0) this.scene?.clearMarker();
  }

  /** Cancels a pending mouse-click search (called on double-click). */
  cancelClick() {
    clearTimeout(this.clickTimer);
  }

  setAutoRotate(on: boolean) {
    this.autoRotate = on;
    this.scene?.setAutoRotate(on);
  }

  setLayerFocus(layer: number) {
    this.layerFocus = layer;
    this.scene?.setLayerFocus(layer);
  }

  setMode(mode: SceneMode) {
    this.mode = mode;
    this.apply();
  }

  setVisible(visible: boolean) {
    if (visible === this.visible) return;
    this.visible = visible;
    this.apply();
  }

  /** Keeps the scene in step with the engine's dataset, build progress and graph. */
  sync() {
    const scene = this.scene;
    if (!scene) return;
    if (!this.laidOut && engine.layout) {
      scene.setLayout(engine.layout);
      this.laidOut = true;
    }
    if (!this.laidOut) return;
    if (engine.graph && engine.graph !== this.shownGraph) {
      if (this.shownGraph) scene.clearTrace(); // a rebuild: the old search path no longer exists
      scene.setGraph(engine.graph);
      this.shownGraph = engine.graph;
    } else if (!engine.graph) {
      scene.setBuilt(engine.built);
    }
  }

  private apply() {
    const scene = this.scene;
    if (!scene) return;
    scene.setActive(this.mode !== "hidden" && (this.visible || this.mode !== "hero"));
    scene.setInteractive(this.mode === "explore");
    this.autoRotate = true; // setInteractive restarts the drift
    if (this.mode !== "explore") {
      this.hover = null;
      this.selected = null;
      scene.clearMarker();
      if (this.layerFocus !== -1) this.setLayerFocus(-1);
    }
  }
}

export const sceneController = new SceneController();
