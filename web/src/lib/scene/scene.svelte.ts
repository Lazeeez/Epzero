// The one 3D scene the whole site shares. It lives in a fixed layer behind the pages: the landing
// hero shows it as a cinematic backdrop, the playground makes it interactive, and pages that don't
// use it switch it off entirely. It follows the engine: nodes appear as the graph is built.
import { engine } from "../engine/engine.svelte";
import { onThemeChange } from "../theme";
import type { GraphScene, Hover } from "./graph-scene";

export type SceneMode = "hidden" | "hero" | "explore";

class SceneController {
  mode = $state<SceneMode>("hidden");
  hover = $state<Hover | null>(null);
  /** Debugger playhead: event index into the current trace, total events, whether it's playing. */
  playhead = $state({ event: -1, total: 0, playing: false });
  ready = $state(false);
  /** False while the page has scrolled the scene out of sight: rendering pauses. */
  private visible = true;

  scene: GraphScene | null = null;
  private clickHandler: ((id: number) => void) | null = null;
  private laidOut = false;
  private graphShown = false;
  private creating: Promise<void> | null = null;

  /** Creates the WebGL scene in `host` (lazily loads three.js). */
  attach(host: HTMLElement): Promise<void> {
    this.creating ??= (async () => {
      const { GraphScene } = await import("./graph-scene");
      const scene = new GraphScene(host);
      scene.onHover((hover) => (this.hover = hover));
      scene.onClick((id) => this.clickHandler?.(id));
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
    if (this.laidOut && !this.graphShown) {
      if (engine.graph) {
        scene.setGraph(engine.graph);
        this.graphShown = true;
      } else {
        scene.setBuilt(engine.built);
      }
    }
  }

  private apply() {
    const scene = this.scene;
    if (!scene) return;
    scene.setActive(this.mode !== "hidden" && (this.visible || this.mode !== "hero"));
    scene.setInteractive(this.mode === "explore");
    if (this.mode !== "explore") this.hover = null;
  }
}

export const sceneController = new SceneController();
