import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { isDark, token } from "../theme";
import type { GraphInfo, Trace } from "../engine/protocol";

// World layout: UMAP coordinates spread over the XZ plane, one horizontal slab per HNSW layer.
const SPREAD = 42;
const LAYER_GAP = 11;

// Point states, read by the shader to pick color and size.
export const enum State {
  Idle = 0,
  Visited = 1, // scored during search, rejected
  Accepted = 2, // entered the beam
  Result = 3,
  Entry = 4,
}

interface Palette {
  background: THREE.Color;
  idleA: THREE.Color;
  idleB: THREE.Color;
  visited: THREE.Color;
  accepted: THREE.Color;
  result: THREE.Color;
  path: THREE.Color;
  edge: THREE.Color;
  plane: THREE.Color;
  dark: boolean;
}

function readPalette(): Palette {
  const dark = isDark();
  return {
    background: new THREE.Color(token("--page")),
    // Idle nodes: a quiet two-tone gradient across the map, so clusters read without shouting.
    idleA: new THREE.Color(dark ? "#4a6694" : "#5d7fb8"),
    idleB: new THREE.Color(dark ? "#6a58a6" : "#8a74c4"),
    visited: new THREE.Color(dark ? "#5a6a86" : "#8a93a3"),
    accepted: new THREE.Color(token("--accent")),
    result: new THREE.Color(dark ? "#ff8a4c" : "#eb6834"),
    // The search path must stand out against the lanes: near-white on dark, deep blue on light.
    path: new THREE.Color(dark ? "#b9d4ff" : "#184f95"),
    edge: new THREE.Color(token("--accent")),
    plane: new THREE.Color(dark ? "#9fb7ff" : "#2a78d6"),
    dark,
  };
}

const POINT_VERTEX = /* glsl */ `
  attribute float aState;
  attribute float aTone;
  uniform float uPixelRatio;
  uniform float uTime;
  uniform vec3 uIdleA, uIdleB, uVisited, uAccepted, uResult;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float size = 2.2;
    vec3 color = mix(uIdleA, uIdleB, aTone);
    float alpha = 0.78;
    if (aState > 3.5) { size = 10.0; color = uAccepted; alpha = 1.0; }
    else if (aState > 2.5) { size = 7.5 + 1.5 * sin(uTime * 4.0); color = uResult; alpha = 1.0; }
    else if (aState > 1.5) { size = 4.2; color = uAccepted; alpha = 1.0; }
    else if (aState > 0.5) { size = 3.0; color = uVisited; alpha = 0.95; }
    vColor = color;
    vAlpha = alpha;
    gl_PointSize = size * uPixelRatio * (48.0 / -mv.z);
  }
`;

const POINT_FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    vec2 p = gl_PointCoord - 0.5;
    float d = length(p);
    if (d > 0.5) discard;
    float edge = smoothstep(0.5, 0.36, d);
    gl_FragColor = vec4(vColor, vAlpha * edge);
  }
`;

export interface Hover {
  id: number;
  x: number;
  y: number;
}

export class GraphScene {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(42, 1, 0.5, 600);
  private readonly controls: OrbitControls;
  private readonly composer: EffectComposer;
  private readonly bloom: UnrealBloomPass;
  private palette = readPalette();
  private readonly reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  private count = 0;
  private positions: Float32Array<ArrayBufferLike> = new Float32Array(0); // node id -> (x, z) on the map
  private levels: Uint8Array<ArrayBufferLike> = new Uint8Array(0);
  private points: THREE.Points | null = null;
  private states = new Float32Array(0);
  private pointIndex: Int32Array[] = []; // pointIndex[layer][node] -> index into the points buffer
  private pointNode = new Int32Array(0); // index into the points buffer -> node
  private pointerMoved = false;
  private graphGroup = new THREE.Group();
  private planes = new THREE.Group();
  private trail: THREE.LineSegments | null = null;
  private trailTimes = new Float32Array(0);
  private trailCount = 0;
  private stateEvents: { time: number; point: number; state: State }[] = [];
  private builtCount = 0;
  private readonly raycaster = new THREE.Raycaster();
  private readonly pointer = new THREE.Vector2(-9, -9);
  private hoverListener: ((hover: Hover | null) => void) | null = null;
  private clickListener: ((id: number) => void) | null = null;
  private lastHover = -1;
  private cameraGoal: { target: THREE.Vector3; position: THREE.Vector3 } | null = null;

  constructor(private readonly host: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    // Phones and tablets have 3x screens and small GPUs: 1.5x is visually identical for points and
    // lines under bloom, and renders about half the pixels of 2x.
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, matchMedia("(pointer: coarse)").matches ? 1.5 : 2));
    host.appendChild(this.renderer.domElement);
    this.renderer.domElement.setAttribute("aria-label", "3D view of the HNSW graph layers");
    this.renderer.domElement.setAttribute("role", "img");

    this.camera.position.set(78, 62, 104);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 12;
    this.controls.maxDistance = 220;
    this.controls.autoRotate = !this.reducedMotion;
    this.controls.autoRotateSpeed = 0.35;
    this.controls.target.set(0, 12, 0);
    this.controls.addEventListener("start", () => {
      this.controls.autoRotate = false;
      this.cameraGoal = null; // the viewer took over
    });

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.6, 0.5, 0.2);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.scene.add(this.graphGroup, this.planes);
    this.raycaster.params.Points = { threshold: 0.55 };

    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointermove", (event) => {
      const box = canvas.getBoundingClientRect();
      this.pointer.set(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
      this.pointerMoved = true;
    });
    canvas.addEventListener("pointerleave", () => {
      this.pointer.set(-9, -9);
      this.pointerMoved = true;
    });
    let down = { x: 0, y: 0 };
    canvas.addEventListener("pointerdown", (event) => (down = { x: event.clientX, y: event.clientY }));
    canvas.addEventListener("click", (event) => {
      // A drag is camera control, not a pick.
      if (Math.hypot(event.clientX - down.x, event.clientY - down.y) > 5) return;
      if (this.lastHover >= 0) this.clickListener?.(this.lastHover);
    });

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.applyPalette();
    this.renderer.setAnimationLoop((time) => this.frame(time));
  }

  onHover(listener: (hover: Hover | null) => void) {
    this.hoverListener = listener;
  }

  onClick(listener: (id: number) => void) {
    this.clickListener = listener;
  }

  /** Map positions (normalized UMAP, [-1, 1]^2) for every node; shown as a base-layer cloud. */
  setLayout(xy: Float32Array) {
    this.count = xy.length / 2;
    this.positions = xy;
    this.levels = new Uint8Array(this.count);
    this.rebuildPoints();
  }

  /** Reveals nodes as they are inserted during the build (layer 0 only until topology arrives). */
  setBuilt(count: number) {
    this.builtCount = count;
    this.points?.geometry.setDrawRange(0, count);
  }

  setGraph(graph: GraphInfo) {
    this.levels = graph.levels;
    this.builtCount = this.count;
    this.rebuildPoints();
    this.rebuildEdges(graph);
    this.rebuildPlanes(graph.maxLevel);
  }

  themeChanged() {
    this.palette = readPalette();
    this.applyPalette();
    if (this.levels.some((l) => l > 0)) this.rebuildPlanes(Math.max(...this.levels));
  }

  // ---------- Geometry ----------

  private worldX(node: number) {
    return this.positions[node * 2] * SPREAD;
  }
  private worldZ(node: number) {
    return this.positions[node * 2 + 1] * SPREAD;
  }
  private worldPoint(node: number, layer: number) {
    return new THREE.Vector3(this.worldX(node), layer * LAYER_GAP, this.worldZ(node));
  }

  private rebuildPoints() {
    if (this.points) {
      this.graphGroup.remove(this.points);
      this.points.geometry.dispose();
    }
    const maxLevel = this.levels.reduce((m, l) => Math.max(m, l), 0);
    this.pointIndex = Array.from({ length: maxLevel + 1 }, () => new Int32Array(this.count).fill(-1));
    const xyz: number[] = [];
    const tone: number[] = [];
    const owners: number[] = [];
    let index = 0;
    for (let layer = 0; layer <= maxLevel; layer++) {
      for (let node = 0; node < this.count; node++) {
        if (this.levels[node] < layer) continue;
        this.pointIndex[layer][node] = index++;
        owners.push(node);
        xyz.push(this.worldX(node), layer * LAYER_GAP, this.worldZ(node));
        tone.push(0.5 + 0.5 * Math.sin(this.positions[node * 2] * 2.1 + this.positions[node * 2 + 1] * 1.3));
      }
    }
    this.pointNode = Int32Array.from(owners);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
    geometry.setAttribute("aTone", new THREE.Float32BufferAttribute(tone, 1));
    this.states = new Float32Array(index);
    geometry.setAttribute("aState", new THREE.BufferAttribute(this.states, 1).setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, maxLevel === 0 ? this.builtCount : index);
    const material = new THREE.ShaderMaterial({
      vertexShader: POINT_VERTEX,
      fragmentShader: POINT_FRAGMENT,
      transparent: true,
      depthWrite: false,
      uniforms: {
        uPixelRatio: { value: this.renderer.getPixelRatio() },
        uTime: { value: 0 },
        uIdleA: { value: this.palette.idleA },
        uIdleB: { value: this.palette.idleB },
        uVisited: { value: this.palette.visited },
        uAccepted: { value: this.palette.accepted },
        uResult: { value: this.palette.result },
      },
    });
    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    this.graphGroup.add(this.points);
    this.applyPalette();
  }

  private rebuildEdges(graph: GraphInfo) {
    for (const child of [...this.graphGroup.children]) {
      if (child !== this.points) {
        this.graphGroup.remove(child);
        (child as THREE.LineSegments).geometry.dispose();
      }
    }
    const line = (positions: number[], opacity: number, name: string) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      const material = new THREE.LineBasicMaterial({ transparent: true, opacity, depthWrite: false });
      const segments = new THREE.LineSegments(geometry, material);
      segments.name = name;
      segments.frustumCulled = false;
      this.graphGroup.add(segments);
    };
    // Upper layers: the sparse "express lanes" HNSW descends through. Opacity falls with edge
    // count so a dense layer reads as texture and a sparse one as a clear skeleton.
    for (let layer = 1; layer < graph.edges.length; layer++) {
      const edges = graph.edges[layer];
      const lanes: number[] = [];
      for (let i = 0; i < edges.length; i += 2) {
        lanes.push(this.worldX(edges[i]), layer * LAYER_GAP, this.worldZ(edges[i]));
        lanes.push(this.worldX(edges[i + 1]), layer * LAYER_GAP, this.worldZ(edges[i + 1]));
      }
      const opacity = Math.min(0.55, Math.max(0.035, 5 / Math.sqrt(edges.length / 2 + 1)));
      line(lanes, opacity, `lanes-${layer}`);
    }
    // Pillars: each upper-layer node is the same vector on every layer below it.
    const pillars: number[] = [];
    for (let node = 0; node < this.count; node++) {
      if (this.levels[node] === 0) continue;
      pillars.push(this.worldX(node), 0, this.worldZ(node), this.worldX(node), this.levels[node] * LAYER_GAP, this.worldZ(node));
    }
    line(pillars, 0.06, "pillars");

    // Trail: the animated path of a search, preallocated for the largest trace we expect.
    const capacity = 40_000;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(capacity * 6), 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setAttribute("color", new THREE.BufferAttribute(new Float32Array(capacity * 6), 3).setUsage(THREE.DynamicDrawUsage));
    geometry.setDrawRange(0, 0);
    this.trail = new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false }),
    );
    this.trail.frustumCulled = false;
    this.trail.name = "trail";
    this.trailTimes = new Float32Array(capacity);
    this.graphGroup.add(this.trail);
    this.applyPalette();
  }

  private rebuildPlanes(maxLevel: number) {
    for (const child of [...this.planes.children]) {
      this.planes.remove(child);
      if (child instanceof THREE.Mesh || child instanceof THREE.LineLoop || child instanceof THREE.Sprite) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }
    const counts = Array.from({ length: maxLevel + 1 }, (_, layer) => this.levels.filter((l) => l >= layer).length);
    for (let layer = 0; layer <= maxLevel; layer++) {
      // A soft disc per layer: radial fade, brighter for sparser (higher) layers.
      const radius = SPREAD * 1.18;
      const disc = new THREE.Mesh(
        new THREE.CircleGeometry(radius, 96),
        new THREE.ShaderMaterial({
          transparent: true,
          depthWrite: false,
          side: THREE.DoubleSide,
          uniforms: { uColor: { value: this.palette.plane }, uAlpha: { value: this.palette.dark ? 0.05 : 0.06 } },
          vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`,
          fragmentShader: `uniform vec3 uColor; uniform float uAlpha; varying vec2 vUv;
            void main(){ float d = length(vUv - 0.5) * 2.0; gl_FragColor = vec4(uColor, uAlpha * (1.0 - smoothstep(0.55, 1.0, d))); }`,
        }),
      );
      disc.rotation.x = -Math.PI / 2;
      disc.position.y = layer * LAYER_GAP - 0.05;
      const ring = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(
          Array.from({ length: 128 }, (_, i) => new THREE.Vector3(Math.cos((i / 128) * Math.PI * 2) * radius, 0, Math.sin((i / 128) * Math.PI * 2) * radius)),
        ),
        new THREE.LineBasicMaterial({ color: this.palette.plane, transparent: true, opacity: this.palette.dark ? 0.16 : 0.22 }),
      );
      ring.position.y = layer * LAYER_GAP;
      this.planes.add(disc, ring, this.label(`Layer ${layer} · ${counts[layer].toLocaleString()} nodes`, layer, radius));
    }
  }

  private label(text: string, layer: number, radius: number) {
    const canvas = document.createElement("canvas");
    const scale = 2;
    canvas.width = 420 * scale;
    canvas.height = 56 * scale;
    const ctx = canvas.getContext("2d")!;
    ctx.scale(scale, scale);
    ctx.font = `500 26px ${getComputedStyle(document.body).fontFamily}`;
    ctx.fillStyle = token("--text-2");
    ctx.textBaseline = "middle";
    ctx.fillText(text, 4, 28);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false }));
    sprite.scale.set(12, 1.6, 1);
    sprite.position.set(radius * 0.72, layer * LAYER_GAP + 1.4, radius * 0.72);
    sprite.center.set(0, 0.5);
    return sprite;
  }

  private applyPalette() {
    const p = this.palette;
    this.scene.background = p.background;
    this.bloom.enabled = p.dark && !this.reducedMotion;
    const blending = p.dark ? THREE.AdditiveBlending : THREE.NormalBlending;
    if (this.points) {
      const material = this.points.material as THREE.ShaderMaterial;
      Object.assign(material.uniforms.uIdleA, { value: p.idleA });
      Object.assign(material.uniforms.uIdleB, { value: p.idleB });
      Object.assign(material.uniforms.uVisited, { value: p.visited });
      Object.assign(material.uniforms.uAccepted, { value: p.accepted });
      Object.assign(material.uniforms.uResult, { value: p.result });
      material.blending = blending;
      material.needsUpdate = true;
    }
    for (const child of this.graphGroup.children) {
      if (child instanceof THREE.LineSegments) {
        const material = child.material as THREE.LineBasicMaterial;
        if (child.name !== "trail") material.color = p.edge;
        // Only the search path glows; thousands of overlapping lane lines would add up to a blob.
        material.blending = child.name === "trail" ? blending : THREE.NormalBlending;
        material.needsUpdate = true;
      }
    }
  }

  // ---------- Search replay (a seekable timeline) ----------

  /**
   * Loads a search trace onto the timeline and starts playing it. Upper layers play hop by hop
   * (greedy descent); the base layer plays as a spreading beam, scaled to a fixed duration so small
   * and large ef both read well. Every event has a timestamp, so the replay can be paused,
   * scrubbed and stepped (the debugger), not only watched.
   */
  play(trace: Trace, options: { durationScale?: number; flyTo?: boolean } = {}) {
    if (!this.trail) return;
    const durationScale = options.durationScale ?? 1;
    const n = trace.to.length;
    const positions = this.trail.geometry.getAttribute("position") as THREE.BufferAttribute;
    const colors = this.trail.geometry.getAttribute("color") as THREE.BufferAttribute;
    const pos = positions.array as Float32Array;
    const col = colors.array as Float32Array;
    const bright = this.palette.path;
    const dim = this.palette.visited.clone().multiplyScalar(this.palette.dark ? 0.55 : 1);
    const drop = this.palette.result;
    const upperStep = 70 * durationScale;
    const baseEvents = trace.layer.filter((l) => l === 0).length;
    const baseStep = Math.max(0.6, (2600 * durationScale) / Math.max(baseEvents, 1));

    this.stateEvents = [];
    this.eventTimes = new Float32Array(n);
    let t = 0;
    let segment = 0;
    const addSegment = (a: THREE.Vector3, b: THREE.Vector3, color: THREE.Color, time: number) => {
      if (segment >= this.trailTimes.length) return;
      pos.set([a.x, a.y, a.z, b.x, b.y, b.z], segment * 6);
      col.set([color.r, color.g, color.b, color.r, color.g, color.b], segment * 6);
      this.trailTimes[segment++] = time;
    };
    const pointAt = (node: number, layer: number) => this.pointIndex[layer]?.[node] ?? -1;

    const entryPoint = pointAt(trace.entryLabel, trace.entryLayer);
    if (entryPoint >= 0) this.stateEvents.push({ time: 0, point: entryPoint, state: State.Entry });
    let previousLayer = trace.entryLayer;
    for (let i = 0; i < n; i++) {
      const layer = trace.layer[i];
      const from = trace.from[i];
      const to = trace.to[i];
      if (layer !== previousLayer) {
        // Descend: the same node, one layer down.
        for (let l = previousLayer; l > layer; l--) {
          addSegment(this.worldPoint(from, l), this.worldPoint(from, l - 1), drop, t);
          const p = pointAt(from, l - 1);
          if (p >= 0) this.stateEvents.push({ time: t, point: p, state: State.Accepted });
        }
        previousLayer = layer;
      }
      this.eventTimes[i] = t;
      const accepted = trace.accepted[i] === 1;
      addSegment(this.worldPoint(from, layer), this.worldPoint(to, layer), accepted ? bright : dim, t);
      const p = pointAt(to, layer);
      if (p >= 0) this.stateEvents.push({ time: t, point: p, state: accepted ? State.Accepted : State.Visited });
      t += layer > 0 ? upperStep : baseStep;
    }
    for (let r = 0; r < trace.results.length; r++) {
      const p = pointAt(trace.results[r], 0);
      if (p >= 0) this.stateEvents.push({ time: t + 150 + r * 40, point: p, state: State.Result });
    }
    this.trace = trace;
    this.trailCount = segment;
    this.duration = t + 150 + trace.results.length * 40;
    positions.needsUpdate = true;
    colors.needsUpdate = true;
    this.resetPlayback();
    this.playing = true;
    if (this.reducedMotion) this.playhead = this.duration; // show the finished search at once

    if (options.flyTo ?? true) {
      // Fly toward the result neighborhood, keeping the current viewing angle, so the base-layer
      // beam and the results are framed up close while the upper layers stay in view above.
      const centroid = new THREE.Vector3();
      for (const id of trace.results) centroid.add(this.worldPoint(id, 0));
      centroid.divideScalar(Math.max(trace.results.length, 1)).setY(LAYER_GAP * 0.8);
      const direction = this.camera.position.clone().sub(this.controls.target).normalize();
      this.cameraGoal = { target: centroid, position: centroid.clone().addScaledVector(direction, 88 * this.fitScale()) };
      this.controls.autoRotate = false;
    }
  }

  /** Clears the current search from the scene. */
  clearTrace() {
    this.trace = null;
    this.trailCount = 0;
    this.stateEvents = [];
    this.duration = 0;
    this.resetPlayback();
    this.trail?.geometry.setDrawRange(0, 0);
  }

  pause() {
    this.playing = false;
    this.emitPlayhead();
  }

  resume() {
    if (this.playhead >= this.duration) this.seekTime(0);
    this.playing = true;
  }

  setSpeed(speed: number) {
    this.speed = speed;
  }

  /** Moves the timeline so that trace event `index` has just happened (and pauses). */
  seekEvent(index: number) {
    if (!this.trace || this.eventTimes.length === 0) return;
    const i = Math.max(-1, Math.min(index, this.eventTimes.length - 1));
    this.playing = false;
    this.seekTime(i < 0 ? 0 : this.eventTimes[i] + 0.001);
    this.focusEvent(i);
  }

  /** Called with (current event index, total events, playing) whenever the playhead moves. */
  onPlayhead(listener: ((event: number, total: number, playing: boolean) => void) | null) {
    this.playheadListener = listener;
  }

  private trace: Trace | null = null;
  private eventTimes = new Float32Array(0);
  private duration = 0;
  private playhead = 0; // ms along the timeline
  private playing = false;
  private speed = 1;
  private applied = 0; // playhead value the scene currently reflects
  private stateCursor = 0;
  private lastFrame = 0;
  private lastEmitted = -2;
  private playheadListener: ((event: number, total: number, playing: boolean) => void) | null = null;
  private focus: THREE.Mesh | null = null;

  private resetPlayback() {
    this.states.fill(State.Idle);
    this.stateCursor = 0;
    this.playhead = 0;
    this.applied = -1;
    this.lastEmitted = -2;
    if (this.points) (this.points.geometry.getAttribute("aState") as THREE.BufferAttribute).needsUpdate = true;
    if (this.focus) this.focus.visible = false;
  }

  private seekTime(time: number) {
    if (time < this.applied) {
      // Going backwards: rebuild node states from the start (a few thousand events at most).
      this.states.fill(State.Idle);
      this.stateCursor = 0;
    }
    this.playhead = Math.min(time, this.duration);
    this.applyPlayhead();
  }

  private applyPlayhead() {
    const t = this.playhead;
    if (this.trail) {
      // Binary search the last trail segment whose time has come.
      let lo = 0;
      let hi = this.trailCount;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (this.trailTimes[mid] <= t) lo = mid + 1;
        else hi = mid;
      }
      this.trail.geometry.setDrawRange(0, lo * 2);
    }
    if (this.points) {
      let changed = false;
      while (this.stateCursor < this.stateEvents.length && this.stateEvents[this.stateCursor].time <= t) {
        const e = this.stateEvents[this.stateCursor++];
        if (this.states[e.point] < State.Result || e.state >= State.Result) this.states[e.point] = Math.max(this.states[e.point], e.state);
        changed = true;
      }
      if (changed) (this.points.geometry.getAttribute("aState") as THREE.BufferAttribute).needsUpdate = true;
    }
    this.applied = t;
    this.emitPlayhead();
  }

  private currentEvent(): number {
    let lo = 0;
    let hi = this.eventTimes.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.eventTimes[mid] <= this.playhead) lo = mid + 1;
      else hi = mid;
    }
    return lo - 1;
  }

  private emitPlayhead() {
    if (!this.playheadListener || !this.trace) return;
    const event = this.currentEvent();
    const key = event * 2 + (this.playing ? 1 : 0);
    if (key === this.lastEmitted) return;
    this.lastEmitted = key;
    this.playheadListener(event, this.eventTimes.length, this.playing);
  }

  // A ring around the node the paused debugger is looking at.
  private focusEvent(index: number) {
    if (!this.trace || index < 0) {
      if (this.focus) this.focus.visible = false;
      return;
    }
    if (!this.focus) {
      this.focus = new THREE.Mesh(
        new THREE.RingGeometry(1.1, 1.45, 48),
        new THREE.MeshBasicMaterial({ color: this.palette.path, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthTest: false }),
      );
      this.focus.rotation.x = -Math.PI / 2;
      this.focus.renderOrder = 10;
      this.scene.add(this.focus);
    }
    (this.focus.material as THREE.MeshBasicMaterial).color = this.palette.path;
    const p = this.worldPoint(this.trace.to[index], this.trace.layer[index]);
    this.focus.position.copy(p).setY(p.y + 0.05);
    this.focus.visible = true;
  }

  // ---------- Modes ----------

  private active = true;
  private interactive = true;

  /** Stops rendering entirely (no GPU or battery use) while no page shows the scene. */
  setActive(active: boolean) {
    if (active === this.active) return;
    this.active = active;
    this.lastFrame = 0;
    this.renderer.setAnimationLoop(active ? (time) => this.frame(time) : null);
  }

  /**
   * Interactive: orbit/zoom with the mouse and pick nodes. Non-interactive (the landing hero): the
   * canvas ignores the pointer so the page scrolls normally, and the camera drifts on its own.
   */
  setInteractive(interactive: boolean) {
    this.interactive = interactive;
    this.controls.enabled = interactive;
    this.renderer.domElement.style.pointerEvents = interactive ? "auto" : "none";
    this.controls.autoRotate = !this.reducedMotion;
    this.controls.autoRotateSpeed = interactive ? 0.35 : 0.55;
  }

  /** Eases the camera back to the default overview of all layers. */
  overview() {
    this.cameraGoal = {
      target: new THREE.Vector3(0, 12, 0),
      position: new THREE.Vector3(78, 62, 104).multiplyScalar(this.fitScale()),
    };
    this.controls.autoRotate = !this.reducedMotion;
  }

  private frame(time: number) {
    const dt = this.lastFrame ? Math.min(time - this.lastFrame, 100) : 0;
    this.lastFrame = time;
    if (this.playing && this.trace) {
      this.playhead = Math.min(this.playhead + dt * this.speed, this.duration);
      if (this.playhead >= this.duration) this.playing = false;
    }
    if (this.playhead !== this.applied) this.applyPlayhead();
    else if (!this.playing) this.emitPlayhead();
    if (this.points) (this.points.material as THREE.ShaderMaterial).uniforms.uTime.value = time / 1000;
    if (this.cameraGoal) {
      const ease = this.reducedMotion ? 1 : 0.035;
      this.controls.target.lerp(this.cameraGoal.target, ease);
      this.camera.position.lerp(this.cameraGoal.position, ease);
      if (this.camera.position.distanceTo(this.cameraGoal.position) < 0.05) this.cameraGoal = null;
    }
    this.controls.update();
    if (this.interactive) this.pick();
    this.composer.render();
  }

  private pick() {
    // Raycasting ~35k points is cheap but not free: only do it when the pointer moved.
    if (!this.points || !this.hoverListener || !this.pointerMoved) return;
    this.pointerMoved = false;
    let node = -1;
    let anchor: THREE.Vector3 | null = null;
    if (this.pointer.x > -1.5) {
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hit = this.raycaster.intersectObject(this.points)[0];
      if (hit?.index !== undefined && hit.index < this.builtCount + this.pointNode.length - this.count) {
        node = this.pointNode[hit.index];
        const position = this.points.geometry.getAttribute("position");
        anchor = new THREE.Vector3().fromBufferAttribute(position, hit.index);
      }
    }
    if (node === this.lastHover) return;
    this.lastHover = node;
    this.renderer.domElement.style.cursor = node >= 0 ? "pointer" : "";
    if (!anchor) {
      this.hoverListener(null);
      return;
    }
    const screen = anchor.project(this.camera);
    const box = this.renderer.domElement.getBoundingClientRect();
    this.hoverListener({ id: node, x: ((screen.x + 1) / 2) * box.width, y: ((1 - screen.y) / 2) * box.height });
  }

  private inset = { left: 0, bottom: 0 };
  private fitted = false;

  /** Pixels covered by UI panels; the view is shifted to center the graph in the free area. */
  setInsets(left: number, bottom: number) {
    this.inset = { left, bottom };
    this.fitted = false; // refit the initial framing once the panel size is known
    this.resize();
  }

  /** Portrait viewports see less horizontally, so pull the camera back to keep the graph in frame. */
  private fitScale() {
    const { clientWidth: w, clientHeight: h } = this.host;
    const visibleHeight = Math.max(h - this.inset.bottom, 1);
    return Math.max(1, 1.25 / Math.max((w - this.inset.left) / visibleHeight, 0.1));
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.host;
    if (w === 0 || h === 0) return;
    if (!this.cameraGoal && !this.fitted && this.trailCount === 0) {
      this.camera.position.set(78, 62, 104).multiplyScalar(this.fitScale());
      this.fitted = w > 0;
    }
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.bloom.setSize(w, h);
    this.camera.aspect = w / h;
    const { left, bottom } = this.inset;
    if (left > 0 || bottom > 0) this.camera.setViewOffset(w, h, -left / 2, bottom / 2, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }
}
