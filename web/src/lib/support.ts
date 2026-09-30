// What this browser can run, checked up front so an unsupported browser gets a clear explanation
// instead of a generic failure deep inside the engine.

// The smallest module using a SIMD instruction (the standard feature-detection probe).
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

/** WebAssembly with fixed-width SIMD: the engine is compiled for it (Chrome 91+, Firefox 89+, Safari 16.4+). */
export function wasmSimdSupported(): boolean {
  try {
    return typeof WebAssembly === "object" && WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
}

export const UNSUPPORTED_BROWSER =
  "This browser can't run the engine: it needs WebAssembly SIMD (Chrome or Edge 91+, Firefox 89+, Safari 16.4+). Updating your browser will fix it.";
