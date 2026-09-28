// WebAssembly SIMD128 kernels (compiled with -msimd128). Every browser that runs the demo supports
// fixed-width SIMD, so there is no runtime check; FMA would need relaxed-SIMD, which is not
// universally available, so products and sums are separate instructions.
#if defined(__wasm_simd128__)

#include <wasm_simd128.h>

#include "kernels.h"

namespace oryx::kernels {
namespace {

inline float hsum(v128_t v) noexcept {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) +
         wasm_f32x4_extract_lane(v, 3);
}

}  // namespace

// Four independent accumulators so consecutive adds do not wait on each other.
float l2_wasm(const float* a, const float* b, std::size_t dim) noexcept {
  v128_t acc0 = wasm_f32x4_splat(0.0f);
  v128_t acc1 = wasm_f32x4_splat(0.0f);
  v128_t acc2 = wasm_f32x4_splat(0.0f);
  v128_t acc3 = wasm_f32x4_splat(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const v128_t d0 = wasm_f32x4_sub(wasm_v128_load(a + i), wasm_v128_load(b + i));
    const v128_t d1 = wasm_f32x4_sub(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4));
    const v128_t d2 = wasm_f32x4_sub(wasm_v128_load(a + i + 8), wasm_v128_load(b + i + 8));
    const v128_t d3 = wasm_f32x4_sub(wasm_v128_load(a + i + 12), wasm_v128_load(b + i + 12));
    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(d0, d0));
    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(d1, d1));
    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(d2, d2));
    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(d3, d3));
  }
  for (; i + 4 <= dim; i += 4) {
    const v128_t d = wasm_f32x4_sub(wasm_v128_load(a + i), wasm_v128_load(b + i));
    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(d, d));
  }
  float sum = hsum(wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3)));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float ip_wasm(const float* a, const float* b, std::size_t dim) noexcept {
  v128_t acc0 = wasm_f32x4_splat(0.0f);
  v128_t acc1 = wasm_f32x4_splat(0.0f);
  v128_t acc2 = wasm_f32x4_splat(0.0f);
  v128_t acc3 = wasm_f32x4_splat(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
    acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(a + i + 8), wasm_v128_load(b + i + 8)));
    acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(a + i + 12), wasm_v128_load(b + i + 12)));
  }
  for (; i + 4 <= dim; i += 4) {
    acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
  }
  float dot = hsum(wasm_f32x4_add(wasm_f32x4_add(acc0, acc1), wasm_f32x4_add(acc2, acc3)));
  for (; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return 1.0f - dot;
}

}  // namespace oryx::kernels

#endif
