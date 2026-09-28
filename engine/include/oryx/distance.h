#pragma once

#include <cstddef>
#include <string_view>

namespace oryx {

enum class Metric {
  L2,            // squared Euclidean distance
  InnerProduct,  // 1 - dot(a, b)
  Cosine,        // vectors are normalized on insert/query, then scored as InnerProduct
};

enum class SimdLevel { Scalar, AVX2, AVX512, NEON, WasmSimd128 };

// All kernels return a distance where smaller means closer.
using DistanceFn = float (*)(const float* a, const float* b, std::size_t dim) noexcept;

std::string_view to_string(Metric metric) noexcept;
std::string_view to_string(SimdLevel level) noexcept;

// True if the kernel for `level` was compiled in and the running CPU supports it.
bool simd_level_supported(SimdLevel level) noexcept;

// Best supported level, detected once per process. Setting ORYX_SIMD=scalar|avx2|avx512|neon|wasm
// forces a specific (supported) level, which is how the benchmarks measure SIMD speedups.
SimdLevel active_simd_level() noexcept;

// Returns nullptr if `level` is not supported on this machine.
DistanceFn get_distance_fn(Metric metric, SimdLevel level) noexcept;
DistanceFn get_distance_fn(Metric metric) noexcept;

// Scales `v` to unit length in place. Zero vectors are left unchanged.
void normalize(float* v, std::size_t dim) noexcept;

}  // namespace oryx
