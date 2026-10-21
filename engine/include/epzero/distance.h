#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace epzero {

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

// Best supported level, detected once per process. Setting EPZERO_SIMD=scalar|avx2|avx512|neon|wasm
// forces a specific (supported) level, which is how the benchmarks measure SIMD speedups.
SimdLevel active_simd_level() noexcept;

// Returns nullptr if `level` is not supported on this machine.
DistanceFn get_distance_fn(Metric metric, SimdLevel level) noexcept;
DistanceFn get_distance_fn(Metric metric) noexcept;

// Asymmetric-distance kernel for 8-bit scalar quantization: returns sum_i w[i] * codes[i].
// The quantizer folds its per-dimension scales and offsets into `w` once per query, so scoring a
// node reads only `dim` bytes. Returns nullptr if `level` is not supported.
using Sq8DotFn = float (*)(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept;
Sq8DotFn get_sq8_dot_fn(SimdLevel level) noexcept;

// The same dot product in integers: int8 weights (|w| <= kSq8IntWeightMax) against the uint8
// codes. The quantizer rounds its folded per-query weights to int8 once per query, and each node
// then costs one integer multiply-add per byte: on CPUs with VNNI one instruction covers 64 (AVX-512)
// or 32 (AVX) dimensions, against about ten instructions per 32 for the float kernel. The bound keeps
// the AVX2 kernel's 16-bit pair sums (2 x 255 x 63) from saturating, so every kernel agrees exactly.
inline constexpr int kSq8IntWeightMax = 63;
using Sq8IntDotFn = std::int32_t (*)(const std::int8_t* w, const std::uint8_t* codes, std::size_t dim) noexcept;
// The fastest integer kernel for `level` on this CPU (VNNI when present), or nullptr where there is
// none faster than the float kernel (NEON, WebAssembly) or `level` is unsupported.
Sq8IntDotFn get_sq8_int_dot_fn(SimdLevel level) noexcept;
// Every integer kernel this CPU can run, by name (for tests and benchmarks).
std::vector<std::pair<std::string_view, Sq8IntDotFn>> sq8_int_dot_variants();

// RaBitQ kernel: sum over j < 4 of 2^j * popcount(bits & planes[j]), with `planes` the four bit
// planes of a 4-bit query (`words` 64-bit words each). See src/rabitq.h.
using RabitqIpFn = std::uint32_t (*)(const std::uint64_t* bits, const std::uint64_t* planes, std::size_t words) noexcept;
RabitqIpFn get_rabitq_ip_fn(SimdLevel level) noexcept;

// Scales `v` to unit length in place. Zero vectors are left unchanged.
void normalize(float* v, std::size_t dim) noexcept;

}  // namespace epzero
