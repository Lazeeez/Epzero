#include "oryx/distance.h"

#include <cmath>
#include <cstdlib>
#include <string_view>

#include "kernels.h"

namespace oryx {

std::string_view to_string(Metric metric) noexcept {
  switch (metric) {
    case Metric::L2: return "l2";
    case Metric::InnerProduct: return "ip";
    case Metric::Cosine: return "cosine";
  }
  return "unknown";
}

std::string_view to_string(SimdLevel level) noexcept {
  switch (level) {
    case SimdLevel::Scalar: return "scalar";
    case SimdLevel::AVX2: return "avx2";
    case SimdLevel::AVX512: return "avx512";
    case SimdLevel::NEON: return "neon";
  }
  return "unknown";
}

bool simd_level_supported(SimdLevel level) noexcept {
  switch (level) {
    case SimdLevel::Scalar: return true;
#if defined(__x86_64__) || defined(_M_X64)
    case SimdLevel::AVX2: return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    case SimdLevel::AVX512: return __builtin_cpu_supports("avx512f");
#endif
#if defined(__aarch64__)
    case SimdLevel::NEON: return true;
#endif
    default: return false;
  }
}

namespace {

SimdLevel detect_simd_level() noexcept {
  if (const char* forced = std::getenv("ORYX_SIMD")) {
    const std::string_view name(forced);
    for (SimdLevel level : {SimdLevel::Scalar, SimdLevel::AVX2, SimdLevel::AVX512, SimdLevel::NEON}) {
      if (name == to_string(level) && simd_level_supported(level)) return level;
    }
  }
  for (SimdLevel level : {SimdLevel::AVX512, SimdLevel::AVX2, SimdLevel::NEON}) {
    if (simd_level_supported(level)) return level;
  }
  return SimdLevel::Scalar;
}

}  // namespace

SimdLevel active_simd_level() noexcept {
  static const SimdLevel level = detect_simd_level();
  return level;
}

DistanceFn get_distance_fn(Metric metric, SimdLevel level) noexcept {
  if (!simd_level_supported(level)) return nullptr;
  // Cosine vectors are stored normalized, so cosine distance is the inner-product kernel.
  const bool l2 = metric == Metric::L2;
  switch (level) {
    case SimdLevel::Scalar: return l2 ? kernels::l2_scalar : kernels::ip_scalar;
#if defined(__x86_64__) || defined(_M_X64)
    case SimdLevel::AVX2: return l2 ? kernels::l2_avx2 : kernels::ip_avx2;
    case SimdLevel::AVX512: return l2 ? kernels::l2_avx512 : kernels::ip_avx512;
#endif
#if defined(__aarch64__)
    case SimdLevel::NEON: return l2 ? kernels::l2_neon : kernels::ip_neon;
#endif
    default: return nullptr;
  }
}

DistanceFn get_distance_fn(Metric metric) noexcept {
  return get_distance_fn(metric, active_simd_level());
}

void normalize(float* v, std::size_t dim) noexcept {
  float norm_sq = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) norm_sq += v[i] * v[i];
  if (norm_sq <= 0.0f) return;
  const float inv = 1.0f / std::sqrt(norm_sq);
  for (std::size_t i = 0; i < dim; ++i) v[i] *= inv;
}

}  // namespace oryx
