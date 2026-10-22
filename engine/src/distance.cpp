#include "epzero/distance.h"

#include <cmath>
#include <cstdlib>
#include <string_view>

#include "kernels.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

namespace epzero {

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
    case SimdLevel::WasmSimd128: return "wasm";
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
#if defined(__wasm_simd128__)
    case SimdLevel::WasmSimd128: return true;
#endif
    default: return false;
  }
}

namespace {

// Intel's first AVX-512 cores (Skylake-SP, Cascade Lake), as against Ice Lake and later and AMD Zen 4:
// the newer ones are also the ones with VBMI2. On the older ones the AVX-512 integer SQ8 kernel is
// slow on short vectors (see get_sq8_int_dot_fn).
bool avx512_runs_at_full_clock() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  return __builtin_cpu_supports("avx512vbmi2");
#else
  return false;
#endif
}

SimdLevel detect_simd_level() noexcept {
  if (const char* forced = std::getenv("EPZERO_SIMD")) {
    const std::string_view name(forced);
    for (SimdLevel level :
         {SimdLevel::Scalar, SimdLevel::AVX2, SimdLevel::AVX512, SimdLevel::NEON, SimdLevel::WasmSimd128}) {
      if (name == to_string(level) && simd_level_supported(level)) return level;
    }
  }
  for (SimdLevel level : {SimdLevel::AVX512, SimdLevel::AVX2, SimdLevel::NEON, SimdLevel::WasmSimd128}) {
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
#if defined(__wasm_simd128__)
    case SimdLevel::WasmSimd128: return l2 ? kernels::l2_wasm : kernels::ip_wasm;
#endif
    default: return nullptr;
  }
}

DistanceFn get_distance_fn(Metric metric) noexcept {
  return get_distance_fn(metric, active_simd_level());
}

Sq8DotFn get_sq8_dot_fn(SimdLevel level) noexcept {
  if (!simd_level_supported(level)) return nullptr;
  switch (level) {
    case SimdLevel::Scalar: return kernels::sq8_dot_scalar;
#if defined(__x86_64__) || defined(_M_X64)
    case SimdLevel::AVX2: return kernels::sq8_dot_avx2;
    case SimdLevel::AVX512: return kernels::sq8_dot_avx512;
#endif
#if defined(__aarch64__)
    case SimdLevel::NEON: return kernels::sq8_dot_neon;
#endif
#if defined(__wasm_simd128__)
    case SimdLevel::WasmSimd128: return kernels::sq8_dot_wasm;
#endif
    default: return nullptr;
  }
}

namespace {

#if defined(__x86_64__) || defined(_M_X64)
// VNNI is an extension within each SIMD level, so it is checked on its own (CPUID leaf 7).
// The level's own check already covers the OS saving the vector registers.
bool cpu_has_avx_vnni() noexcept {
  unsigned a = 0, b = 0, c = 0, d = 0;
  return __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4)) != 0;
}
bool cpu_has_avx512_vnni() noexcept {
  unsigned a = 0, b = 0, c = 0, d = 0;
  return __get_cpuid_count(7, 0, &a, &b, &c, &d) && (c & (1u << 11)) != 0 && (b & (1u << 30)) != 0;  // + AVX512BW
}
#endif

}  // namespace

std::vector<std::pair<std::string_view, Sq8IntDotFn>> sq8_int_dot_variants() {
  std::vector<std::pair<std::string_view, Sq8IntDotFn>> out = {{"scalar", kernels::sq8_int_dot_scalar}};
#if defined(__x86_64__) || defined(_M_X64)
  if (simd_level_supported(SimdLevel::AVX2)) out.emplace_back("avx2", kernels::sq8_int_dot_avx2);
#if defined(EPZERO_HAVE_AVXVNNI)
  if (simd_level_supported(SimdLevel::AVX2) && cpu_has_avx_vnni()) out.emplace_back("avx-vnni", kernels::sq8_int_dot_avxvnni);
#endif
#if defined(EPZERO_HAVE_AVX512VNNI)
  if (simd_level_supported(SimdLevel::AVX512) && cpu_has_avx512_vnni()) {
    out.emplace_back("avx512-vnni", kernels::sq8_int_dot_avx512vnni);
  }
#endif
#endif
  return out;
}

Sq8IntDotFn get_sq8_int_dot_fn(SimdLevel level, std::size_t dim) noexcept {
  if (!simd_level_supported(level)) return nullptr;
  switch (level) {
    case SimdLevel::Scalar: return kernels::sq8_int_dot_scalar;
#if defined(__x86_64__) || defined(_M_X64)
    case SimdLevel::AVX512:
#if defined(EPZERO_HAVE_AVX512VNNI)
      // On the first AVX-512 Xeons only for whole 64-byte blocks: on a Gold 6226 (Cascade Lake),
      // SQ8 search with the AVX2 integer kernel was 61% faster on GloVe-100 and 73% on DEEP10M (100
      // and 96 dimensions: the AVX-512 kernel's masked tail), while on GIST1M and DBpedia (960 and
      // 1,536) the AVX-512 kernel was 3-5% faster. Float search stays on AVX-512 throughout.
      // EPZERO_SQ8_KERNEL=avx512 forces it (for A/B runs).
      if (cpu_has_avx512_vnni()) {
        const char* forced = std::getenv("EPZERO_SQ8_KERNEL");
        const bool whole_blocks = dim > 0 && dim % 64 == 0;
        if (avx512_runs_at_full_clock() || whole_blocks || (forced != nullptr && std::string_view(forced) == "avx512")) {
          return kernels::sq8_int_dot_avx512vnni;
        }
      }
#endif
      [[fallthrough]];
    case SimdLevel::AVX2:
#if defined(EPZERO_HAVE_AVXVNNI)
      if (cpu_has_avx_vnni()) return kernels::sq8_int_dot_avxvnni;
#endif
      return kernels::sq8_int_dot_avx2;
#endif
    default: return nullptr;  // NEON and WebAssembly: the float kernel is as fast
  }
}

RabitqIpFn get_rabitq_ip_fn(SimdLevel level) noexcept {
  if (!simd_level_supported(level)) return nullptr;
#if defined(__x86_64__) || defined(_M_X64)
  if (level == SimdLevel::AVX2 || level == SimdLevel::AVX512) return kernels::rabitq_ip_popcnt;
#endif
  return kernels::rabitq_ip_scalar;
}

void normalize(float* v, std::size_t dim) noexcept {
  float norm_sq = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) norm_sq += v[i] * v[i];
  if (norm_sq <= 0.0f) return;
  const float inv = 1.0f / std::sqrt(norm_sq);
  for (std::size_t i = 0; i < dim; ++i) v[i] *= inv;
}

}  // namespace epzero
