// What differs between compilers and operating systems, in one place: atomic access to plain
// integers, prefetching, aligned allocation, and x86 CPU feature detection. GCC and Clang keep their
// builtins; MSVC (Windows) gets the equivalents.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_MSC_VER) && !defined(__clang__)
#define EPZERO_MSVC 1
#include <intrin.h>
#include <malloc.h>
#endif
#if !defined(EPZERO_MSVC) && (defined(__x86_64__) || defined(_M_X64))
#include <cpuid.h>
#endif

namespace epzero::platform {

// Acquire loads and release stores of plain integers that threads share without a lock (neighbour
// lists, tombstones). GCC and Clang builtins, as Apple's standard library long lacked atomic_ref.
template <typename T>
inline T load_acquire(const T& slot) noexcept {
#if defined(EPZERO_MSVC)
  return std::atomic_ref<T>(const_cast<T&>(slot)).load(std::memory_order_acquire);
#else
  return __atomic_load_n(&slot, __ATOMIC_ACQUIRE);
#endif
}

template <typename T>
inline void store_release(T& slot, T value) noexcept {
#if defined(EPZERO_MSVC)
  std::atomic_ref<T>(slot).store(value, std::memory_order_release);
#else
  __atomic_store_n(&slot, value, __ATOMIC_RELEASE);
#endif
}

inline void prefetch(const void* p) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p, 0, 3);
#elif defined(EPZERO_MSVC) && (defined(_M_X64) || defined(_M_IX86))
  _mm_prefetch(static_cast<const char*>(p), _MM_HINT_T0);
#else
  (void)p;
#endif
}

// `bytes` must be a multiple of `alignment`. Free with aligned_free.
inline void* aligned_alloc(std::size_t alignment, std::size_t bytes) noexcept {
#if defined(EPZERO_MSVC) || defined(_WIN32)
  return _aligned_malloc(bytes, alignment);
#else
  return std::aligned_alloc(alignment, bytes);
#endif
}

inline void aligned_free(void* p) noexcept {
#if defined(EPZERO_MSVC) || defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

#if defined(__x86_64__) || defined(_M_X64)
struct CpuFeatures {
  bool avx2 = false;
  bool fma = false;
  bool f16c = false;
  bool avx512f = false;
  bool avx512bw = false;
  bool avx512vnni = false;
  bool avx512vbmi2 = false;
  bool avx_vnni = false;
};

namespace detail {
inline void cpuid(unsigned leaf, unsigned subleaf, unsigned out[4]) noexcept {
#if defined(EPZERO_MSVC)
  int regs[4];
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  for (int i = 0; i < 4; ++i) out[i] = static_cast<unsigned>(regs[i]);
#else
  if (!__get_cpuid_count(leaf, subleaf, &out[0], &out[1], &out[2], &out[3])) out[0] = out[1] = out[2] = out[3] = 0;
#endif
}

inline CpuFeatures detect() noexcept {
  CpuFeatures f;
#if defined(EPZERO_MSVC)
  // The CPU must have each feature and the OS must save its registers (XCR0): YMM for AVX, and the
  // opmask and upper ZMM state for AVX-512. __builtin_cpu_supports checks the same on GCC and Clang.
  unsigned r[4];
  cpuid(0, 0, r);
  const unsigned max_leaf = r[0];
  cpuid(1, 0, r);
  const bool osxsave = (r[2] & (1u << 27)) != 0;
  const unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
  const bool ymm = (xcr0 & 0x6) == 0x6;
  const bool zmm = ymm && (xcr0 & 0xe0) == 0xe0;
  f.fma = ymm && (r[2] & (1u << 12)) != 0;
  f.f16c = ymm && (r[2] & (1u << 29)) != 0;
  if (max_leaf >= 7) {
    cpuid(7, 0, r);
    f.avx2 = ymm && (r[1] & (1u << 5)) != 0;
    f.avx512f = zmm && (r[1] & (1u << 16)) != 0;
    f.avx512bw = zmm && (r[1] & (1u << 30)) != 0;
    f.avx512vbmi2 = zmm && (r[2] & (1u << 6)) != 0;
    f.avx512vnni = zmm && (r[2] & (1u << 11)) != 0;
    cpuid(7, 1, r);
    f.avx_vnni = ymm && (r[0] & (1u << 4)) != 0;
  }
#else
  f.avx2 = __builtin_cpu_supports("avx2");
  f.fma = __builtin_cpu_supports("fma");
  f.f16c = __builtin_cpu_supports("f16c");
  f.avx512f = __builtin_cpu_supports("avx512f");
  f.avx512bw = __builtin_cpu_supports("avx512bw");
  f.avx512vbmi2 = __builtin_cpu_supports("avx512vbmi2");
  unsigned r[4];
  cpuid(7, 0, r);
  f.avx512vnni = (r[2] & (1u << 11)) != 0 && (r[1] & (1u << 30)) != 0;
  cpuid(7, 1, r);
  f.avx_vnni = (r[0] & (1u << 4)) != 0;
#endif
  return f;
}
}  // namespace detail

inline const CpuFeatures& cpu() noexcept {
  static const CpuFeatures features = detail::detect();
  return features;
}

// The CPU's brand string ("" if it has none).
inline std::string cpu_brand() {
  unsigned words[12] = {};
  unsigned r[4];
  detail::cpuid(0x80000000u, 0, r);
  if (r[0] < 0x80000004u) return {};
  for (unsigned leaf = 0; leaf < 3; ++leaf) detail::cpuid(0x80000002u + leaf, 0, &words[4 * leaf]);
  std::string brand(reinterpret_cast<const char*>(words), sizeof(words));
  brand.erase(brand.find('\0') == std::string::npos ? brand.size() : brand.find('\0'));
  return brand;
}
#endif

}  // namespace epzero::platform
