// Compiled with -mavx2 -mavxvnni. Only called after runtime CPU detection (AVX-VNNI: Alder Lake and
// later client CPUs, Sapphire Rapids and later servers).
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include <cstring>

#include "kernels.h"

namespace epzero::kernels {

// vpdpbusd multiplies 32 uint8 codes by 32 int8 weights and adds each group of four products
// straight into an int32 lane: one instruction per 32 dimensions.
std::int32_t sq8_int_dot_avxvnni(const std::int8_t* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  std::size_t i = 0;
  const auto load = [](const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); };
  for (; i + 64 <= dim; i += 64) {
    acc0 = _mm256_dpbusd_avx_epi32(acc0, load(codes + i), load(w + i));
    acc1 = _mm256_dpbusd_avx_epi32(acc1, load(codes + i + 32), load(w + i + 32));
  }
  if (i + 32 <= dim) {
    acc0 = _mm256_dpbusd_avx_epi32(acc0, load(codes + i), load(w + i));
    i += 32;
  }
  // The tail in 16-, 8- and 4-byte loads, which read nothing past the codes. A 32-byte masked load
  // here lost 31% on 100 dimensions on a Cascade Lake Xeon (it also crosses a cache line), and a scalar
  // loop runs on every node scored.
  __m128i tail = _mm_setzero_si128();
  const auto dot128 = [&](__m128i c, __m128i x) { tail = _mm_dpbusd_avx_epi32(tail, c, x); };
  if (i + 16 <= dim) {
    dot128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i)), _mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
    i += 16;
  }
  if (i + 8 <= dim) {
    dot128(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i)), _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w + i)));
    i += 8;
  }
  if (i + 4 <= dim) {
    std::int32_t c = 0;
    std::int32_t x = 0;
    std::memcpy(&c, codes + i, 4);
    std::memcpy(&x, w + i, 4);
    dot128(_mm_cvtsi32_si128(c), _mm_cvtsi32_si128(x));
    i += 4;
  }
  const __m256i sum = _mm256_add_epi32(acc0, acc1);
  __m128i s = _mm_add_epi32(_mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1)), tail);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  std::int32_t dot = _mm_cvtsi128_si32(s);
  for (; i < dim; ++i) dot += static_cast<std::int32_t>(w[i]) * codes[i];
  return dot;
}

}  // namespace epzero::kernels

#endif
