// Compiled with -mavx2 -mavxvnni. Only called after runtime CPU detection (AVX-VNNI: Alder Lake and
// later client CPUs, Sapphire Rapids and later servers).
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

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
  const __m256i sum = _mm256_add_epi32(acc0, acc1);
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  std::int32_t dot = _mm_cvtsi128_si32(s);
  for (; i < dim; ++i) dot += static_cast<std::int32_t>(w[i]) * codes[i];
  return dot;
}

}  // namespace epzero::kernels

#endif
