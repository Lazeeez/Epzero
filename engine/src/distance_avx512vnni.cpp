// Compiled with -mavx512f -mavx512bw -mavx512vnni. Only called after runtime CPU detection
// (AVX-512 VNNI: Cascade Lake and later Xeons, Ice Lake and later client CPUs).
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "kernels.h"

namespace epzero::kernels {

// vpdpbusd on 64 bytes: 64 uint8 codes times 64 int8 weights, summed in fours into int32 lanes.
// The tail is a masked load, so there is no scalar loop.
std::int32_t sq8_int_dot_avx512vnni(const std::int8_t* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  std::size_t i = 0;
  for (; i + 128 <= dim; i += 128) {
    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512(codes + i), _mm512_loadu_si512(w + i));
    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_loadu_si512(codes + i + 64), _mm512_loadu_si512(w + i + 64));
  }
  if (i + 64 <= dim) {
    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_loadu_si512(codes + i), _mm512_loadu_si512(w + i));
    i += 64;
  }
  if (i < dim) {
    const __mmask64 m = (dim - i) >= 64 ? ~__mmask64{0} : ((__mmask64{1} << (dim - i)) - 1);
    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_maskz_loadu_epi8(m, codes + i), _mm512_maskz_loadu_epi8(m, w + i));
  }
  return _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
}

// 4-bit codes (src/sq4.h), 128 per iteration: one 64-byte load is two 64-code blocks, whose low nibbles
// are codes 0-31 and 64-95 and whose high nibbles are 32-63 and 96-127, so each half of the weights is
// assembled from two 32-byte loads. A last 64- or 32-code block puts its low and high nibbles side by
// side, which matches the weights as they lie.
std::int32_t sq4_int_dot_avx512vnni(const std::int8_t* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  const __m512i mask = _mm512_set1_epi8(0x0F);
  __m512i acc0 = _mm512_setzero_si512();
  __m512i acc1 = _mm512_setzero_si512();
  const auto half = [](const std::int8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
  std::size_t i = 0;
  for (; i + 128 <= dim; i += 128, codes += 64) {
    const __m512i v = _mm512_loadu_si512(codes);
    const __m512i wl = _mm512_inserti64x4(_mm512_castsi256_si512(half(w + i)), half(w + i + 64), 1);
    const __m512i wh = _mm512_inserti64x4(_mm512_castsi256_si512(half(w + i + 32)), half(w + i + 96), 1);
    acc0 = _mm512_dpbusd_epi32(acc0, _mm512_and_si512(v, mask), wl);
    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_and_si512(_mm512_srli_epi16(v, 4), mask), wh);
  }
  if (i + 64 <= dim) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codes));
    const __m256i m = _mm256_set1_epi8(0x0F);
    const __m512i c = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(v, m)),
                                         _mm256_and_si256(_mm256_srli_epi16(v, 4), m), 1);
    acc0 = _mm512_dpbusd_epi32(acc0, c, _mm512_loadu_si512(w + i));
    i += 64;
    codes += 32;
  }
  if (i < dim) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
    const __m128i m = _mm_set1_epi8(0x0F);
    const __m256i c = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(v, 4), m), _mm_and_si128(v, m));
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(w + i));
    // Zero-extended (a plain cast leaves the upper half undefined).
    acc1 = _mm512_dpbusd_epi32(acc1, _mm512_inserti64x4(_mm512_setzero_si512(), c, 0),
                               _mm512_inserti64x4(_mm512_setzero_si512(), x, 0));
  }
  return _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
}

}  // namespace epzero::kernels

#endif
