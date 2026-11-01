// Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512vnni. Only called after runtime CPU detection
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

// The same instruction on 256-bit registers, for the first AVX-512 Xeons: Cascade Lake has VNNI only
// in its EVEX encoding, so without this it would fall back to the AVX2 kernel's three instructions
// per 32 bytes and a scalar tail. These registers avoid the 512-bit kernel's trouble with short
// vectors; the tail is a masked load.
std::int32_t sq8_int_dot_avx512vnni_256(const std::int8_t* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  std::size_t i = 0;
  const auto load = [](const void* p) { return _mm256_loadu_si256(static_cast<const __m256i*>(p)); };
  for (; i + 64 <= dim; i += 64) {
    acc0 = _mm256_dpbusd_epi32(acc0, load(codes + i), load(w + i));
    acc1 = _mm256_dpbusd_epi32(acc1, load(codes + i + 32), load(w + i + 32));
  }
  if (i + 32 <= dim) {
    acc0 = _mm256_dpbusd_epi32(acc0, load(codes + i), load(w + i));
    i += 32;
  }
  if (i < dim) {
    const auto m = static_cast<__mmask32>((std::uint64_t{1} << (dim - i)) - 1);
    acc1 = _mm256_dpbusd_epi32(acc1, _mm256_maskz_loadu_epi8(m, codes + i), _mm256_maskz_loadu_epi8(m, w + i));
  }
  const __m256i sum = _mm256_add_epi32(acc0, acc1);
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

}  // namespace epzero::kernels

#endif
