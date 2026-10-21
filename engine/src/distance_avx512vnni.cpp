// Compiled with -mavx512f -mavx512bw -mavx512vnni. Only called after runtime CPU detection
// (AVX-512 VNNI: Cascade Lake and later Xeons, Ice Lake and later client CPUs).
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "kernels.h"

namespace oryx::kernels {

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

}  // namespace oryx::kernels

#endif
