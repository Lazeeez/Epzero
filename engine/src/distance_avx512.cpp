// Compiled with -mavx512f. Only called after runtime CPU detection.
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "kernels.h"

namespace oryx::kernels {
namespace {

// Loads the final (dim % 16) floats with a mask, so there is no scalar tail loop.
inline __mmask16 tail_mask(std::size_t remaining) noexcept {
  return static_cast<__mmask16>((1u << remaining) - 1u);
}

}  // namespace

float l2_avx512(const float* a, const float* b, std::size_t dim) noexcept {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    const __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    const __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
    acc0 = _mm512_fmadd_ps(d0, d0, acc0);
    acc1 = _mm512_fmadd_ps(d1, d1, acc1);
  }
  if (i + 16 <= dim) {
    const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    acc0 = _mm512_fmadd_ps(d, d, acc0);
    i += 16;
  }
  if (i < dim) {
    const __mmask16 m = tail_mask(dim - i);
    const __m512 d = _mm512_sub_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i));
    acc1 = _mm512_fmadd_ps(d, d, acc1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

float ip_avx512(const float* a, const float* b, std::size_t dim) noexcept {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= dim; i += 32) {
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
  }
  if (i + 16 <= dim) {
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
    i += 16;
  }
  if (i < dim) {
    const __mmask16 m = tail_mask(dim - i);
    acc1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, a + i), _mm512_maskz_loadu_ps(m, b + i), acc1);
  }
  return 1.0f - _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
}

}  // namespace oryx::kernels

#endif
