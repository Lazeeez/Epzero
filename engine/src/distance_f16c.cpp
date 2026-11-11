// Compiled with -mavx2 -mfma -mf16c. Only called after runtime CPU detection (AVX2 and F16C).
// The re-rank against full vectors stored as half floats (HnswParams::half_vectors): each 8 halves
// widen to floats in one instruction, so reading half the bytes costs almost nothing in arithmetic.
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "kernels.h"

namespace epzero::kernels {
namespace {

inline float hsum(__m256 v) noexcept {
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
}

inline __m256 load_half(const std::uint16_t* p) noexcept {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

}  // namespace

float l2_half_f16c(const float* a, const std::uint16_t* b, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), load_half(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), load_half(b + i + 8));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  if (i + 8 <= dim) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), load_half(b + i));
    acc0 = _mm256_fmadd_ps(d, d, acc0);
    i += 8;
  }
  float sum = hsum(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) {
    const float d = a[i] - _cvtsh_ss(b[i]);
    sum += d * d;
  }
  return sum;
}

float ip_half_f16c(const float* a, const std::uint16_t* b, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), load_half(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), load_half(b + i + 8), acc1);
  }
  if (i + 8 <= dim) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), load_half(b + i), acc0);
    i += 8;
  }
  float dot = hsum(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) dot += a[i] * _cvtsh_ss(b[i]);
  return 1.0f - dot;
}

}  // namespace epzero::kernels

#endif
