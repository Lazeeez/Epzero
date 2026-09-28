// Compiled with -mavx2 -mfma. Only called after runtime CPU detection.
#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>

#include "kernels.h"

namespace oryx::kernels {
namespace {

inline float hsum(__m256 v) noexcept {
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  return _mm_cvtss_f32(_mm_add_ss(sums, shuf));
}

}  // namespace

// Two independent accumulators hide the 4-cycle FMA latency; 16 floats per iteration.
float l2_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    acc0 = _mm256_fmadd_ps(d0, d0, acc0);
    acc1 = _mm256_fmadd_ps(d1, d1, acc1);
  }
  if (i + 8 <= dim) {
    const __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    acc0 = _mm256_fmadd_ps(d, d, acc0);
    i += 8;
  }
  float sum = hsum(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

// 16 codes per iteration: widen uint8 -> int32 -> float, then FMA with the per-query weights.
float sq8_dot_avx2(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + i));
    const __m256 c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
    const __m256 c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(bytes, 8)));
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), c0, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i + 8), c1, acc1);
  }
  if (i + 8 <= dim) {
    const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i));
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + i), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)), acc0);
    i += 8;
  }
  float dot = hsum(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) dot += w[i] * static_cast<float>(codes[i]);
  return dot;
}

float ip_avx2(const float* a, const float* b, std::size_t dim) noexcept {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  if (i + 8 <= dim) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    i += 8;
  }
  float dot = hsum(_mm256_add_ps(acc0, acc1));
  for (; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return 1.0f - dot;
}

}  // namespace oryx::kernels

#endif
