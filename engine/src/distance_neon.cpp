// NEON is mandatory on AArch64 (e.g. Oracle Ampere A1, Apple Silicon), so no runtime check.
#if defined(__aarch64__)

#include <arm_neon.h>

#include "kernels.h"

namespace oryx::kernels {

float l2_neon(const float* a, const float* b, std::size_t dim) noexcept {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
    acc2 = vfmaq_f32(acc2, d2, d2);
    acc3 = vfmaq_f32(acc3, d3, d3);
  }
  for (; i + 4 <= dim; i += 4) {
    const float32x4_t d = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    acc0 = vfmaq_f32(acc0, d, d);
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float sq8_dot_neon(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const uint8x16_t bytes = vld1q_u8(codes + i);
    const uint16x8_t lo = vmovl_u8(vget_low_u8(bytes));
    const uint16x8_t hi = vmovl_u8(vget_high_u8(bytes));
    acc0 = vfmaq_f32(acc0, vld1q_f32(w + i), vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))));
    acc1 = vfmaq_f32(acc1, vld1q_f32(w + i + 4), vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))));
    acc2 = vfmaq_f32(acc2, vld1q_f32(w + i + 8), vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))));
    acc3 = vfmaq_f32(acc3, vld1q_f32(w + i + 12), vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))));
  }
  float dot = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; i < dim; ++i) dot += w[i] * static_cast<float>(codes[i]);
  return dot;
}

float ip_neon(const float* a, const float* b, std::size_t dim) noexcept {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= dim; i += 4) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
  }
  float dot = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
  for (; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return 1.0f - dot;
}

}  // namespace oryx::kernels

#endif
