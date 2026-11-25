// NEON is mandatory on AArch64 (e.g. Oracle Ampere A1, Apple Silicon), so no runtime check.
#if defined(__aarch64__)

#include <arm_neon.h>

#include "half.h"
#include "kernels.h"

namespace epzero::kernels {

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

namespace {
inline float32x4_t load_half(const std::uint16_t* p) noexcept { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p))); }
}  // namespace

float l2_half_neon(const float* a, const std::uint16_t* b, std::size_t dim) noexcept {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), load_half(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), load_half(b + i + 4));
    acc0 = vfmaq_f32(acc0, d0, d0);
    acc1 = vfmaq_f32(acc1, d1, d1);
  }
  float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < dim; ++i) {
    const float d = a[i] - half::to_float(b[i]);
    sum += d * d;
  }
  return sum;
}

float ip_half_neon(const float* a, const std::uint16_t* b, std::size_t dim) noexcept {
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), load_half(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), load_half(b + i + 4));
  }
  float dot = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; i < dim; ++i) dot += a[i] * half::to_float(b[i]);
  return 1.0f - dot;
}

// 4-bit codes (src/sq4.h), 64 per iteration: two 16-byte loads give the block's first 32 codes in their low
// nibbles and its last 32 in their high ones. Codes are at most 15, so they are valid int8 and every product
// with an int8 weight fits int16 (15 x 127), which vmull_s8 makes and vpadalq_s16 sums pairwise into int32:
// exact, and with nothing beyond base ARMv8 (no dot-product extension).
std::int32_t sq4_int_dot_neon(const std::int8_t* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  const uint8x16_t mask = vdupq_n_u8(0x0F);
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  const auto mul = [](int32x4_t acc, uint8x16_t c, const std::int8_t* x) {
    const int8x16_t s = vreinterpretq_s8_u8(c);
    const int8x16_t v = vld1q_s8(x);
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(s), vget_low_s8(v)));
    return vpadalq_s16(acc, vmull_high_s8(s, v));
  };
  std::size_t i = 0;
  for (; i + 64 <= dim; i += 64, codes += 32) {
    const uint8x16_t a = vld1q_u8(codes);
    const uint8x16_t b = vld1q_u8(codes + 16);
    acc0 = mul(acc0, vandq_u8(a, mask), w + i);
    acc1 = mul(acc1, vandq_u8(b, mask), w + i + 16);
    acc0 = mul(acc0, vshrq_n_u8(a, 4), w + i + 32);
    acc1 = mul(acc1, vshrq_n_u8(b, 4), w + i + 48);
  }
  if (i < dim) {  // a final block of 32 codes in 16 bytes
    const uint8x16_t a = vld1q_u8(codes);
    acc0 = mul(acc0, vandq_u8(a, mask), w + i);
    acc1 = mul(acc1, vshrq_n_u8(a, 4), w + i + 16);
  }
  return vaddvq_s32(vaddq_s32(acc0, acc1));
}

}  // namespace epzero::kernels

#endif
