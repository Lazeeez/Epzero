// IEEE 754 half precision (binary16) conversions, portable and exact: round to nearest even,
// subnormals, infinities and NaN. For storing the full vectors at half the memory
// (HnswParams::half_vectors); the x86 and ARM kernels convert with F16C / NEON instead.
#pragma once

#include <bit>
#include <cstdint>

namespace epzero::half {

inline float to_float(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  std::uint32_t exponent = (h >> 10) & 0x1fu;
  std::uint32_t mantissa = h & 0x3ffu;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {  // subnormal: normalize
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      bits = sign | (exponent << 23) | ((mantissa & 0x3ffu) << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  return std::bit_cast<float>(bits);
}

inline std::uint16_t from_float(float f) noexcept {
  const std::uint32_t x = std::bit_cast<std::uint32_t>(f);
  const auto sign = static_cast<std::uint16_t>((x >> 16) & 0x8000u);
  const std::uint32_t magnitude = x & 0x7fffffffu;
  if (magnitude >= 0x7f800000u) {  // infinity or NaN (kept a quiet NaN)
    return static_cast<std::uint16_t>(sign | 0x7c00u | (magnitude > 0x7f800000u ? 0x200u : 0u));
  }
  if (magnitude >= 0x477ff000u) return static_cast<std::uint16_t>(sign | 0x7c00u);  // rounds past 65504
  if (magnitude < 0x38800000u) {  // below 2^-14: a subnormal half, or zero
    if (magnitude < 0x33000000u) return sign;  // below 2^-25: rounds to zero
    const std::uint32_t exponent = magnitude >> 23;
    const std::uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
    const std::uint32_t shift = 126 - exponent;
    std::uint32_t n = mantissa >> shift;
    const std::uint32_t rest = mantissa & ((1u << shift) - 1u);
    const std::uint32_t halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (n & 1u) != 0)) ++n;
    return static_cast<std::uint16_t>(sign | n);
  }
  std::uint32_t h = (magnitude - 0x38000000u) >> 13;  // rebias the exponent, keep 10 mantissa bits
  const std::uint32_t rest = magnitude & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (h & 1u) != 0)) ++h;  // a carry may reach infinity: correct
  return static_cast<std::uint16_t>(sign | h);
}

}  // namespace epzero::half
