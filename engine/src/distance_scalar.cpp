#include <bit>

#include "kernels.h"

namespace oryx::kernels {

// Deliberately plain loops: without -ffast-math the compiler cannot reorder the float
// reduction, so this stays a true scalar baseline for the SIMD speedup measurements.

float l2_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

float ip_scalar(const float* a, const float* b, std::size_t dim) noexcept {
  float dot = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    dot += a[i] * b[i];
  }
  return 1.0f - dot;
}

float sq8_dot_scalar(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  float dot = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    dot += w[i] * static_cast<float>(codes[i]);
  }
  return dot;
}

std::int32_t sq8_int_dot_scalar(const std::int8_t* w, const std::uint8_t* codes, std::size_t dim) noexcept {
  std::int32_t dot = 0;
  for (std::size_t i = 0; i < dim; ++i) dot += static_cast<std::int32_t>(w[i]) * codes[i];
  return dot;
}

// Portable popcount: one instruction on ARM64 (CNT) and WebAssembly (i64.popcnt); on baseline
// x86-64 a short bit-twiddling sequence (the AVX2 build uses POPCNT, see distance_avx2.cpp).
std::uint32_t rabitq_ip_scalar(const std::uint64_t* bits, const std::uint64_t* planes, std::size_t words) noexcept {
  std::uint32_t sum = 0;
  for (std::uint32_t j = 0; j < 4; ++j) {
    std::uint32_t count = 0;
    for (std::size_t w = 0; w < words; ++w) count += static_cast<std::uint32_t>(std::popcount(bits[w] & planes[j * words + w]));
    sum += count << j;
  }
  return sum;
}

}  // namespace oryx::kernels
