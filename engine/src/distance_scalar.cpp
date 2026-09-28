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

}  // namespace oryx::kernels
