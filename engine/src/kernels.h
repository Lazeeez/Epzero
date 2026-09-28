#pragma once

// Per-ISA distance kernels. Each ISA lives in its own translation unit compiled with only
// that ISA's flags, so no AVX instruction can leak into code that runs before dispatch.

#include <cstddef>

namespace oryx::kernels {

float l2_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float ip_scalar(const float* a, const float* b, std::size_t dim) noexcept;

#if defined(__x86_64__) || defined(_M_X64)
float l2_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float ip_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float l2_avx512(const float* a, const float* b, std::size_t dim) noexcept;
float ip_avx512(const float* a, const float* b, std::size_t dim) noexcept;
#endif

#if defined(__aarch64__)
float l2_neon(const float* a, const float* b, std::size_t dim) noexcept;
float ip_neon(const float* a, const float* b, std::size_t dim) noexcept;
#endif

}  // namespace oryx::kernels
