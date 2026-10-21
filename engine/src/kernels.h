#pragma once

// Per-ISA distance kernels. Each ISA lives in its own translation unit compiled with only
// that ISA's flags, so no AVX instruction can leak into code that runs before dispatch.

#include <cstddef>
#include <cstdint>

namespace oryx::kernels {

// Float kernels: L2 returns the squared distance, IP returns 1 - dot.
// SQ8 kernels return sum_i w[i] * codes[i]: the core of asymmetric distance against uint8 codes.

float l2_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float ip_scalar(const float* a, const float* b, std::size_t dim) noexcept;
float sq8_dot_scalar(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept;
std::uint32_t rabitq_ip_scalar(const std::uint64_t* bits, const std::uint64_t* planes, std::size_t words) noexcept;

#if defined(__x86_64__) || defined(_M_X64)
float l2_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float ip_avx2(const float* a, const float* b, std::size_t dim) noexcept;
float sq8_dot_avx2(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept;
std::uint32_t rabitq_ip_popcnt(const std::uint64_t* bits, const std::uint64_t* planes, std::size_t words) noexcept;
float l2_avx512(const float* a, const float* b, std::size_t dim) noexcept;
float ip_avx512(const float* a, const float* b, std::size_t dim) noexcept;
float sq8_dot_avx512(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept;
#endif

#if defined(__aarch64__)
float l2_neon(const float* a, const float* b, std::size_t dim) noexcept;
float ip_neon(const float* a, const float* b, std::size_t dim) noexcept;
float sq8_dot_neon(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept;
#endif

#if defined(__wasm_simd128__)
float l2_wasm(const float* a, const float* b, std::size_t dim) noexcept;
float ip_wasm(const float* a, const float* b, std::size_t dim) noexcept;
float sq8_dot_wasm(const float* w, const std::uint8_t* codes, std::size_t dim) noexcept;
#endif

}  // namespace oryx::kernels
