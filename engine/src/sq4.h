#pragma once

// 4-bit scalar codes (HnswParams::code_bits = 4): the packed layout shared by the encoder, the scorer
// and every kernel.
//
// Codes are padded to a multiple of 32 dimensions (padding codes are 0, and their weights are 0).
// Whole blocks of 64 dimensions take 32 bytes: byte b holds dimension b in its low nibble and dimension
// 32 + b in its high one, so a 32-byte load gives the block's first 32 codes with a mask and its last 32
// with a shift and a mask, each already in the order of the weights: no shuffles. A final block of 32
// dimensions takes 16 bytes the same way (dimension b low, 16 + b high).

#include <cstddef>
#include <cstdint>

namespace epzero::sq4 {

inline constexpr std::size_t padded(std::size_t dim) noexcept { return (dim + 31) / 32 * 32; }
inline constexpr std::size_t bytes(std::size_t dim) noexcept { return padded(dim) / 2; }

// Where dimension `i` (of `dim` codes) lives: its byte, and 0 for the low nibble or 4 for the high one.
struct Slot {
  std::size_t byte;
  unsigned shift;
};
inline constexpr Slot slot(std::size_t i, std::size_t dim) noexcept {
  const std::size_t whole = padded(dim) / 64 * 64;
  if (i < whole) {
    const std::size_t j = i % 64;
    return {i / 64 * 32 + (j & 31), j < 32 ? 0u : 4u};
  }
  const std::size_t j = i - whole;
  return {whole / 2 + (j & 15), j < 16 ? 0u : 4u};
}

inline std::uint8_t get(const std::uint8_t* codes, std::size_t i, std::size_t dim) noexcept {
  const Slot s = slot(i, dim);
  return static_cast<std::uint8_t>((codes[s.byte] >> s.shift) & 0x0F);
}

// Writes `dim` codes (each 0..15) into bytes(dim) bytes, padding with zeros.
inline void pack(const std::uint8_t* values, std::size_t dim, std::uint8_t* out) noexcept {
  for (std::size_t b = 0; b < bytes(dim); ++b) out[b] = 0;
  for (std::size_t i = 0; i < dim; ++i) {
    const Slot s = slot(i, dim);
    out[s.byte] = static_cast<std::uint8_t>(out[s.byte] | ((values[i] & 0x0F) << s.shift));
  }
}

}  // namespace epzero::sq4
