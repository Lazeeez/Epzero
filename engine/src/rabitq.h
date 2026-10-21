#pragma once

// RaBitQ (Gao and Long, SIGMOD 2024): 1-bit quantization with an unbiased distance estimator.
//
// A vector o_r is stored relative to the dataset's centroid c, as the unit vector
// o = (o_r - c) / |o_r - c| randomly rotated (o' = P o), then reduced to its signs: bit i is
// o'_i > 0. The quantized unit vector is ō' = (2b - 1) / sqrt(D). For a query q (also centred,
// normalized and rotated), the paper's estimator is
//
//     <o, q>  ≈  <ō, q> / <ō, o>,    <ō, q> = (2 <b, q'> - sum_i q'_i) / sqrt(D),
//
// where <ō, o> = sum_i |o'_i| / sqrt(D) is stored per vector. <b, q'> is computed from q'
// quantized to 4 bits as Δ <b, q_u> + low * popcount(b), and <b, q_u> = sum_j 2^j popcount(b & q_u_j)
// over the four bit planes of q_u: a handful of AND + POPCNT per vector.
//
// P is three rounds of (random signs, fast Walsh-Hadamard transform): orthogonal, O(D log D), and
// needs D a power of two, so vectors are zero-padded to D = padded_dim(dim).
//
// In the index (see Scorer::bounded in hnsw.cpp) the estimate is a filter, not a replacement: a
// neighbour whose estimate, less its error bound, cannot beat the beam's worst is skipped without
// reading its full vector; any other is scored exactly. Recall matches full precision. Measured on
// a laptop (one thread): it skips 27% of full-vector reads on SIFT1M (128-d) and 64% on CLIP
// (512-d), yet is no faster than full precision (0.6x and 1.0x the throughput at ef 64), because
// the reads that remain go to a separate array and each costs a cache miss. Its place is where a
// full vector is expensive to fetch (SSD, far larger collections); in memory, SQ8 is faster.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace oryx::rabitq {

// Per-vector numbers stored beside the bits.
struct Factors {
  float norm;   // |o_r - c|
  float ip_ob;  // <ō, o>
  float dot_c;  // <o_r, c> (for inner product and cosine)
  float ones;   // popcount(b)
};

inline std::size_t padded_dim(std::size_t dim) noexcept { return std::max<std::size_t>(64, std::bit_ceil(dim)); }
inline std::size_t words(std::size_t padded) noexcept { return padded / 64; }

// Three rounds of D random signs, from the index's seed.
inline std::vector<std::int8_t> random_signs(std::size_t padded, std::uint64_t seed) {
  std::mt19937_64 rng(seed ^ 0x52614269745121ull);
  std::vector<std::int8_t> signs(3 * padded);
  for (auto& s : signs) s = (rng() & 1) ? 1 : -1;
  return signs;
}

// In place: x <- P x for P = H S3 H S2 H S1 / D^(3/2), H the (unnormalized) Walsh-Hadamard matrix.
inline void rotate(float* x, std::size_t padded, const std::int8_t* signs) noexcept {
  const float scale = 1.0f / std::sqrt(static_cast<float>(padded));
  for (int round = 0; round < 3; ++round) {
    const std::int8_t* s = signs + round * padded;
    for (std::size_t i = 0; i < padded; ++i) x[i] *= s[i];
    for (std::size_t len = 1; len < padded; len <<= 1) {
      for (std::size_t i = 0; i < padded; i += len << 1) {
        for (std::size_t j = i; j < i + len; ++j) {
          const float a = x[j];
          const float b = x[j + len];
          x[j] = a + b;
          x[j + len] = a - b;
        }
      }
    }
    for (std::size_t i = 0; i < padded; ++i) x[i] *= scale;
  }
}

// Encodes v (dim floats) into `bits` (words(padded) words) and its factors. `scratch` holds `padded`.
inline void encode(const float* v, std::size_t dim, const float* center, const std::int8_t* signs, std::size_t padded,
                   std::uint64_t* bits, Factors& f, float* scratch) noexcept {
  float norm_sq = 0.0f;
  float dot_c = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float d = v[i] - center[i];
    scratch[i] = d;
    norm_sq += d * d;
    dot_c += v[i] * center[i];
  }
  std::fill(scratch + dim, scratch + padded, 0.0f);
  const float norm = std::sqrt(norm_sq);
  const float inv = norm > 0.0f ? 1.0f / norm : 0.0f;
  for (std::size_t i = 0; i < dim; ++i) scratch[i] *= inv;
  rotate(scratch, padded, signs);
  std::fill(bits, bits + words(padded), 0);
  float abs_sum = 0.0f;
  std::uint32_t ones = 0;
  for (std::size_t i = 0; i < padded; ++i) {
    abs_sum += std::fabs(scratch[i]);
    if (scratch[i] > 0.0f) {
      bits[i / 64] |= std::uint64_t{1} << (i % 64);
      ++ones;
    }
  }
  const float ip_ob = abs_sum / std::sqrt(static_cast<float>(padded));
  f = {norm, ip_ob > 1e-6f ? ip_ob : 1.0f, dot_c, static_cast<float>(ones)};
}

// A query, prepared once per search.
struct Query {
  std::vector<std::uint64_t> planes;  // 4 bit planes of q_u, words(padded) words each
  float low = 0.0f;                   // q'_i ≈ low + delta * q_u_i
  float delta = 1.0f;
  float sum = 0.0f;                   // sum_i q'_i
  float norm = 0.0f;                  // |q_r - c|
  float dot_c = 0.0f;                 // <q_r, c>
  float inv_sqrt_d = 1.0f;
};

inline void prepare(const float* q, std::size_t dim, const float* center, const std::int8_t* signs, std::size_t padded,
                    Query& out) {
  std::vector<float> x(padded, 0.0f);
  float norm_sq = 0.0f;
  out.dot_c = 0.0f;
  for (std::size_t i = 0; i < dim; ++i) {
    x[i] = q[i] - center[i];
    norm_sq += x[i] * x[i];
    out.dot_c += q[i] * center[i];
  }
  out.norm = std::sqrt(norm_sq);
  const float inv = out.norm > 0.0f ? 1.0f / out.norm : 0.0f;
  for (std::size_t i = 0; i < dim; ++i) x[i] *= inv;
  rotate(x.data(), padded, signs);
  const auto [lo, hi] = std::minmax_element(x.begin(), x.end());
  out.low = *lo;
  out.delta = *hi > *lo ? (*hi - *lo) / 15.0f : 1.0f;
  out.sum = 0.0f;
  const std::size_t w = words(padded);
  out.planes.assign(4 * w, 0);
  for (std::size_t i = 0; i < padded; ++i) {
    out.sum += x[i];
    const auto code = static_cast<std::uint32_t>(std::lround((x[i] - out.low) / out.delta));
    for (std::uint32_t j = 0; j < 4; ++j) {
      if (code >> j & 1) out.planes[j * w + i / 64] |= std::uint64_t{1} << (i % 64);
    }
  }
  out.inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(padded));
}

// The estimate of <o, q> for unit o and q, from <b, q_u> (the popcount kernel's result).
inline float estimate(const Query& q, std::uint32_t ip_bqu, const Factors& f) noexcept {
  const float ip_bq = q.delta * static_cast<float>(ip_bqu) + q.low * f.ones;
  const float ip_obq = (2.0f * ip_bq - q.sum) * q.inv_sqrt_d;
  return std::clamp(ip_obq / f.ip_ob, -1.0f, 1.0f);
}

// The paper's error bound on that estimate (Theorem 3.2): with high probability (eps0 = 1.9),
// |estimate - <o, q>| <= eps0 * sqrt((1 - <ō, o>^2) / <ō, o>^2) / sqrt(D - 1).
inline float error_bound(const Factors& f, std::size_t padded) noexcept {
  constexpr float kEps0 = 1.9f;
  const float ip = std::min(f.ip_ob, 1.0f);
  return kEps0 * std::sqrt(std::max(0.0f, 1.0f - ip * ip)) / ip / std::sqrt(static_cast<float>(padded - 1));
}

}  // namespace oryx::rabitq
