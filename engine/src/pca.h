// The top principal directions of a sample, for the graph walk's compact codes (HnswParams::nav_dim).
// Subspace iteration on the covariance matrix, with no dependencies. The directions only have to
// span the top of the spectrum: each becomes one code byte with its own scale, so their order and
// the exact split between them don't matter.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

#include "epzero/parallel.h"

namespace epzero::pca {

inline constexpr std::size_t kMaxSample = 50'000;  // evenly spaced rows that go into the covariance
inline constexpr int kIterations = 16;

// Makes the `k` rows of `q` (each `dim` long) orthonormal: Gram-Schmidt, run twice for accuracy.
inline void orthonormalize(std::vector<double>& q, std::size_t k, std::size_t dim, std::size_t threads) {
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t i = 0; i < k; ++i) {
      double* row = q.data() + i * dim;
      double norm = 0.0;
      for (std::size_t d = 0; d < dim; ++d) norm += row[d] * row[d];
      norm = std::sqrt(norm);
      if (norm > 0.0) {
        for (std::size_t d = 0; d < dim; ++d) row[d] /= norm;
      }
      parallel_for(i + 1, k, threads, [&](std::size_t j) {
        double* other = q.data() + j * dim;
        double dot = 0.0;
        for (std::size_t d = 0; d < dim; ++d) dot += row[d] * other[d];
        for (std::size_t d = 0; d < dim; ++d) other[d] -= dot * row[d];
      });
    }
  }
}

// The covariance of up to `max_sample` evenly spaced rows, dim x dim. `row(i, out)` writes row i.
template <typename RowFn>
std::vector<double> covariance(RowFn row, std::size_t n, std::size_t dim, std::size_t max_sample, std::size_t threads) {
  const std::size_t sample = std::min(n, max_sample);
  std::vector<float> x(sample * dim);
  parallel_for(0, sample, threads, [&](std::size_t i) { row(i * n / sample, x.data() + i * dim); });
  std::vector<double> mean(dim, 0.0);
  for (std::size_t i = 0; i < sample; ++i) {
    const float* v = x.data() + i * dim;
    for (std::size_t d = 0; d < dim; ++d) mean[d] += v[d];
  }
  for (double& m : mean) m /= static_cast<double>(sample);
  parallel_for(0, sample, threads, [&](std::size_t i) {
    float* v = x.data() + i * dim;
    for (std::size_t d = 0; d < dim; ++d) v[d] -= static_cast<float>(mean[d]);
  });
  // The upper triangle, one row of it per task, then mirrored.
  std::vector<double> cov(dim * dim, 0.0);
  parallel_for(0, dim, threads, [&](std::size_t a) {
    double* out = cov.data() + a * dim;
    for (std::size_t i = 0; i < sample; ++i) {
      const float* v = x.data() + i * dim;
      const double va = v[a];
      for (std::size_t b = a; b < dim; ++b) out[b] += va * v[b];
    }
  });
  for (std::size_t a = 0; a < dim; ++a) {
    for (std::size_t b = 0; b < a; ++b) cov[a * dim + b] = cov[b * dim + a];
  }
  return cov;
}

// `k` orthonormal rows spanning the top of the covariance's spectrum: subspace iteration.
inline std::vector<double> subspace(const std::vector<double>& cov, std::size_t dim, std::size_t k, std::uint64_t seed,
                                    std::size_t threads) {
  std::vector<double> q(k * dim);
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> normal;
  for (double& v : q) v = normal(rng);
  orthonormalize(q, k, dim, threads);
  std::vector<double> next(k * dim);
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    parallel_for(0, k, threads, [&](std::size_t r) {
      const double* in = q.data() + r * dim;
      double* out = next.data() + r * dim;
      for (std::size_t a = 0; a < dim; ++a) {
        const double* c = cov.data() + a * dim;
        double sum = 0.0;
        for (std::size_t b = 0; b < dim; ++b) sum += c[b] * in[b];
        out[a] = sum;
      }
    });
    q.swap(next);
    orthonormalize(q, k, dim, threads);
  }
  return q;
}

// Returns `k` orthonormal directions as k rows of `dim` floats. `row(i, out)` writes sample row i.
template <typename RowFn>
std::vector<float> top_directions(RowFn row, std::size_t n, std::size_t dim, std::size_t k, std::uint64_t seed,
                                  std::size_t threads = 0) {
  const auto q = subspace(covariance(row, n, dim, kMaxSample, threads), dim, k, seed, threads);
  return std::vector<float>(q.begin(), q.end());
}

// How much of the variance the top directions hold: `held[i]` is the share held by the top i + 1 of
// them, for up to `k` directions (fewer when the spectrum is exhausted). From a smaller sample than
// top_directions uses, as it only has to rank datasets, not build codes.
template <typename RowFn>
std::vector<double> variance_held(RowFn row, std::size_t n, std::size_t dim, std::size_t k, std::uint64_t seed,
                                  std::size_t threads = 0, std::size_t max_sample = 20'000) {
  const auto cov = covariance(row, n, dim, max_sample, threads);
  double total = 0.0;
  for (std::size_t d = 0; d < dim; ++d) total += cov[d * dim + d];
  k = std::min(k, dim);
  if (total <= 0.0 || k == 0) return std::vector<double>(k, 1.0);
  const auto q = subspace(cov, dim, k, seed, threads);
  // Each direction's variance (its Rayleigh quotient), largest first: the iteration only spans the
  // top subspace, it doesn't order the directions within it.
  std::vector<double> variance(k);
  parallel_for(0, k, threads, [&](std::size_t r) {
    const double* v = q.data() + r * dim;
    double sum = 0.0;
    for (std::size_t a = 0; a < dim; ++a) {
      const double* c = cov.data() + a * dim;
      double cv = 0.0;
      for (std::size_t b = 0; b < dim; ++b) cv += c[b] * v[b];
      sum += v[a] * cv;
    }
    variance[r] = sum;
  });
  std::sort(variance.begin(), variance.end(), std::greater<>());
  std::vector<double> held(k);
  double running = 0.0;
  for (std::size_t i = 0; i < k; ++i) held[i] = std::min(1.0, (running += variance[i]) / total);
  return held;
}

}  // namespace epzero::pca
