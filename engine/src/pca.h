// The top principal directions of a sample, for the graph walk's compact codes (HnswParams::nav_dim).
// Subspace iteration on the covariance matrix, with no dependencies. The directions only have to
// span the top of the spectrum: each becomes one code byte with its own scale, so their order and
// the exact split between them don't matter.
//
// Every step is a blocked matrix product run as one parallel_for: the covariance reads each sample
// row once per block of covariance rows, the iteration streams the covariance once per block of its
// rows, and the directions are orthonormalized by Cholesky QR. Straightforward loops read the sample
// once per dimension and the covariance once per direction, tens of gigabytes at 1,000 dimensions,
// and Gram-Schmidt started a thread pool per direction.
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
inline constexpr std::size_t kRowBlock = 16;       // covariance rows per task

// Gram-Schmidt, run twice: the fallback when Cholesky QR meets a (near-)degenerate set of rows.
inline void gram_schmidt(std::vector<double>& q, std::size_t k, std::size_t dim) {
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t i = 0; i < k; ++i) {
      double* row = q.data() + i * dim;
      for (std::size_t j = 0; j < i; ++j) {
        const double* other = q.data() + j * dim;
        double dot = 0.0;
        for (std::size_t d = 0; d < dim; ++d) dot += row[d] * other[d];
        for (std::size_t d = 0; d < dim; ++d) row[d] -= dot * other[d];
      }
      double norm = 0.0;
      for (std::size_t d = 0; d < dim; ++d) norm += row[d] * row[d];
      norm = std::sqrt(norm);
      for (std::size_t d = 0; d < dim; ++d) row[d] = norm > 0.0 ? row[d] / norm : 0.0;
    }
  }
}

// One Cholesky QR step: Q <- L^-1 Q where L L^T = Q Q^T. Returns false if Q Q^T is not safely positive
// definite (the rows are nearly dependent), leaving q unchanged.
inline bool cholesky_qr(std::vector<double>& q, std::size_t k, std::size_t dim, std::size_t threads) {
  std::vector<double> gram(k * k, 0.0);
  parallel_for(0, k, threads, [&](std::size_t i) {
    const double* a = q.data() + i * dim;
    for (std::size_t j = 0; j <= i; ++j) {
      const double* b = q.data() + j * dim;
      double dot = 0.0;
      for (std::size_t d = 0; d < dim; ++d) dot += a[d] * b[d];
      gram[i * k + j] = dot;
    }
  });
  // In place: the lower triangle of `gram` becomes L.
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double sum = gram[i * k + j];
      for (std::size_t m = 0; m < j; ++m) sum -= gram[i * k + m] * gram[j * k + m];
      if (i == j) {
        if (!(sum > 1e-10 * std::max(1.0, gram[i * k + i]))) return false;
        gram[i * k + i] = std::sqrt(sum);
      } else {
        gram[i * k + j] = sum / gram[j * k + j];
      }
    }
  }
  // Forward substitution, column by column: each task owns a slice of the columns.
  constexpr std::size_t kColumns = 64;
  parallel_for(0, (dim + kColumns - 1) / kColumns, threads, [&](std::size_t slice) {
    const std::size_t begin = slice * kColumns, end = std::min(dim, begin + kColumns);
    for (std::size_t i = 0; i < k; ++i) {
      double* row = q.data() + i * dim;
      for (std::size_t m = 0; m < i; ++m) {
        const double l = gram[i * k + m];
        const double* done = q.data() + m * dim;
        for (std::size_t d = begin; d < end; ++d) row[d] -= l * done[d];
      }
      const double inv = 1.0 / gram[i * k + i];
      for (std::size_t d = begin; d < end; ++d) row[d] *= inv;
    }
  });
  return true;
}

// Makes the `k` rows of `q` (each `dim` long) orthonormal. Cholesky QR twice is as accurate as
// Householder QR for well-conditioned rows, which random starts and power iterations give.
inline void orthonormalize(std::vector<double>& q, std::size_t k, std::size_t dim, std::size_t threads) {
  if (!cholesky_qr(q, k, dim, threads) || !cholesky_qr(q, k, dim, threads)) gram_schmidt(q, k, dim);
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
  std::vector<float> centred_mean(mean.begin(), mean.end());
  parallel_for(0, sample, threads, [&](std::size_t i) {
    float* v = x.data() + i * dim;
    for (std::size_t d = 0; d < dim; ++d) v[d] -= centred_mean[d];
  });
  // The upper triangle, a block of rows per task (each sample row is read once per block), then mirrored.
  std::vector<double> cov(dim * dim, 0.0);
  parallel_for(0, (dim + kRowBlock - 1) / kRowBlock, threads, [&](std::size_t block) {
    const std::size_t first = block * kRowBlock, last = std::min(dim, first + kRowBlock);
    for (std::size_t i = 0; i < sample; ++i) {
      const float* v = x.data() + i * dim;
      for (std::size_t a = first; a < last; ++a) {
        const double va = v[a];
        double* out = cov.data() + a * dim;
        for (std::size_t b = a; b < dim; ++b) out[b] += va * v[b];
      }
    }
  });
  for (std::size_t a = 0; a < dim; ++a) {
    for (std::size_t b = 0; b < a; ++b) cov[a * dim + b] = cov[b * dim + a];
  }
  return cov;
}

// out (k x dim) = q (k x dim) times the symmetric cov (dim x dim): a block of cov's rows per task,
// so cov streams through memory once and each block stays in cache while every row of q passes.
inline void multiply(const std::vector<double>& cov, const std::vector<double>& q, std::vector<double>& out,
                     std::size_t k, std::size_t dim, std::size_t threads) {
  parallel_for(0, (dim + kRowBlock - 1) / kRowBlock, threads, [&](std::size_t block) {
    const std::size_t first = block * kRowBlock, last = std::min(dim, first + kRowBlock);
    for (std::size_t r = 0; r < k; ++r) {
      const double* in = q.data() + r * dim;
      for (std::size_t a = first; a < last; ++a) {
        const double* c = cov.data() + a * dim;
        double sum = 0.0;
        for (std::size_t b = 0; b < dim; ++b) sum += c[b] * in[b];
        out[r * dim + a] = sum;
      }
    }
  });
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
    multiply(cov, q, next, k, dim, threads);
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
// them, for up to `k` directions. From a smaller sample than top_directions uses, as it only has to
// rank datasets, not build codes.
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
  std::vector<double> cq(k * dim);
  multiply(cov, q, cq, k, dim, threads);
  std::vector<double> variance(k);
  for (std::size_t r = 0; r < k; ++r) {
    double sum = 0.0;
    for (std::size_t a = 0; a < dim; ++a) sum += q[r * dim + a] * cq[r * dim + a];
    variance[r] = sum;
  }
  std::sort(variance.begin(), variance.end(), std::greater<>());
  std::vector<double> held(k);
  double running = 0.0;
  for (std::size_t i = 0; i < k; ++i) held[i] = std::min(1.0, (running += variance[i]) / total);
  return held;
}

}  // namespace epzero::pca
