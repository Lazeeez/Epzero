#include "epzero/dataset.h"

#include <fstream>
#include <random>
#include <stdexcept>

namespace epzero {
namespace {

// Reads TEXMEX-style rows of `T`, converting each value to `Out`.
template <typename T, typename Out>
std::size_t read_vecs(const std::string& path, std::size_t max_rows, std::vector<Out>& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);

  std::size_t dim = 0;
  std::size_t rows = 0;
  std::vector<T> buffer;
  std::int32_t row_dim;
  while ((max_rows == 0 || rows < max_rows) && in.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim))) {
    if (row_dim <= 0) throw std::runtime_error(path + ": invalid dimension");
    if (dim == 0) {
      dim = static_cast<std::size_t>(row_dim);
      buffer.resize(dim);
    } else if (static_cast<std::size_t>(row_dim) != dim) {
      throw std::runtime_error(path + ": inconsistent dimensions");
    }
    if (!in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(dim * sizeof(T)))) {
      throw std::runtime_error(path + ": truncated row");
    }
    out.insert(out.end(), buffer.begin(), buffer.end());
    ++rows;
  }
  return dim;
}

}  // namespace

Dataset read_fvecs(const std::string& path, std::size_t max_rows) {
  Dataset ds;
  ds.dim = read_vecs<float>(path, max_rows, ds.data);
  ds.n = ds.dim == 0 ? 0 : ds.data.size() / ds.dim;
  return ds;
}

Dataset read_bvecs(const std::string& path, std::size_t max_rows) {
  Dataset ds;
  ds.dim = read_vecs<std::uint8_t>(path, max_rows, ds.data);
  ds.n = ds.dim == 0 ? 0 : ds.data.size() / ds.dim;
  return ds;
}

std::vector<std::vector<std::uint32_t>> read_ivecs(const std::string& path, std::size_t max_rows) {
  std::vector<std::int32_t> flat;
  const std::size_t dim = read_vecs<std::int32_t>(path, max_rows, flat);
  std::vector<std::vector<std::uint32_t>> rows;
  if (dim == 0) return rows;
  rows.reserve(flat.size() / dim);
  for (std::size_t i = 0; i < flat.size(); i += dim) rows.emplace_back(flat.begin() + i, flat.begin() + i + dim);
  return rows;
}

SyntheticData make_clustered(std::size_t num_base, std::size_t num_queries, std::size_t dim, std::size_t clusters,
                             std::uint64_t seed) {
  if (clusters == 0) throw std::invalid_argument("clusters must be > 0");
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> center_dist(0.0f, 1.0f);
  std::normal_distribution<float> spread(0.0f, 0.35f);
  std::uniform_int_distribution<std::size_t> pick(0, clusters - 1);

  std::vector<float> centers(clusters * dim);
  for (float& c : centers) c = center_dist(rng);

  auto sample = [&](std::size_t n) {
    Dataset ds{n, dim, std::vector<float>(n * dim)};
    for (std::size_t i = 0; i < n; ++i) {
      const float* center = centers.data() + pick(rng) * dim;
      for (std::size_t j = 0; j < dim; ++j) ds.data[i * dim + j] = center[j] + spread(rng);
    }
    return ds;
  };
  SyntheticData out;
  out.base = sample(num_base);
  out.queries = sample(num_queries);
  return out;
}

}  // namespace epzero
