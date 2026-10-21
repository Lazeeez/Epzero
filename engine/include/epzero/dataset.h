#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace epzero {

struct Dataset {
  std::size_t n = 0;
  std::size_t dim = 0;
  std::vector<float> data;  // row-major, n * dim

  const float* row(std::size_t i) const noexcept { return data.data() + i * dim; }
};

// TEXMEX formats used by SIFT1M / GIST1M / ann-benchmarks: each row is an int32 dimension
// followed by that many values. `max_rows` = 0 reads the whole file.
Dataset read_fvecs(const std::string& path, std::size_t max_rows = 0);
Dataset read_bvecs(const std::string& path, std::size_t max_rows = 0);
std::vector<std::vector<std::uint32_t>> read_ivecs(const std::string& path, std::size_t max_rows = 0);

struct SyntheticData {
  Dataset base;
  Dataset queries;
};

// Points drawn around random Gaussian cluster centers. Real embeddings are clustered, so this
// is a more honest stand-in than uniform noise when no real dataset is available.
SyntheticData make_clustered(std::size_t num_base, std::size_t num_queries, std::size_t dim, std::size_t clusters,
                             std::uint64_t seed);

}  // namespace epzero
