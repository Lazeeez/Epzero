#pragma once

// Automatic configuration: an index set up for its data and for the machine it is built on, with
// every choice overridable. The rules come from measurements in docs/performance.md.
//
//   epzero::AutoOptions options;
//   options.metric = epzero::Metric::Cosine;
//   options.fixed.nav_dim = 0;                       // override any choice
//   auto index = epzero::build_auto(data, nullptr, n, dim, options);
//   index->profile();                                // what it chose, and why
//
// From the data: 8-bit codes, whether to walk on principal directions, the build passes, the
// re-rank, and the beam width (ef) that reaches the recall target, calibrated on rows held out of
// the graph (or on queries you pass). From the machine: the prefetch setting, timed here, and every
// core. All of it is saved with the index; see IndexProfile.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "epzero/hnsw.h"

namespace epzero {

// Settings build_auto() would otherwise choose. Any that is set is used as given.
struct AutoFixed {
  std::optional<Quantization> quantization;
  std::optional<std::size_t> nav_dim;
  std::optional<bool> fast_build;
  std::optional<bool> refine;      // refine_links() after the build
  std::optional<bool> fill_after;  // fill_free_links() after the build
  std::optional<std::size_t> rerank;
  std::optional<float> rerank_z;
  std::optional<std::size_t> ef;   // skips calibration
  std::optional<std::pair<std::size_t, std::size_t>> prefetch;  // (depth, lines); skips timing
};

struct AutoOptions {
  Metric metric = Metric::L2;
  std::size_t M = 16;
  std::size_t ef_construction = 200;
  std::size_t k = 10;            // the k the beam width is calibrated for
  float recall = 0.95f;          // the recall target
  std::size_t threads = 0;       // 0: every core
  std::size_t capacity = 0;      // vectors the index can hold (0: the rows given)
  std::uint64_t seed = 42;
  // Optional typical queries (query_count rows of dim floats) to calibrate on. Rows of the data are
  // accurate when queries look like the data, and optimistic when they don't.
  const float* queries = nullptr;
  std::size_t query_count = 0;
  AutoFixed fixed;
};

struct AutoPlan {
  HnswParams params;
  IndexProfile profile;  // ef is 0 while it still has to be calibrated
  bool calibrate = true;  // build_auto will calibrate ef
  bool tune = true;       // build_auto will time prefetch settings
};

// What build_auto() would do with `data` (n rows of dim floats), without building.
AutoPlan plan_auto(const float* data, std::size_t n, std::size_t dim, const AutoOptions& options = {});

// Builds an index of `data` with labels[i] for row i (labels null: 0..n-1), configured automatically.
std::unique_ptr<HnswIndex> build_auto(const float* data, const std::uint64_t* labels, std::size_t n, std::size_t dim,
                                      const AutoOptions& options = {});

}  // namespace epzero
