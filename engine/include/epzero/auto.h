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
#include <string>
#include <utility>
#include <vector>

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
  std::optional<bool> half_vectors;  // 16-bit full vectors: half their memory (never chosen automatically)
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

// Re-measures the beam width that reaches `recall` (0: the profile's target, else 0.95) on `queries`
// (count rows of dim floats) and stores it in the index's profile, which search() then uses. Pass real
// queries: rows already in the graph are linked to their own neighbours, so they make the search look
// easier than it is. Not thread-safe against concurrent searches. Returns (ef, recall reached).
std::pair<std::size_t, float> recalibrate(HnswIndex& index, const float* queries, std::size_t count, float recall = 0.0f,
                                          std::size_t threads = 0);

// What an index needs, if anything, since it was built: a measurement plus advice in plain words.
struct IndexHealth {
  std::size_t live = 0;
  std::size_t deleted = 0;          // tombstones still in the graph
  double growth = 0.0;              // change in live vectors since calibration, as a share (0.3: 30% more)
  double nav_variance_built = 0.0;  // share of the variance the principal directions held when built
  double nav_variance_now = 0.0;    // and on the vectors in the index now (0 without principal directions)
  double code_error = 0.0;          // share of the variance the 8-bit codes miss (HnswIndex::code_error)
  std::vector<std::string> advice;  // empty when nothing needs doing
};
IndexHealth check_health(const HnswIndex& index, std::size_t threads = 0);

// Whether `queries` (count rows) look like the data the index was calibrated on. A query from the data's
// distribution sits about as far from its nearest neighbour as rows of the data do from theirs; text
// queries against image vectors, say, sit far farther (VIBE's text-to-image sets: 2.3x, against 0.99-1.09x
// for queries like the data). Then recall calibrated on rows of the data overestimates theirs (0.74-0.80
// against a 0.95 target on those sets), and `advice` says to recalibrate on them.
struct QueryCheck {
  double query_distance = 0.0;  // median distance from a query to its nearest neighbour (sample)
  double row_distance = 0.0;    // the same for rows of the data
  double ratio = 0.0;           // query_distance / row_distance
  bool shifted = false;         // ratio above 1.5
  std::vector<std::string> advice;
};
QueryCheck check_queries(const HnswIndex& index, const float* queries, std::size_t count, std::size_t threads = 0);

// Share of the variance of the index's vectors (a sample) that its principal directions hold.
double nav_variance(const HnswIndex& index, std::size_t threads = 0);

}  // namespace epzero
