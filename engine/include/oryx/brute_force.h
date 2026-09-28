#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "oryx/distance.h"
#include "oryx/hnsw.h"

namespace oryx {

// Exact k-NN by scanning every row: the ground truth for recall and the baseline for speedups.
// Labels are row indices. For Metric::Cosine, `data` must already be normalized (the query is
// normalized here).
std::vector<SearchResult> brute_force_search(const float* data, std::size_t n, std::size_t dim, const float* query,
                                             std::size_t k, Metric metric);

// Exact neighbors (row indices) for a batch of queries, split across `threads` workers
// (0 = hardware concurrency).
std::vector<std::vector<std::uint32_t>> compute_ground_truth(const float* data, std::size_t n, std::size_t dim,
                                                             const float* queries, std::size_t num_queries,
                                                             std::size_t k, Metric metric, std::size_t threads = 0);

// Fraction of the true k nearest neighbors that appear in the returned top k, averaged over queries.
double recall_at_k(const std::vector<std::vector<SearchResult>>& results,
                   const std::vector<std::vector<std::uint32_t>>& ground_truth, std::size_t k);

}  // namespace oryx
