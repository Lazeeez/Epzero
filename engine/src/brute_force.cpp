#include "oryx/brute_force.h"

#include <algorithm>
#include <atomic>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace oryx {

std::vector<SearchResult> brute_force_search(const float* data, std::size_t n, std::size_t dim, const float* query,
                                             std::size_t k, Metric metric) {
  std::vector<float> normalized;
  if (metric == Metric::Cosine) {
    normalized.assign(query, query + dim);
    normalize(normalized.data(), dim);
    query = normalized.data();
  }
  const DistanceFn dist = get_distance_fn(metric);
  auto farther = [](const SearchResult& a, const SearchResult& b) { return a.distance < b.distance; };
  std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(farther)> best(farther);

  for (std::size_t i = 0; i < n; ++i) {
    const float d = dist(query, data + i * dim, dim);
    if (best.size() < k) {
      best.push({i, d});
    } else if (d < best.top().distance) {
      best.pop();
      best.push({i, d});
    }
  }

  std::vector<SearchResult> results(best.size());
  for (std::size_t i = results.size(); i-- > 0;) {
    results[i] = best.top();
    best.pop();
  }
  return results;
}

namespace {

using TopK = std::priority_queue<SearchResult, std::vector<SearchResult>, bool (*)(const SearchResult&, const SearchResult&)>;

bool farther_first(const SearchResult& a, const SearchResult& b) { return a.distance < b.distance; }

// Scores one tile of queries against rows [start, end). A separate, frequently called function
// on purpose: WebAssembly engines start code in a baseline tier and switch a function to
// optimized code only on a later call, so the hot loop must not live inside one long-running
// call (see also the warm-up in compute_ground_truth).
[[gnu::noinline]] void score_block(const float* tile, std::size_t count, std::size_t dim, const float* data,
                                   std::size_t start, std::size_t end, DistanceFn dist, std::size_t k,
                                   std::vector<TopK>& best) {
  for (std::size_t q = 0; q < count; ++q) {
    const float* query = tile + q * dim;
    TopK& heap = best[q];
    for (std::size_t i = start; i < end; ++i) {
      const float d = dist(query, data + i * dim, dim);
      if (heap.size() < k) {
        heap.push({i, d});
      } else if (d < heap.top().distance) {
        heap.pop();
        heap.push({i, d});
      }
    }
  }
}

}  // namespace

// Exact k-NN for a batch of queries, blocked for the cache. A naive scan is bound by memory
// bandwidth: every query streams the whole dataset from DRAM (1,000 queries over 100k x 128-d is
// ~51 GB of traffic). Instead each worker takes a tile of queries and walks the dataset in blocks
// small enough to stay in L2, scoring every query in the tile against a block before moving on,
// so each block is fetched once per tile instead of once per query (32x less traffic). Measured
// 2.5x faster natively and 1.7x in WebAssembly (100k x 128-d, 1,000 queries, 12 threads).
// Distances and scan order per query are unchanged, so the results are identical to
// brute_force_search.
std::vector<std::vector<std::uint32_t>> compute_ground_truth(const float* data, std::size_t n, std::size_t dim,
                                                             const float* queries, std::size_t num_queries,
                                                             std::size_t k, Metric metric, std::size_t threads) {
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  constexpr std::size_t kTile = 32;  // queries sharing one pass over each block
  // Rows per block: ~128 KiB of vectors, comfortably inside L2 alongside the tile's queries.
  const std::size_t block = std::max<std::size_t>(16, (128 * 1024) / (dim * sizeof(float)));
  const DistanceFn dist = get_distance_fn(metric);
  const std::size_t tiles = (num_queries + kTile - 1) / kTile;
  std::vector<std::vector<std::uint32_t>> truth(num_queries);
  std::atomic<std::size_t> next_tile{0};

#ifdef __EMSCRIPTEN__
  // Warm-up (~6 ms, results discarded): a few quick calls on one thread get score_block through
  // the engine's baseline tier while the CPU is idle. Otherwise the optimizing compiler runs while
  // every core is busy with the real scan, and a cold first run measured 1.2-4.6x slower.
  if (num_queries > 0 && n > 0) {
    const std::size_t count = std::min(kTile, num_queries);
    std::vector<float> tile(queries, queries + count * dim);
    if (metric == Metric::Cosine) {
      for (std::size_t q = 0; q < count; ++q) normalize(tile.data() + q * dim, dim);
    }
    std::vector<TopK> scratch(count, TopK(farther_first));
    const std::size_t rows = std::min(n, 16 * block);
    for (std::size_t start = 0; start < rows; start += block) {
      score_block(tile.data(), count, dim, data, start, std::min(rows, start + block), dist, k, scratch);
    }
  }
#endif

  auto worker = [&] {
    std::vector<float> tile(kTile * dim);
    std::vector<TopK> best;
    for (std::size_t t = next_tile++; t < tiles; t = next_tile++) {
      const std::size_t first = t * kTile;
      const std::size_t count = std::min(kTile, num_queries - first);
      std::copy(queries + first * dim, queries + (first + count) * dim, tile.begin());
      if (metric == Metric::Cosine) {
        for (std::size_t q = 0; q < count; ++q) normalize(tile.data() + q * dim, dim);
      }
      best.assign(count, TopK(farther_first));
      for (std::size_t start = 0; start < n; start += block) {
        score_block(tile.data(), count, dim, data, start, std::min(n, start + block), dist, k, best);
      }
      for (std::size_t q = 0; q < count; ++q) {
        std::vector<std::uint32_t>& out = truth[first + q];
        out.resize(best[q].size());
        for (std::size_t i = out.size(); i-- > 0;) {
          out[i] = static_cast<std::uint32_t>(best[q].top().label);
          best[q].pop();
        }
      }
    }
  };

  threads = std::min(threads, std::max<std::size_t>(tiles, 1));
  std::vector<std::thread> workers;
  for (std::size_t t = 1; t < threads; ++t) workers.emplace_back(worker);
  worker();  // the calling thread works too
  for (auto& w : workers) w.join();
  return truth;
}

double recall_at_k(const std::vector<std::vector<SearchResult>>& results,
                   const std::vector<std::vector<std::uint32_t>>& ground_truth, std::size_t k) {
  if (results.size() != ground_truth.size()) throw std::invalid_argument("results/ground truth size mismatch");
  if (results.empty() || k == 0) return 0.0;
  std::size_t hits = 0;
  for (std::size_t q = 0; q < results.size(); ++q) {
    const std::size_t depth = std::min(k, ground_truth[q].size());
    const std::unordered_set<std::uint64_t> truth(ground_truth[q].begin(), ground_truth[q].begin() + depth);
    for (std::size_t i = 0; i < results[q].size() && i < k; ++i) hits += truth.count(results[q][i].label);
  }
  return static_cast<double>(hits) / static_cast<double>(results.size() * k);
}

}  // namespace oryx
