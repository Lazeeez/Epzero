#include "oryx/brute_force.h"

#include <algorithm>
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

std::vector<std::vector<std::uint32_t>> compute_ground_truth(const float* data, std::size_t n, std::size_t dim,
                                                             const float* queries, std::size_t num_queries,
                                                             std::size_t k, Metric metric, std::size_t threads) {
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::vector<std::uint32_t>> truth(num_queries);
  std::vector<std::thread> workers;
  for (std::size_t t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      for (std::size_t q = t; q < num_queries; q += threads) {
        for (const SearchResult& r : brute_force_search(data, n, dim, queries + q * dim, k, metric)) {
          truth[q].push_back(static_cast<std::uint32_t>(r.label));
        }
      }
    });
  }
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
