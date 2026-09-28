// Embind bindings for the in-browser benchmark lab. The whole workload (data generation, exact
// ground truth, parallel index build, ef sweeps) runs inside WebAssembly, so the page measures
// the engine rather than JavaScript. Designed to be driven from a Web Worker: long calls block.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "oryx/brute_force.h"
#include "oryx/dataset.h"
#include "oryx/hnsw.h"
#include "oryx/parallel.h"

using emscripten::val;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

template <typename T>
val copy_to_js(const char* type, const std::vector<T>& data) {
  // Constructing a typed array from a view copies it out of the WASM heap; the view itself would
  // be invalidated as soon as the heap grows.
  return val::global(type).new_(emscripten::typed_memory_view(data.size(), data.data()));
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(p * static_cast<double>(values.size() - 1))];
}

class Lab {
 public:
  // Clustered synthetic vectors, generated inside WASM (no download).
  double generate(std::size_t n, std::size_t num_queries, std::size_t dim, std::size_t clusters, double seed) {
    const auto start = Clock::now();
    auto data = oryx::make_clustered(n, num_queries, dim, clusters, static_cast<std::uint64_t>(seed));
    base_ = std::move(data.base);
    queries_ = std::move(data.queries);
    truth_.clear();
    index_.reset();
    return ms_since(start);
  }

  double compute_truth(std::size_t k, std::size_t threads) {
    require_data();
    const auto start = Clock::now();
    truth_ = oryx::compute_ground_truth(base_.data.data(), base_.n, base_.dim, queries_.data.data(), queries_.n, k,
                                        oryx::Metric::L2, threads);
    truth_k_ = k;
    return ms_since(start);
  }

  void begin_build(std::size_t M, std::size_t ef_construction) {
    require_data();
    index_ = std::make_unique<oryx::HnswIndex>(oryx::HnswParams{
        .dim = base_.dim, .metric = oryx::Metric::L2, .max_elements = base_.n, .M = M,
        .ef_construction = ef_construction});
    built_ = 0;
    build_ms_ = 0.0;
  }

  // Inserts the next `count` vectors on `threads` threads and returns the total inserted, so the
  // worker can report progress between steps.
  std::size_t build_step(std::size_t count, std::size_t threads) {
    require_index();
    const std::size_t end = std::min(base_.n, built_ + count);
    const auto start = Clock::now();
    oryx::parallel_for(built_, end, threads, [&](std::size_t i) { index_->add(i, base_.row(i)); });
    build_ms_ += ms_since(start);
    built_ = end;
    return built_;
  }

  // One ef setting over every query, single-threaded: recall, throughput, latency percentiles.
  val sweep(std::size_t ef, std::size_t k) const {
    require_index();
    std::vector<double> latencies(queries_.n);
    std::vector<std::vector<oryx::SearchResult>> results(queries_.n);
    std::size_t distances = 0;
    const auto start = Clock::now();
    for (std::size_t q = 0; q < queries_.n; ++q) {
      oryx::SearchStats stats;
      const auto t0 = Clock::now();
      results[q] = index_->search(queries_.row(q), k, {.ef = ef, .stats = &stats});
      latencies[q] = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
      distances += stats.distance_computations;
    }
    const double total_ms = ms_since(start);
    val out = val::object();
    out.set("ef", ef);
    out.set("recall", truth_.empty() ? -1.0 : oryx::recall_at_k(results, truth_, std::min(k, truth_k_)));
    out.set("qps", 1000.0 * static_cast<double>(queries_.n) / total_ms);
    out.set("p50Us", percentile(latencies, 0.5));
    out.set("p99Us", percentile(latencies, 0.99));
    out.set("distancesPerQuery", static_cast<double>(distances) / static_cast<double>(queries_.n));
    return out;
  }

  // Exact search throughput on up to `max_queries` queries, single-threaded.
  double brute_force_qps(std::size_t k, std::size_t max_queries) const {
    require_data();
    const std::size_t n = std::min(max_queries, queries_.n);
    const auto start = Clock::now();
    for (std::size_t q = 0; q < n; ++q) {
      (void)oryx::brute_force_search(base_.data.data(), base_.n, base_.dim, queries_.row(q), k, oryx::Metric::L2);
    }
    return 1000.0 * static_cast<double>(n) / ms_since(start);
  }

  // Nanoseconds per L2 distance call for the SIMD128 kernel vs the scalar one.
  val kernel_bench(std::size_t dim) const {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    constexpr std::size_t kRows = 16;
    constexpr std::size_t kCalls = 400'000;
    std::vector<float> data(kRows * dim);
    for (float& x : data) x = uniform(rng);
    auto time = [&](oryx::DistanceFn fn) {
      volatile float sink = 0.0f;
      float acc = 0.0f;
      const auto start = Clock::now();
      for (std::size_t i = 0; i < kCalls; ++i) {
        acc += fn(data.data() + (i % kRows) * dim, data.data() + ((i * 7 + 3) % kRows) * dim, dim);
      }
      const double ns = std::chrono::duration<double, std::nano>(Clock::now() - start).count() / kCalls;
      sink = acc;
      (void)sink;
      return ns;
    };
    val out = val::object();
    out.set("dim", dim);
    out.set("scalarNs", time(oryx::get_distance_fn(oryx::Metric::L2, oryx::SimdLevel::Scalar)));
    out.set("simdNs", time(oryx::get_distance_fn(oryx::Metric::L2)));
    return out;
  }

  // Full traversal of one query, as columns (struct of arrays) for cheap transfer to JS.
  val trace(std::size_t query, std::size_t k, std::size_t ef) const {
    require_index();
    if (query >= queries_.n) throw std::out_of_range("query index out of range");
    oryx::SearchTrace trace;
    oryx::SearchStats stats;
    const auto results = index_->search(queries_.row(query), k, {.ef = ef, .stats = &stats, .trace = &trace});
    const std::size_t n = trace.events.size();
    std::vector<std::uint8_t> layer(n), accepted(n);
    std::vector<std::uint32_t> from(n), to(n);
    std::vector<float> distance(n);
    for (std::size_t i = 0; i < n; ++i) {
      const auto& e = trace.events[i];
      layer[i] = static_cast<std::uint8_t>(e.layer);
      from[i] = static_cast<std::uint32_t>(e.from);
      to[i] = static_cast<std::uint32_t>(e.to);
      distance[i] = e.distance;
      accepted[i] = e.accepted ? 1 : 0;
    }
    std::vector<std::uint32_t> result_labels;
    for (const auto& r : results) result_labels.push_back(static_cast<std::uint32_t>(r.label));
    val out = val::object();
    out.set("entryLabel", static_cast<std::uint32_t>(trace.entry_label));
    out.set("entryLayer", trace.entry_layer);
    out.set("layer", copy_to_js("Uint8Array", layer));
    out.set("from", copy_to_js("Uint32Array", from));
    out.set("to", copy_to_js("Uint32Array", to));
    out.set("distance", copy_to_js("Float32Array", distance));
    out.set("accepted", copy_to_js("Uint8Array", accepted));
    out.set("results", copy_to_js("Uint32Array", result_labels));
    out.set("distanceComputations", stats.distance_computations);
    return out;
  }

  val info() const {
    val out = val::object();
    out.set("simd", std::string(oryx::to_string(oryx::active_simd_level())));
    out.set("threads", std::thread::hardware_concurrency());
    out.set("n", base_.n);
    out.set("dim", base_.dim);
    out.set("queries", queries_.n);
    out.set("built", built_);
    out.set("buildMs", build_ms_);
    if (index_) {
      out.set("maxLevel", index_->max_level());
      out.set("memoryBytes", static_cast<double>(index_->memory_bytes()));
    }
    return out;
  }

 private:
  void require_data() const {
    if (base_.n == 0) throw std::logic_error("generate() first");
  }
  void require_index() const {
    if (!index_) throw std::logic_error("begin_build() first");
  }

  oryx::Dataset base_;
  oryx::Dataset queries_;
  std::vector<std::vector<std::uint32_t>> truth_;
  std::size_t truth_k_ = 0;
  std::unique_ptr<oryx::HnswIndex> index_;
  std::size_t built_ = 0;
  double build_ms_ = 0.0;
};

}  // namespace

EMSCRIPTEN_BINDINGS(oryx) {
  emscripten::class_<Lab>("Lab")
      .constructor<>()
      .function("generate", &Lab::generate)
      .function("computeTruth", &Lab::compute_truth)
      .function("beginBuild", &Lab::begin_build)
      .function("buildStep", &Lab::build_step)
      .function("sweep", &Lab::sweep)
      .function("bruteForceQps", &Lab::brute_force_qps)
      .function("kernelBench", &Lab::kernel_bench)
      .function("trace", &Lab::trace)
      .function("info", &Lab::info);
}
