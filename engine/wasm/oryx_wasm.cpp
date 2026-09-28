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

// Converts a search trace into columns (struct of arrays) plus results, for cheap transfer to JS.
val trace_to_js(const oryx::SearchTrace& trace, const std::vector<oryx::SearchResult>& results,
                const oryx::SearchStats& stats, double micros) {
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
  std::vector<std::uint32_t> labels;
  std::vector<float> scores;
  for (const auto& r : results) {
    labels.push_back(static_cast<std::uint32_t>(r.label));
    scores.push_back(r.distance);
  }
  val out = val::object();
  out.set("entryLabel", static_cast<std::uint32_t>(trace.entry_label));
  out.set("entryLayer", trace.entry_layer);
  out.set("layer", copy_to_js("Uint8Array", layer));
  out.set("from", copy_to_js("Uint32Array", from));
  out.set("to", copy_to_js("Uint32Array", to));
  out.set("distance", copy_to_js("Float32Array", distance));
  out.set("accepted", copy_to_js("Uint8Array", accepted));
  out.set("results", copy_to_js("Uint32Array", labels));
  out.set("resultDistances", copy_to_js("Float32Array", scores));
  out.set("distanceComputations", stats.distance_computations);
  out.set("micros", micros);
  return out;
}

// A real dataset for the visualizer: int8 vectors with per-dimension scales (as exported by
// pipeline/build_dataset.py), indexed with cosine distance. Labels are row numbers. The
// dequantized, normalized vectors are kept so every search can also be checked against an exact
// scan, which the page shows next to the HNSW result.
class Collection {
 public:
  void load(const val& int8_vectors, const val& scales, std::size_t dim) {
    const auto codes = emscripten::convertJSArrayToNumberVector<std::int8_t>(int8_vectors);
    const auto scale = emscripten::convertJSArrayToNumberVector<float>(scales);
    if (dim == 0 || scale.size() != dim || codes.size() % dim != 0) throw std::invalid_argument("bad vector data");
    dim_ = dim;
    n_ = codes.size() / dim;
    floats_.resize(codes.size());
    for (std::size_t i = 0; i < codes.size(); ++i) floats_[i] = codes[i] * (scale[i % dim] / 127.0f);
    for (std::size_t i = 0; i < n_; ++i) oryx::normalize(floats_.data() + i * dim_, dim_);
    index_.reset();
    built_ = 0;
  }

  void begin_build(std::size_t M, std::size_t ef_construction) {
    if (n_ == 0) throw std::logic_error("load() first");
    index_ = std::make_unique<oryx::HnswIndex>(oryx::HnswParams{
        .dim = dim_, .metric = oryx::Metric::Cosine, .max_elements = n_, .M = M, .ef_construction = ef_construction});
    built_ = 0;
  }

  std::size_t build_step(std::size_t count, std::size_t threads) {
    if (!index_) throw std::logic_error("begin_build() first");
    const std::size_t end = std::min(n_, built_ + count);
    oryx::parallel_for(built_, end, threads, [&](std::size_t i) { index_->add(i, vector(i)); });
    built_ = end;
    return built_;
  }

  // Top level of every node (0 = base layer only).
  val levels() const {
    require_index();
    std::vector<std::uint8_t> out(built_);
    for (std::size_t i = 0; i < built_; ++i) out[i] = static_cast<std::uint8_t>(index_->level_of(i));
    return copy_to_js("Uint8Array", out);
  }

  // Edges of one layer as a flat [a0, b0, a1, b1, ...] list, each undirected edge once.
  val edges(int layer) const {
    require_index();
    std::vector<std::uint32_t> pairs;
    for (std::size_t i = 0; i < built_; ++i) {
      for (const std::uint64_t j : index_->neighbors_of(i, layer)) {
        // Keep an edge once: from the lower id, or from either end if the link is one-way.
        const auto back = index_->neighbors_of(j, layer);
        const bool mutual = std::find(back.begin(), back.end(), i) != back.end();
        if (!mutual || i < j) {
          pairs.push_back(static_cast<std::uint32_t>(i));
          pairs.push_back(static_cast<std::uint32_t>(j));
        }
      }
    }
    return copy_to_js("Uint32Array", pairs);
  }

  val search(const val& query, std::size_t k, std::size_t ef) const {
    const auto q = emscripten::convertJSArrayToNumberVector<float>(query);
    if (q.size() != dim_) throw std::invalid_argument("query has the wrong dimension");
    return traced(q.data(), k, ef);
  }

  // "More like this": search with a stored photo's own vector.
  val search_by_id(std::size_t id, std::size_t k, std::size_t ef) const {
    if (id >= n_) throw std::out_of_range("id out of range");
    return traced(vector(id), k, ef);
  }

  val info() const {
    val out = val::object();
    out.set("count", n_);
    out.set("dim", dim_);
    out.set("built", built_);
    if (index_) {
      out.set("maxLevel", index_->max_level());
      out.set("memoryBytes", static_cast<double>(index_->memory_bytes()));
    }
    return out;
  }

 private:
  const float* vector(std::size_t i) const { return floats_.data() + i * dim_; }

  val traced(const float* query, std::size_t k, std::size_t ef) const {
    require_index();
    // Time the search without tracing (recording every hop costs more than the search itself),
    // taking the best of a few runs because a single query is near the timer's resolution.
    double micros = 1e18;
    for (int run = 0; run < 5; ++run) {
      const auto start = Clock::now();
      (void)index_->search(query, k, {.ef = ef});
      micros = std::min(micros, std::chrono::duration<double, std::micro>(Clock::now() - start).count());
    }
    oryx::SearchTrace trace;
    oryx::SearchStats stats;
    const auto results = index_->search(query, k, {.ef = ef, .stats = &stats, .trace = &trace});
    val out = trace_to_js(trace, results, stats, micros);

    // The same query as an exact scan over every vector, for recall and speedup.
    std::vector<oryx::SearchResult> exact;
    double exact_micros = 1e18;
    for (int run = 0; run < 3; ++run) {  // best of 3, like the HNSW timing above
      const auto exact_start = Clock::now();
      exact = oryx::brute_force_search(floats_.data(), built_, dim_, query, k, oryx::Metric::Cosine);
      exact_micros = std::min(exact_micros, std::chrono::duration<double, std::micro>(Clock::now() - exact_start).count());
    }
    std::vector<std::uint32_t> exact_labels;
    for (const auto& r : exact) exact_labels.push_back(static_cast<std::uint32_t>(r.label));
    out.set("exact", copy_to_js("Uint32Array", exact_labels));
    out.set("exactMicros", exact_micros);
    return out;
  }

  void require_index() const {
    if (!index_) throw std::logic_error("begin_build() first");
  }

  std::vector<float> floats_;  // n x dim, dequantized and normalized
  std::size_t dim_ = 0;
  std::size_t n_ = 0;
  std::unique_ptr<oryx::HnswIndex> index_;
  std::size_t built_ = 0;
};

}  // namespace

EMSCRIPTEN_BINDINGS(oryx) {
  emscripten::class_<Collection>("Collection")
      .constructor<>()
      .function("load", &Collection::load)
      .function("beginBuild", &Collection::begin_build)
      .function("buildStep", &Collection::build_step)
      .function("levels", &Collection::levels)
      .function("edges", &Collection::edges)
      .function("search", &Collection::search)
      .function("searchById", &Collection::search_by_id)
      .function("info", &Collection::info);

  emscripten::class_<Lab>("Lab")
      .constructor<>()
      .function("generate", &Lab::generate)
      .function("computeTruth", &Lab::compute_truth)
      .function("beginBuild", &Lab::begin_build)
      .function("buildStep", &Lab::build_step)
      .function("sweep", &Lab::sweep)
      .function("bruteForceQps", &Lab::brute_force_qps)
      .function("kernelBench", &Lab::kernel_bench)
      .function("info", &Lab::info);
}
