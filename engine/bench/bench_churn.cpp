// Churn benchmark: how search holds up while the index changes underneath it.
//
//  1. Mixed load: R reader threads search while W writer threads delete and re-insert vectors
//     (same vector, same label, so the ground truth stays valid). Reports search latency
//     percentiles, throughput and recall, next to the same readers on an idle index.
//  2. Delete/re-insert cycles: each cycle removes a random share of the vectors and inserts them
//     again. Tombstones keep routing searches, so recall should hold while memory grows; then
//     compacted() rebuilds the graph without them.
//
//   bench_churn --synthetic --n 200000
//   bench_churn --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs --json churn.json

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "epzero/brute_force.h"
#include "epzero/dataset.h"
#include "epzero/hnsw.h"

using namespace epzero;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
  std::string base, query, gt, json;
  bool synthetic = false;
  std::size_t n = 0, num_queries = 1'000, dim = 128, k = 10, threads = 0;
  std::size_t M = 16, ef_construction = 200, ef = 80;
  std::size_t readers = 4, writers = 2;
  double seconds = 10.0;         // length of each mixed-load phase
  std::size_t cycles = 5;        // delete/re-insert cycles
  double churn = 0.1;            // share of the vectors replaced per cycle
};

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
      return argv[++i];
    };
    if (arg == "--synthetic") o.synthetic = true;
    else if (arg == "--base") o.base = value();
    else if (arg == "--query") o.query = value();
    else if (arg == "--gt") o.gt = value();
    else if (arg == "--json") o.json = value();
    else if (arg == "--n") o.n = std::stoul(value());
    else if (arg == "--queries") o.num_queries = std::stoul(value());
    else if (arg == "--dim") o.dim = std::stoul(value());
    else if (arg == "--k") o.k = std::stoul(value());
    else if (arg == "--threads") o.threads = std::stoul(value());
    else if (arg == "--M") o.M = std::stoul(value());
    else if (arg == "--efc") o.ef_construction = std::stoul(value());
    else if (arg == "--ef") o.ef = std::stoul(value());
    else if (arg == "--readers") o.readers = std::stoul(value());
    else if (arg == "--writers") o.writers = std::stoul(value());
    else if (arg == "--seconds") o.seconds = std::stod(value());
    else if (arg == "--cycles") o.cycles = std::stoul(value());
    else if (arg == "--churn") o.churn = std::stod(value());
    else throw std::invalid_argument("unknown argument " + arg);
  }
  if (!o.synthetic && (o.base.empty() || o.query.empty())) {
    throw std::invalid_argument("pass --synthetic or both --base and --query");
  }
  if (!o.gt.empty() && o.n != 0) throw std::invalid_argument("--gt is only valid for the full base set");
  if (o.synthetic && o.n == 0) o.n = 200'000;
  if (o.threads == 0) o.threads = std::max(1u, std::thread::hardware_concurrency());
  return o;
}

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

double percentile(std::vector<double>& sorted, double p) {
  if (sorted.empty()) return 0.0;
  return sorted[static_cast<std::size_t>(p * static_cast<double>(sorted.size() - 1))];
}

void build(HnswIndex& index, const Dataset& base, std::size_t threads) {
  std::atomic<std::size_t> next{0};
  std::vector<std::thread> workers;
  for (std::size_t t = 0; t < threads; ++t) {
    workers.emplace_back([&] {
      for (std::size_t i = next++; i < base.n; i = next++) index.add(i, base.row(i));
    });
  }
  for (auto& w : workers) w.join();
}

double recall_of(const HnswIndex& index, const Dataset& queries, const std::vector<std::vector<std::uint32_t>>& truth,
                 std::size_t k, std::size_t ef, double* qps = nullptr) {
  std::vector<std::vector<SearchResult>> results(queries.n);
  const auto start = Clock::now();
  for (std::size_t q = 0; q < queries.n; ++q) results[q] = index.search(queries.row(q), k, {.ef = ef});
  if (qps != nullptr) *qps = queries.n / seconds_since(start);
  return recall_at_k(results, truth, k);
}

struct MixedResult {
  std::size_t writers;
  double seconds, search_qps, writes_per_s, recall, p50_us, p99_us, p999_us;
};

// R readers search round-robin through the queries for `seconds`; W writers meanwhile remove and
// re-insert random labels. Returns latency percentiles over every search in the window.
MixedResult mixed(HnswIndex& index, const Dataset& base, const Dataset& queries,
                  const std::vector<std::vector<std::uint32_t>>& truth, const Options& opt, std::size_t writers) {
  std::atomic<bool> stop{false};
  std::atomic<std::size_t> writes{0};
  std::vector<std::vector<double>> latencies(opt.readers);
  std::vector<std::size_t> hits(opt.readers, 0), scored(opt.readers, 0);
  std::vector<std::thread> threads;
  for (std::size_t r = 0; r < opt.readers; ++r) {
    threads.emplace_back([&, r] {
      auto& lat = latencies[r];
      lat.reserve(1 << 20);
      for (std::size_t i = r; !stop.load(std::memory_order_relaxed); i += opt.readers) {
        const std::size_t q = i % queries.n;
        const auto start = Clock::now();
        const auto found = index.search(queries.row(q), opt.k, {.ef = opt.ef});
        lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        for (const auto& hit : found) {
          for (std::size_t t = 0; t < opt.k; ++t) {
            if (truth[q][t] == hit.label) {
              ++hits[r];
              break;
            }
          }
        }
        scored[r] += opt.k;
      }
    });
  }
  for (std::size_t w = 0; w < writers; ++w) {
    threads.emplace_back([&, w] {
      std::mt19937_64 rng(1000 + w);
      std::uniform_int_distribution<std::size_t> pick(0, base.n - 1);
      // Every re-insert takes a new slot until compaction; stop writing before the index is full.
      const std::size_t limit = index.params().max_elements - 2 * opt.writers;
      while (!stop.load(std::memory_order_relaxed) && index.size() < limit) {
        const std::size_t label = pick(rng);
        if (!index.remove(label)) continue;  // another writer has it mid-flight
        index.add(label, base.row(label));
        writes.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  const auto start = Clock::now();
  std::this_thread::sleep_for(std::chrono::duration<double>(opt.seconds));
  stop = true;
  for (auto& t : threads) t.join();
  const double elapsed = seconds_since(start);

  std::vector<double> all;
  std::size_t total_hits = 0, total_scored = 0;
  for (std::size_t r = 0; r < opt.readers; ++r) {
    all.insert(all.end(), latencies[r].begin(), latencies[r].end());
    total_hits += hits[r];
    total_scored += scored[r];
  }
  std::sort(all.begin(), all.end());
  return {writers, elapsed, all.size() / elapsed, writes.load() / elapsed,
          total_scored ? static_cast<double>(total_hits) / total_scored : 0.0, percentile(all, 0.5),
          percentile(all, 0.99), percentile(all, 0.999)};
}

struct CycleResult {
  std::size_t cycle;
  double recall, qps;
  std::size_t slots;  // node slots in use, live or tombstoned
  double deleted_share;
  bool compacted;
};

}  // namespace

int main(int argc, char** argv) try {
  const Options opt = parse(argc, argv);

  Dataset base, queries;
  std::string dataset_name;
  if (opt.synthetic) {
    auto data = make_clustered(opt.n, opt.num_queries, opt.dim, 100, 42);
    base = std::move(data.base);
    queries = std::move(data.queries);
    dataset_name = "synthetic-clustered";
  } else {
    base = read_fvecs(opt.base, opt.n);
    queries = read_fvecs(opt.query, opt.num_queries);
    dataset_name = opt.base;
  }
  const std::size_t dim = base.dim;
  const auto truth = opt.gt.empty()
                         ? compute_ground_truth(base.data.data(), base.n, dim, queries.data.data(), queries.n, opt.k,
                                                Metric::L2, opt.threads)
                         : read_ivecs(opt.gt, queries.n);
  std::printf("dataset   %s: %zu x %zu, %zu queries, ef=%zu, %zu readers\n", dataset_name.c_str(), base.n, dim,
              queries.n, opt.ef, opt.readers);

  // Room for the re-inserts (tombstoned slots are only reclaimed by compaction): the cycles, plus
  // up to one full set of vectors for the mixed-load writers, who stop if they reach it.
  const auto capacity =
      static_cast<std::size_t>(static_cast<double>(base.n) * (2.0 + opt.churn * static_cast<double>(opt.cycles))) + 1000;
  HnswParams params{.dim = dim, .metric = Metric::L2, .max_elements = capacity, .M = opt.M,
                    .ef_construction = opt.ef_construction};
  auto index = std::make_unique<HnswIndex>(params);
  auto t0 = Clock::now();
  build(*index, base, opt.threads);
  std::printf("build     %.1fs on %zu threads\n", seconds_since(t0), opt.threads);

  // 1. Mixed load: the same readers, with and without writers.
  std::vector<MixedResult> mixed_rows;
  std::printf("\nmixed load (%.0fs each)\n  %-8s %12s %12s %8s %9s %9s %10s\n", opt.seconds, "writers", "search QPS",
              "writes/s", "recall", "p50 us", "p99 us", "p99.9 us");
  for (const std::size_t writers : {std::size_t{0}, opt.writers}) {
    const auto row = mixed(*index, base, queries, truth, opt, writers);
    std::printf("  %-8zu %12.0f %12.0f %8.4f %9.1f %9.1f %10.1f\n", row.writers, row.search_qps, row.writes_per_s,
                row.recall, row.p50_us, row.p99_us, row.p999_us);
    mixed_rows.push_back(row);
  }

  // 2. Delete/re-insert cycles, then compaction.
  std::vector<CycleResult> cycles;
  std::mt19937_64 rng(7);
  const auto measure = [&](std::size_t cycle, bool compacted) {
    double qps = 0;
    const double recall = recall_of(*index, queries, truth, opt.k, opt.ef, &qps);
    const double deleted = static_cast<double>(index->deleted_count()) / static_cast<double>(index->size());
    cycles.push_back({cycle, recall, qps, index->size(), deleted, compacted});
    const std::string name = compacted ? "compacted" : (cycle == 0 ? "start" : std::to_string(cycle));
    std::printf("  %-9s %8.4f %10.0f %10zu %9.1f%%\n", name.c_str(), recall, qps, index->size(), deleted * 100.0);
  };
  std::printf("\ndelete/re-insert cycles on a fresh index (%.0f%% of the vectors each)\n  %-9s %8s %10s %10s %10s\n", opt.churn * 100,
              "cycle", "recall", "QPS", "slots", "tombstones");
  index = std::make_unique<HnswIndex>(params);
  build(*index, base, opt.threads);
  measure(0, false);
  std::vector<std::size_t> labels(base.n);
  for (std::size_t i = 0; i < base.n; ++i) labels[i] = i;
  for (std::size_t c = 1; c <= opt.cycles; ++c) {
    std::shuffle(labels.begin(), labels.end(), rng);
    const auto count = static_cast<std::size_t>(opt.churn * static_cast<double>(base.n));
    for (std::size_t i = 0; i < count; ++i) index->remove(labels[i]);
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> workers;
    for (std::size_t t = 0; t < opt.threads; ++t) {
      workers.emplace_back([&] {
        for (std::size_t i = next++; i < count; i = next++) index->add(labels[i], base.row(labels[i]));
      });
    }
    for (auto& w : workers) w.join();
    measure(c, false);
  }
  t0 = Clock::now();
  index = index->compacted(opt.threads);
  const double compact_s = seconds_since(t0);
  measure(opt.cycles, true);
  std::printf("  (compaction took %.1fs on %zu threads)\n", compact_s, opt.threads);

  if (!opt.json.empty()) {
    std::ofstream out(opt.json);
    out << "{\n  \"dataset\": \"" << dataset_name << "\", \"n\": " << base.n << ", \"dim\": " << dim
        << ", \"queries\": " << queries.n << ", \"k\": " << opt.k << ", \"ef\": " << opt.ef
        << ", \"readers\": " << opt.readers << ", \"simd\": \"" << to_string(index->simd_level())
        << "\",\n  \"mixed\": [\n";
    for (std::size_t i = 0; i < mixed_rows.size(); ++i) {
      const auto& r = mixed_rows[i];
      out << "    {\"writers\": " << r.writers << ", \"seconds\": " << r.seconds << ", \"search_qps\": " << r.search_qps
          << ", \"writes_per_s\": " << r.writes_per_s << ", \"recall\": " << r.recall << ", \"p50_us\": " << r.p50_us
          << ", \"p99_us\": " << r.p99_us << ", \"p999_us\": " << r.p999_us << "}"
          << (i + 1 < mixed_rows.size() ? "," : "") << "\n";
    }
    out << "  ],\n  \"churn\": " << opt.churn << ", \"compaction_seconds\": " << compact_s << ",\n  \"cycles\": [\n";
    for (std::size_t i = 0; i < cycles.size(); ++i) {
      const auto& r = cycles[i];
      out << "    {\"cycle\": " << r.cycle << ", \"compacted\": " << (r.compacted ? "true" : "false")
          << ", \"recall\": " << r.recall << ", \"qps\": " << r.qps << ", \"slots\": " << r.slots
          << ", \"tombstone_share\": " << r.deleted_share << "}" << (i + 1 < cycles.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("\nwrote %s\n", opt.json.c_str());
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
