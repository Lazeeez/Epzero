// End-to-end benchmark: builds an index, then sweeps ef and reports recall@k against
// single-threaded QPS and latency percentiles, next to an exact brute-force baseline.
//
//   bench_recall --synthetic --n 100000 --dim 128
//   bench_recall --base sift_base.fvecs --query sift_query.fvecs --gt sift_groundtruth.ivecs
//
// Pass --json <file> to write machine-readable results (used by CI and the benchmarks page).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "oryx/brute_force.h"
#include "oryx/dataset.h"
#include "oryx/hnsw.h"

using namespace oryx;
using Clock = std::chrono::steady_clock;

namespace {

struct Options {
  std::string base, query, gt, json, save, load;
  bool synthetic = false;
  bool baseline = true;
  std::size_t n = 0;  // 0 = whole file (synthetic defaults to 100k)
  std::size_t num_queries = 1'000, dim = 128, clusters = 100;
  std::size_t M = 16, ef_construction = 200, k = 10, threads = 0;
  std::size_t search_threads = 1;  // >1 measures aggregate throughput of concurrent queries
  std::vector<std::size_t> ef = {10, 20, 40, 80, 160, 320};
  Metric metric = Metric::L2;
};

std::vector<std::size_t> parse_list(const std::string& s) {
  std::vector<std::size_t> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) out.push_back(std::stoul(item));
  return out;
}

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
    else if (arg == "--save") o.save = value();
    else if (arg == "--load") o.load = value();
    else if (arg == "--no-baseline") o.baseline = false;
    else if (arg == "--n") o.n = std::stoul(value());
    else if (arg == "--queries") o.num_queries = std::stoul(value());
    else if (arg == "--dim") o.dim = std::stoul(value());
    else if (arg == "--clusters") o.clusters = std::stoul(value());
    else if (arg == "--M") o.M = std::stoul(value());
    else if (arg == "--efc") o.ef_construction = std::stoul(value());
    else if (arg == "--k") o.k = std::stoul(value());
    else if (arg == "--threads") o.threads = std::stoul(value());
    else if (arg == "--search-threads") o.search_threads = std::stoul(value());
    else if (arg == "--ef") o.ef = parse_list(value());
    else if (arg == "--metric") {
      const std::string m = value();
      o.metric = m == "cosine" ? Metric::Cosine : m == "ip" ? Metric::InnerProduct : Metric::L2;
    } else {
      throw std::invalid_argument("unknown argument " + arg);
    }
  }
  if (!o.synthetic && (o.base.empty() || o.query.empty())) {
    throw std::invalid_argument("pass --synthetic or both --base and --query");
  }
  if (!o.gt.empty() && o.n != 0) {
    throw std::invalid_argument("--gt is only valid for the full base set; drop --n or --gt");
  }
  if (o.synthetic && o.n == 0) o.n = 100'000;
  if (o.threads == 0) o.threads = std::max(1u, std::thread::hardware_concurrency());
  return o;
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  const auto idx = static_cast<std::size_t>(p * static_cast<double>(values.size() - 1));
  return values[idx];
}

struct Row {
  std::size_t ef;
  double recall, qps, p50_us, p99_us, avg_distances;
};

}  // namespace

int main(int argc, char** argv) try {
  const Options opt = parse(argc, argv);

  Dataset base, queries;
  std::string dataset_name;
  if (opt.synthetic) {
    auto data = make_clustered(opt.n, opt.num_queries, opt.dim, opt.clusters, 42);
    base = std::move(data.base);
    queries = std::move(data.queries);
    dataset_name = "synthetic-clustered";
  } else {
    base = read_fvecs(opt.base, opt.n);
    queries = read_fvecs(opt.query, opt.num_queries);
    dataset_name = opt.base;
  }
  if (base.dim != queries.dim) throw std::runtime_error("base and query dimensions differ");
  const std::size_t dim = base.dim;
  if (opt.metric == Metric::Cosine) {  // brute force expects normalized base vectors
    for (std::size_t i = 0; i < base.n; ++i) normalize(base.data.data() + i * dim, dim);
  }

  std::printf("dataset   %s: %zu base x %zu dim, %zu queries, metric=%s\n", dataset_name.c_str(), base.n, dim,
              queries.n, std::string(to_string(opt.metric)).c_str());

  // Ground truth. A provided file is only valid when the whole base set was loaded.
  std::vector<std::vector<std::uint32_t>> truth;
  auto t0 = Clock::now();
  if (!opt.gt.empty()) {
    truth = read_ivecs(opt.gt, queries.n);
  } else {
    truth = compute_ground_truth(base.data.data(), base.n, dim, queries.data.data(), queries.n, opt.k, opt.metric,
                                 opt.threads);
    std::printf("truth     computed by brute force in %.2fs (%zu threads)\n", seconds_since(t0), opt.threads);
  }

  // Build with all threads, or load a saved graph to benchmark search changes in isolation.
  std::unique_ptr<HnswIndex> loaded;
  double build_s = 0.0;
  if (!opt.load.empty()) {
    t0 = Clock::now();
    loaded = HnswIndex::load(opt.load);
    std::printf("load      %s in %.2fs\n", opt.load.c_str(), seconds_since(t0));
  } else {
    HnswParams params{.dim = dim, .metric = opt.metric, .max_elements = base.n, .M = opt.M,
                      .ef_construction = opt.ef_construction};
    loaded = std::make_unique<HnswIndex>(params);
    t0 = Clock::now();
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> builders;
    for (std::size_t t = 0; t < opt.threads; ++t) {
      builders.emplace_back([&] {
        for (std::size_t i = next++; i < base.n; i = next++) loaded->add(i, base.row(i));
      });
    }
    for (auto& b : builders) b.join();
    build_s = seconds_since(t0);
    std::printf("build     %.2fs, %.0f inserts/s (%zu threads, M=%zu, efc=%zu)\n", build_s, base.n / build_s,
                opt.threads, opt.M, opt.ef_construction);
  }
  HnswIndex& index = *loaded;
  std::printf("index     simd=%s, %.1f MiB, %d layers\n", std::string(to_string(index.simd_level())).c_str(),
              index.memory_bytes() / 1048576.0, index.max_level() + 1);
  if (!opt.save.empty()) index.save(opt.save);

  // Exact baseline, single-threaded, on a subset of queries to keep runtime bounded.
  double bf_qps = 0.0;
  if (opt.baseline) {
    const std::size_t bf_queries = std::min<std::size_t>(queries.n, 100);
    t0 = Clock::now();
    for (std::size_t q = 0; q < bf_queries; ++q) {
      (void)brute_force_search(base.data.data(), base.n, dim, queries.row(q), opt.k, opt.metric);
    }
    bf_qps = bf_queries / seconds_since(t0);
    std::printf("baseline  brute force: %.0f QPS (1 thread, recall 1.000)\n", bf_qps);
  }
  std::printf("\n");

  const std::string recall_header = "recall@" + std::to_string(opt.k);
  std::printf("%6s %10s %10s %10s %10s %12s %9s\n", "ef", recall_header.c_str(), "QPS", "p50 us", "p99 us",
              "dist/query", "speedup");
  std::vector<Row> rows;
  for (std::size_t ef : opt.ef) {
    std::vector<std::vector<SearchResult>> results(queries.n);
    std::vector<double> latencies(queries.n);
    std::vector<std::size_t> distances_per_query(queries.n);
    std::atomic<std::size_t> next{0};
    auto worker = [&] {
      for (std::size_t q = next++; q < queries.n; q = next++) {
        SearchStats stats;
        const auto start = Clock::now();
        results[q] = index.search(queries.row(q), opt.k, {.ef = ef, .stats = &stats});
        latencies[q] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        distances_per_query[q] = stats.distance_computations;
      }
    };
    const auto sweep_start = Clock::now();
    if (opt.search_threads <= 1) {
      worker();
    } else {
      std::vector<std::thread> searchers;
      for (std::size_t t = 0; t < opt.search_threads; ++t) searchers.emplace_back(worker);
      for (auto& s : searchers) s.join();
    }
    const double qps = queries.n / seconds_since(sweep_start);
    std::size_t distances = 0;
    for (std::size_t d : distances_per_query) distances += d;
    const Row row{ef, recall_at_k(results, truth, opt.k), qps, percentile(latencies, 0.5),
                  percentile(latencies, 0.99), static_cast<double>(distances) / queries.n};
    rows.push_back(row);
    std::printf("%6zu %10.4f %10.0f %10.1f %10.1f %12.0f", row.ef, row.recall, row.qps, row.p50_us, row.p99_us,
                row.avg_distances);
    if (bf_qps > 0) std::printf(" %8.0fx", row.qps / bf_qps);
    std::printf("\n");
  }

  if (!opt.json.empty()) {
    std::ofstream out(opt.json);
    out << "{\n  \"dataset\": \"" << dataset_name << "\", \"n\": " << base.n << ", \"dim\": " << dim
        << ", \"queries\": " << queries.n << ", \"k\": " << opt.k << ",\n  \"metric\": \"" << to_string(opt.metric)
        << "\", \"simd\": \"" << to_string(index.simd_level()) << "\", \"M\": " << opt.M
        << ", \"ef_construction\": " << opt.ef_construction << ",\n  \"build_seconds\": " << build_s
        << ", \"build_threads\": " << opt.threads << ", \"memory_bytes\": " << index.memory_bytes()
        << ", \"brute_force_qps\": " << bf_qps << ",\n  \"sweep\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const Row& r = rows[i];
      out << "    {\"ef\": " << r.ef << ", \"recall\": " << r.recall << ", \"qps\": " << r.qps
          << ", \"p50_us\": " << r.p50_us << ", \"p99_us\": " << r.p99_us
          << ", \"distances_per_query\": " << r.avg_distances << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("\nwrote %s\n", opt.json.c_str());
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
