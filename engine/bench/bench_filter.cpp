// Filtered search benchmark: for each filter selectivity (the share of vectors a filter allows),
// measures recall@k and single-threaded QPS for every strategy against exact filtered ground
// truth, and records which strategy Auto picks.
//
//   bench_filter --synthetic --n 100000
//   bench_filter --base sift_base.fvecs --query sift_query.fvecs --load sift.epzero --json filter.json
//
// Filters are random subsets of the labels (a fixed seed per selectivity), so which vectors pass
// is uncorrelated with where they sit in the vector space: the case a graph walk finds hardest.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
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
  std::string base, query, json, load;
  bool synthetic = false;
  std::size_t n = 0, num_queries = 1'000, dim = 128, k = 10, threads = 0;
  std::size_t M = 16, ef_construction = 200;
  std::vector<double> selectivity = {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.9};
  std::vector<std::size_t> ef = {10, 20, 40, 80, 160, 320};
  std::vector<FilterStrategy> strategies = {FilterStrategy::Auto, FilterStrategy::Exact, FilterStrategy::Inline,
                                            FilterStrategy::TwoHop};
};

template <typename T>
std::vector<T> parse_list(const std::string& s) {
  std::vector<T> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) out.push_back(static_cast<T>(std::stod(item)));
  return out;
}

FilterStrategy parse_strategy(const std::string& name) {
  for (const auto s : {FilterStrategy::Auto, FilterStrategy::Exact, FilterStrategy::Inline, FilterStrategy::TwoHop}) {
    if (name == to_string(s)) return s;
  }
  throw std::invalid_argument("unknown strategy " + name + " (auto, exact, inline, two-hop)");
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
    else if (arg == "--json") o.json = value();
    else if (arg == "--load") o.load = value();
    else if (arg == "--n") o.n = std::stoul(value());
    else if (arg == "--queries") o.num_queries = std::stoul(value());
    else if (arg == "--dim") o.dim = std::stoul(value());
    else if (arg == "--k") o.k = std::stoul(value());
    else if (arg == "--threads") o.threads = std::stoul(value());
    else if (arg == "--M") o.M = std::stoul(value());
    else if (arg == "--efc") o.ef_construction = std::stoul(value());
    else if (arg == "--selectivity") o.selectivity = parse_list<double>(value());
    else if (arg == "--ef") o.ef = parse_list<std::size_t>(value());
    else if (arg == "--strategies") {
      o.strategies.clear();
      std::stringstream ss(value());
      for (std::string item; std::getline(ss, item, ',');) o.strategies.push_back(parse_strategy(item));
    } else {
      throw std::invalid_argument("unknown argument " + arg);
    }
  }
  if (!o.synthetic && (o.base.empty() || o.query.empty())) {
    throw std::invalid_argument("pass --synthetic or both --base and --query");
  }
  if (o.synthetic && o.n == 0) o.n = 100'000;
  if (o.threads == 0) o.threads = std::max(1u, std::thread::hardware_concurrency());
  return o;
}

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(p * static_cast<double>(values.size() - 1))];
}

struct Row {
  double selectivity;
  std::size_t allowed;
  FilterStrategy strategy;
  std::size_t ef;
  double recall, qps, p50_us, p99_us, avg_distances;
  std::map<std::string, std::size_t> used;  // Auto: how many queries each strategy answered
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
  std::printf("dataset   %s: %zu base x %zu dim, %zu queries\n", dataset_name.c_str(), base.n, dim, queries.n);

  std::unique_ptr<HnswIndex> index;
  if (!opt.load.empty()) {
    index = HnswIndex::load(opt.load);
    std::printf("load      %s\n", opt.load.c_str());
  } else {
    index = std::make_unique<HnswIndex>(HnswParams{
        .dim = dim, .metric = Metric::L2, .max_elements = base.n, .M = opt.M, .ef_construction = opt.ef_construction});
    const auto t0 = Clock::now();
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> builders;
    for (std::size_t t = 0; t < opt.threads; ++t) {
      builders.emplace_back([&] {
        for (std::size_t i = next++; i < base.n; i = next++) index->add(i, base.row(i));
      });
    }
    for (auto& b : builders) b.join();
    std::printf("build     %.2fs on %zu threads\n", seconds_since(t0), opt.threads);
  }

  std::vector<Row> rows;
  for (const double selectivity : opt.selectivity) {
    // A random subset of the labels, the same for every run at this selectivity.
    std::mt19937_64 rng(static_cast<std::uint64_t>(selectivity * 1e9) + 7);
    std::vector<std::uint64_t> allowed(base.n);
    for (std::size_t i = 0; i < base.n; ++i) allowed[i] = i;
    std::shuffle(allowed.begin(), allowed.end(), rng);
    allowed.resize(std::max<std::size_t>(1, static_cast<std::size_t>(selectivity * static_cast<double>(base.n))));
    std::sort(allowed.begin(), allowed.end());
    const SearchFilter filter = index->make_filter(allowed.data(), allowed.size());

    // Exact filtered ground truth: brute force over just the allowed rows.
    auto t0 = Clock::now();
    std::vector<float> rows_allowed(allowed.size() * dim);
    for (std::size_t i = 0; i < allowed.size(); ++i) {
      std::copy_n(base.row(allowed[i]), dim, rows_allowed.data() + i * dim);
    }
    auto truth = compute_ground_truth(rows_allowed.data(), allowed.size(), dim, queries.data.data(), queries.n, opt.k,
                                      Metric::L2, opt.threads);
    for (auto& t : truth) {
      for (auto& id : t) id = static_cast<std::uint32_t>(allowed[id]);
    }
    std::printf("\nselectivity %.3f: %zu allowed (truth in %.1fs)\n", selectivity, allowed.size(), seconds_since(t0));
    std::printf("  %-8s %5s %10s %10s %9s %9s %11s  %s\n", "strategy", "ef", "recall", "QPS", "p50 us", "p99 us",
                "dist/query", "auto chose");

    for (const FilterStrategy strategy : opt.strategies) {
      // Exact scans every allowed vector whatever ef is: one row is enough.
      const std::vector<std::size_t> efs =
          strategy == FilterStrategy::Exact ? std::vector<std::size_t>{opt.ef.front()} : opt.ef;
      for (const std::size_t ef : efs) {
        std::vector<std::vector<SearchResult>> results(queries.n);
        std::vector<double> latencies(queries.n);
        std::size_t distances = 0;
        std::map<std::string, std::size_t> used;
        const auto sweep = Clock::now();
        for (std::size_t q = 0; q < queries.n; ++q) {
          SearchStats stats;
          const auto start = Clock::now();
          results[q] = index->search(queries.row(q), opt.k,
                                     {.ef = ef, .stats = &stats, .filter = &filter, .strategy = strategy});
          latencies[q] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
          distances += stats.distance_computations;
          ++used[std::string(to_string(stats.strategy))];
        }
        const double qps = queries.n / seconds_since(sweep);
        Row row{selectivity, allowed.size(), strategy, ef, recall_at_k(results, truth, opt.k), qps,
                percentile(latencies, 0.5), percentile(latencies, 0.99),
                static_cast<double>(distances) / queries.n, strategy == FilterStrategy::Auto ? used : decltype(used){}};
        std::string chose;
        for (const auto& [name, count] : row.used) chose += name + ":" + std::to_string(count) + " ";
        std::printf("  %-8s %5zu %10.4f %10.0f %9.1f %9.1f %11.0f  %s\n", std::string(to_string(strategy)).c_str(), ef,
                    row.recall, row.qps, row.p50_us, row.p99_us, row.avg_distances, chose.c_str());
        rows.push_back(std::move(row));
      }
    }
  }

  if (!opt.json.empty()) {
    std::ofstream out(opt.json);
    out << "{\n  \"dataset\": \"" << dataset_name << "\", \"n\": " << base.n << ", \"dim\": " << dim
        << ", \"queries\": " << queries.n << ", \"k\": " << opt.k << ", \"simd\": \""
        << to_string(index->simd_level()) << "\",\n  \"filter\": \"random subset of the labels\",\n  \"results\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const Row& r = rows[i];
      out << "    {\"selectivity\": " << r.selectivity << ", \"allowed\": " << r.allowed << ", \"strategy\": \""
          << to_string(r.strategy) << "\", \"ef\": " << r.ef << ", \"recall\": " << r.recall << ", \"qps\": " << r.qps
          << ", \"p50_us\": " << r.p50_us << ", \"p99_us\": " << r.p99_us
          << ", \"distances_per_query\": " << r.avg_distances;
      if (!r.used.empty()) {
        out << ", \"auto_chose\": {";
        std::size_t j = 0;
        for (const auto& [name, count] : r.used) out << (j++ ? ", " : "") << "\"" << name << "\": " << count;
        out << "}";
      }
      out << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("\nwrote %s\n", opt.json.c_str());
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
