// Per-query effort: a fixed beam width for every query against patience, which stops each query
// once its best k have stopped changing (SearchOptions::patience, with ef as a cap).
//
// A fixed ef spends the same effort on every query: too much on easy ones, too little on the
// hardest. For each setting this reports mean recall@k and mean distance computations (the cost
// curve), and the tail: the share of queries answered badly (recall@k below 0.8 and 0.5), the
// 99th-percentile cost, and single-thread throughput.
//
//   bench_adaptive --base sift_base.fvecs --query sift_query.fvecs --truth sift_groundtruth.ivecs \
//                  --index sift.idx --json adaptive.json
//   (--metric cosine for angular datasets such as GloVe; ground truth as ivecs, top-k first)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "oryx/dataset.h"
#include "oryx/hnsw.h"
#include "oryx/parallel.h"

using namespace oryx;

namespace {

struct Options {
  std::string base, query, truth, index, json, name = "dataset";
  Metric metric = Metric::L2;
  std::size_t k = 10, M = 16, ef_construction = 200, threads = 0, queries = 0, rounds = 3;
  std::vector<std::size_t> efs = {10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
  std::vector<std::size_t> patiences = {2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
  std::vector<std::size_t> caps = {256, 1024};
};

std::vector<std::size_t> list(const std::string& s) {
  std::vector<std::size_t> out;
  std::size_t start = 0;
  while (start < s.size()) {
    const std::size_t comma = s.find(',', start);
    out.push_back(std::stoul(s.substr(start, comma - start)));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
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
    if (arg == "--base") o.base = value();
    else if (arg == "--query") o.query = value();
    else if (arg == "--truth") o.truth = value();
    else if (arg == "--index") o.index = value();
    else if (arg == "--json") o.json = value();
    else if (arg == "--name") o.name = value();
    else if (arg == "--metric") o.metric = value() == "cosine" ? Metric::Cosine : Metric::L2;
    else if (arg == "--k") o.k = std::stoul(value());
    else if (arg == "--M") o.M = std::stoul(value());
    else if (arg == "--efc") o.ef_construction = std::stoul(value());
    else if (arg == "--threads") o.threads = std::stoul(value());
    else if (arg == "--queries") o.queries = std::stoul(value());
    else if (arg == "--rounds") o.rounds = std::stoul(value());
    else if (arg == "--efs") o.efs = list(value());
    else if (arg == "--patience") o.patiences = list(value());
    else if (arg == "--caps") o.caps = list(value());
    else throw std::invalid_argument("unknown argument " + arg);
  }
  if (o.base.empty() || o.query.empty() || o.truth.empty()) throw std::invalid_argument("pass --base, --query and --truth");
  if (o.threads == 0) o.threads = std::max(1u, std::thread::hardware_concurrency());
  return o;
}

struct Row {
  std::string policy;  // "fixed" or "patience"
  std::size_t ef, patience;
  double recall, distances, p99_distances, below_08, below_05, qps;
};

}  // namespace

int main(int argc, char** argv) try {
  const Options opt = parse(argc, argv);
  const Dataset queries = read_fvecs(opt.query, opt.queries);
  const auto truth = read_ivecs(opt.truth, queries.n);
  const std::size_t nq = queries.n, k = opt.k;

  std::unique_ptr<HnswIndex> index;
  if (!opt.index.empty() && std::filesystem::exists(opt.index)) {
    index = HnswIndex::load(opt.index);
    std::printf("loaded %s (%zu vectors)\n", opt.index.c_str(), index->size());
  } else {
    const Dataset base = read_fvecs(opt.base);
    index = std::make_unique<HnswIndex>(HnswParams{.dim = base.dim, .metric = opt.metric, .max_elements = base.n,
                                                   .M = opt.M, .ef_construction = opt.ef_construction});
    const auto start = std::chrono::steady_clock::now();
    parallel_for(0, base.n, opt.threads, [&](std::size_t i) { index->add(i, base.row(i)); });
    std::printf("built %zu x %zu in %.1f s\n", base.n, base.dim,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    if (!opt.index.empty()) index->save(opt.index);
  }

  // One setting: per-query recall and cost, then throughput as the best of `rounds` timed passes.
  const auto run = [&](const std::string& policy, std::size_t ef, std::size_t patience) {
    std::vector<double> recall(nq), cost(nq);
    for (std::size_t q = 0; q < nq; ++q) {
      SearchStats stats;
      const auto got = index->search(queries.row(q), k, {.ef = ef, .patience = patience, .stats = &stats});
      std::size_t hits = 0;
      for (std::size_t t = 0; t < k && t < truth[q].size(); ++t) {
        hits += std::any_of(got.begin(), got.end(), [&](const SearchResult& r) { return r.label == truth[q][t]; });
      }
      recall[q] = static_cast<double>(hits) / static_cast<double>(k);
      cost[q] = static_cast<double>(stats.distance_computations);
    }
    double best = 0;
    for (std::size_t r = 0; r < opt.rounds; ++r) {
      const auto start = std::chrono::steady_clock::now();
      for (std::size_t q = 0; q < nq; ++q) index->search(queries.row(q), k, {.ef = ef, .patience = patience});
      best = std::max(best, static_cast<double>(nq) / std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
    }
    Row row{policy, ef, patience, 0, 0, 0, 0, 0, best};
    for (std::size_t q = 0; q < nq; ++q) {
      row.recall += recall[q] / static_cast<double>(nq);
      row.distances += cost[q] / static_cast<double>(nq);
      row.below_08 += (recall[q] < 0.8) / static_cast<double>(nq);
      row.below_05 += (recall[q] < 0.5) / static_cast<double>(nq);
    }
    std::sort(cost.begin(), cost.end());
    row.p99_distances = cost[std::min(nq - 1, nq * 99 / 100)];
    std::printf("%-8s ef %5zu  patience %4zu  recall %.4f  dist %7.0f  p99 dist %7.0f  <0.8 %5.2f%%  <0.5 %5.2f%%  qps %8.0f\n",
                policy.c_str(), ef, patience, row.recall, row.distances, row.p99_distances, 100 * row.below_08,
                100 * row.below_05, row.qps);
    return row;
  };

  std::vector<Row> rows;
  for (const auto ef : opt.efs) rows.push_back(run("fixed", ef, 0));
  for (const auto cap : opt.caps) {
    for (const auto p : opt.patiences) rows.push_back(run("patience", cap, p));
  }

  if (!opt.json.empty()) {
    std::ofstream out(opt.json);
    out << "{\n  \"dataset\": \"" << opt.name << "\", \"n\": " << index->size() << ", \"dim\": " << index->params().dim
        << ", \"queries\": " << nq << ", \"k\": " << k << ", \"M\": " << index->params().M
        << ", \"ef_construction\": " << index->params().ef_construction << ",\n  \"rows\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const Row& r = rows[i];
      out << "    {\"policy\": \"" << r.policy << "\", \"ef\": " << r.ef << ", \"patience\": " << r.patience
          << ", \"recall\": " << r.recall << ", \"distances\": " << r.distances << ", \"p99_distances\": " << r.p99_distances
          << ", \"below_0.8\": " << r.below_08 << ", \"below_0.5\": " << r.below_05 << ", \"qps\": " << r.qps << "}"
          << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("wrote %s\n", opt.json.c_str());
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
