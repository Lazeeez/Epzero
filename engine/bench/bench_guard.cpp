// Recall under drift: how three ways of choosing the beam width hold up as the workload changes.
//
// Queries stream through four phases, each a kind of drift seen in production:
//   steady    the queries the index was tuned for
//   harder    queries shifted off the data distribution (Gaussian noise), e.g. a new kind of user
//   churn     20% of the vectors deleted and re-inserted (tombstones, a rewired graph)
//   filtered  every query restricted to a random 5% of the vectors
// Policies, run on identical streams:
//   fixed-tuned   the smallest ef that meets the target on the steady phase (the usual practice)
//   fixed-worst   the smallest ef that meets the target in every phase (over-provisioned)
//   guard         RecallGuard: ef per segment, steered by sampled exact checks
// For every batch: true recall@k (against exact ground truth), distance computations per query,
// the ef in use, and the guard's own estimate with its confidence interval.
//
//   bench_guard --synthetic --n 200000
//   bench_guard --base sift_base.fvecs --query sift_query.fvecs --json guard.json

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "oryx/brute_force.h"
#include "oryx/dataset.h"
#include "oryx/hnsw.h"
#include "oryx/parallel.h"
#include "oryx/recall.h"

using namespace oryx;

namespace {

struct Options {
  std::string base, query, json;
  bool synthetic = false;
  std::size_t n = 0, num_queries = 8'000, dim = 128, k = 10, threads = 0;
  std::size_t M = 16, ef_construction = 200;
  double target = 0.95, noise = 0.0, sample_rate = 0.05, filter_share = 0.05;
  std::size_t reference_ef = 0;  // 0: exact checks
  std::size_t batch = 500, passes = 4;
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
    else if (arg == "--json") o.json = value();
    else if (arg == "--n") o.n = std::stoul(value());
    else if (arg == "--queries") o.num_queries = std::stoul(value());
    else if (arg == "--dim") o.dim = std::stoul(value());
    else if (arg == "--k") o.k = std::stoul(value());
    else if (arg == "--threads") o.threads = std::stoul(value());
    else if (arg == "--M") o.M = std::stoul(value());
    else if (arg == "--efc") o.ef_construction = std::stoul(value());
    else if (arg == "--target") o.target = std::stod(value());
    else if (arg == "--noise") o.noise = std::stod(value());
    else if (arg == "--sample-rate") o.sample_rate = std::stod(value());
    else if (arg == "--filter-share") o.filter_share = std::stod(value());
    else if (arg == "--reference-ef") o.reference_ef = std::stoul(value());
    else if (arg == "--batch") o.batch = std::stoul(value());
    else if (arg == "--passes") o.passes = std::stoul(value());
    else throw std::invalid_argument("unknown argument " + arg);
  }
  if (!o.synthetic && (o.base.empty() || o.query.empty())) throw std::invalid_argument("pass --synthetic or --base/--query");
  if (o.synthetic && o.n == 0) o.n = 200'000;
  if (o.threads == 0) o.threads = std::max(1u, std::thread::hardware_concurrency());
  return o;
}

struct Phase {
  std::string name;
  std::vector<float> queries;                   // batch x dim
  std::vector<std::vector<std::uint32_t>> truth;  // exact top-k, after the phase's changes
  bool filtered = false;
};

struct Batch {
  std::string policy, phase;
  std::size_t pass, ef;
  double recall, distances, est_recall = -1, est_lower = -1, est_upper = -1;
};

double recall_of(const std::vector<SearchResult>& got, const std::vector<std::uint32_t>& truth) {
  std::size_t hits = 0;
  for (const auto t : truth) {
    if (std::any_of(got.begin(), got.end(), [&](const SearchResult& r) { return r.label == t; })) ++hits;
  }
  return truth.empty() ? 1.0 : static_cast<double>(hits) / static_cast<double>(truth.size());
}

}  // namespace

int main(int argc, char** argv) try {
  const Options opt = parse(argc, argv);
  Dataset base, queries;
  if (opt.synthetic) {
    auto data = make_clustered(opt.n, opt.num_queries, opt.dim, 100, 42);
    base = std::move(data.base);
    queries = std::move(data.queries);
  } else {
    base = read_fvecs(opt.base, opt.n);
    queries = read_fvecs(opt.query, opt.num_queries);
  }
  const std::size_t dim = base.dim, B = opt.batch;
  if (queries.n < 4 * B) throw std::invalid_argument("need at least 4 x --batch queries");

  HnswIndex index({.dim = dim, .metric = Metric::L2, .max_elements = base.n * 2, .M = opt.M,
                   .ef_construction = opt.ef_construction});
  parallel_for(0, base.n, opt.threads, [&](std::size_t i) { index.add(i, base.row(i)); });

  // Noise for the "harder" phase: scaled to the data, a fraction of a typical neighbour distance.
  double noise = opt.noise;
  if (noise <= 0) {
    double sum = 0;
    for (std::size_t q = 0; q < 50; ++q) sum += std::sqrt(index.exact_search(queries.row(q), 1)[0].distance);
    noise = (sum / 50) / std::sqrt(static_cast<double>(dim));  // moves a query about one neighbour distance
  }

  // The random 5% the filtered phase allows.
  std::vector<std::uint64_t> allowed(base.n);
  for (std::size_t i = 0; i < base.n; ++i) allowed[i] = i;
  std::mt19937_64 rng(5);
  std::shuffle(allowed.begin(), allowed.end(), rng);
  allowed.resize(static_cast<std::size_t>(opt.filter_share * static_cast<double>(base.n)));

  // Phase queries (each its own slice of the query set) and their exact answers.
  std::vector<Phase> phases(4);
  const char* names[] = {"steady", "harder", "churn", "filtered"};
  std::normal_distribution<float> gauss(0.0f, static_cast<float>(noise));
  for (std::size_t p = 0; p < 4; ++p) {
    phases[p].name = names[p];
    phases[p].filtered = p == 3;
    phases[p].queries.assign(queries.row(p * B), queries.row(p * B) + B * dim);
    if (p == 1) {
      for (auto& x : phases[p].queries) x += gauss(rng);
    }
  }

  // Each policy replays the same stream on its own copy of the state. The index is shared, so the
  // churn is applied once before the churn phase and undone after (same vectors, same labels).
  struct Policy {
    std::string name;
    std::size_t ef;  // fixed policies
    bool guarded;
  };
  std::vector<Batch> rows;

  // Ground truth per phase (churn re-inserts the same vectors, so its truth is the plain truth).
  const auto truth_for = [&](Phase& ph, const SearchFilter* filter) {
    ph.truth.assign(B, {});
    parallel_for(0, B, opt.threads, [&](std::size_t q) {
      for (const auto& r : index.exact_search(ph.queries.data() + q * dim, opt.k, filter)) {
        ph.truth[q].push_back(static_cast<std::uint32_t>(r.label));
      }
    });
  };
  SearchFilter filter = index.make_filter(allowed.data(), allowed.size());
  for (std::size_t p = 0; p < 3; ++p) truth_for(phases[p], nullptr);
  truth_for(phases[3], &filter);

  const auto churn = [&] {
    std::vector<std::size_t> labels(base.n);
    for (std::size_t i = 0; i < base.n; ++i) labels[i] = i;
    std::mt19937_64 r(11);
    std::shuffle(labels.begin(), labels.end(), r);
    labels.resize(base.n / 5);
    for (const auto l : labels) index.remove(l);
    parallel_for(0, labels.size(), opt.threads, [&](std::size_t i) { index.add(labels[i], base.row(labels[i])); });
  };

  // Fixed policies: per-phase recall at a ladder of ef, to pick the tuned and worst-case ef.
  const std::vector<std::size_t> ladder = {16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024};
  const auto phase_recall = [&](const Phase& ph, std::size_t ef, const SearchFilter* f) {
    double sum = 0;
    for (std::size_t q = 0; q < B; ++q) {
      sum += recall_of(index.search(ph.queries.data() + q * dim, opt.k, {.ef = ef, .filter = f}), ph.truth[q]);
    }
    return sum / B;
  };
  const auto meets = [&](std::size_t ef, std::size_t p) {
    return phase_recall(phases[p], ef, p == 3 ? &filter : nullptr) >= opt.target;
  };
  const auto smallest = [&](auto&& ok) {
    for (const auto ef : ladder) {
      if (ok(ef)) return ef;
    }
    return ladder.back();
  };
  const std::size_t tuned = smallest([&](std::size_t ef) { return meets(ef, 0); });
  const std::size_t worst_static = smallest([&](std::size_t ef) { return meets(ef, 0) && meets(ef, 1) && meets(ef, 3); });
  churn();
  filter = index.make_filter(allowed.data(), allowed.size());  // the churned index has new slots
  const std::size_t worst = std::max(worst_static, smallest([&](std::size_t ef) { return meets(ef, 2); }));
  std::printf("noise %.2f; fixed-tuned ef %zu, fixed-worst ef %zu\n", noise, tuned, worst);

  for (const Policy& policy : {Policy{"fixed-tuned", tuned, false}, Policy{"fixed-worst", worst, false},
                               Policy{"guard", 0, true}}) {
    // A fresh index for each policy, so all three see the same graph in every phase.
    HnswIndex fresh({.dim = dim, .metric = Metric::L2, .max_elements = base.n * 2, .M = opt.M,
                     .ef_construction = opt.ef_construction});
    parallel_for(0, base.n, opt.threads, [&](std::size_t i) { fresh.add(i, base.row(i)); });
    RecallGuard guard(fresh, {.target = opt.target, .initial_ef = tuned, .min_ef = 16, .max_ef = 2048,
                              .min_samples = 100, .margin = 0.01,
                              .monitor = {.sample_rate = opt.sample_rate, .window = 400, .queue = 256,
                                          .reference_ef = opt.reference_ef}});
    for (std::size_t p = 0; p < 4; ++p) {
      Phase& ph = phases[p];
      if (p == 2) {  // churn the fresh index the same way
        std::vector<std::size_t> labels(base.n);
        for (std::size_t i = 0; i < base.n; ++i) labels[i] = i;
        std::mt19937_64 r(11);
        std::shuffle(labels.begin(), labels.end(), r);
        labels.resize(base.n / 5);
        for (const auto l : labels) fresh.remove(l);
        parallel_for(0, labels.size(), opt.threads, [&](std::size_t i) { fresh.add(labels[i], base.row(labels[i])); });
      }
      const SearchFilter f = fresh.make_filter(allowed.data(), allowed.size());
      const SearchFilter* active = ph.filtered ? &f : nullptr;
      for (std::size_t pass = 0; pass < opt.passes; ++pass) {
        double rsum = 0, dsum = 0;
        std::size_t ef_used = policy.ef;
        for (std::size_t q = 0; q < B; ++q) {
          const float* query = ph.queries.data() + q * dim;
          SearchStats stats;
          std::vector<SearchResult> got;
          if (policy.guarded) {
            got = guard.search(query, opt.k, active, &stats);
          } else {
            got = fresh.search(query, opt.k, {.ef = policy.ef, .stats = &stats, .filter = active});
          }
          rsum += recall_of(got, ph.truth[q]);
          dsum += static_cast<double>(stats.distance_computations);
        }
        Batch row{policy.name, ph.name, pass, ef_used, rsum / B, dsum / B};
        if (policy.guarded) {
          guard.monitor().drain();
          const std::string segment = RecallGuard::segment_of(active);
          for (const auto& s : guard.report()) {
            if (s.name != segment) continue;
            row.ef = s.ef;
            row.est_recall = s.estimate.recall;
            row.est_lower = s.estimate.lower;
            row.est_upper = s.estimate.upper;
          }
        }
        std::printf("%-12s %-9s pass %zu  ef %5zu  recall %.4f  dist/query %7.0f", row.policy.c_str(), row.phase.c_str(),
                    pass, row.ef, row.recall, row.distances);
        if (row.est_recall >= 0) std::printf("  estimate %.3f [%.3f, %.3f]", row.est_recall, row.est_lower, row.est_upper);
        std::printf("\n");
        rows.push_back(row);
      }
    }
  }

  if (!opt.json.empty()) {
    std::ofstream out(opt.json);
    out << "{\n  \"n\": " << base.n << ", \"dim\": " << dim << ", \"k\": " << opt.k << ", \"target\": " << opt.target
        << ", \"batch\": " << B << ", \"passes\": " << opt.passes << ", \"noise\": " << noise
        << ", \"sample_rate\": " << opt.sample_rate << ", \"reference_ef\": " << opt.reference_ef << ", \"filter_share\": " << opt.filter_share
        << ", \"fixed_tuned_ef\": " << tuned << ", \"fixed_worst_ef\": " << worst << ",\n  \"batches\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const Batch& r = rows[i];
      out << "    {\"policy\": \"" << r.policy << "\", \"phase\": \"" << r.phase << "\", \"pass\": " << r.pass
          << ", \"ef\": " << r.ef << ", \"recall\": " << r.recall << ", \"distances\": " << r.distances;
      if (r.est_recall >= 0) {
        out << ", \"estimate\": " << r.est_recall << ", \"lower\": " << r.est_lower << ", \"upper\": " << r.est_upper;
      }
      out << "}" << (i + 1 < rows.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    std::printf("wrote %s\n", opt.json.c_str());
  }
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
