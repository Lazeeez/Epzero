#include "epzero/auto.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "epzero/parallel.h"
#include "pca.h"

namespace epzero {
namespace {

// Walking on principal directions paid off at 960 and 1536 dimensions (GIST1M, DBpedia OpenAI) and
// lost at 96-128 (DEEP, GloVe, SIFT), where an 8-bit record is already one or two cache lines.
// Nothing between was measured, so this threshold is provisional.
constexpr std::size_t kNavMinDim = 256;
// The directions kept: enough for this share of the variance. The winners kept about 80% (GIST at
// 96 directions: 0.83; DBpedia at 256: 0.80), as the re-rank repairs what the walk gets wrong.
constexpr double kNavVariance = 0.80;
// Below a 4x smaller record, the walk saves too little to pay for projecting the query and re-ranking.
constexpr std::size_t kNavMinSaving = 4;
constexpr std::size_t kNavRound = 32;  // whole SIMD blocks
constexpr std::size_t kNavRerank = 300;
constexpr float kNavRerankZ = 3.0f;
constexpr std::size_t kCalibrationQueries = 500;
// Prefetch settings are timed on 300 queries at ef 96 or below, and proportionally fewer at wider beams
// (at least 50), so each timed pass stays about as long: GloVe-100 calibrates to ef 600, where 300
// queries made the timing 9 s of a 60 s build on a Xeon Gold 6226.
constexpr std::size_t kTuneQueries = 300;
constexpr std::size_t kTuneEf = 96;
constexpr std::size_t kMinTuneQueries = 50;
constexpr std::size_t kMaxEf = 4096;
// 4-bit codes are tried only from this many codes per node: below it a record's codes are one or two
// cache lines either way (100 dimensions: 236 bytes with links at 8 bits, 200 at 4, four lines each).
constexpr std::size_t kCodeBitsMinCodes = 128;
// 4-bit codes are kept only if their search at the recall target is this much faster than 8-bit's
// (smaller differences are within the timing's noise, as for prefetch settings).
constexpr double kCodeBitsMinGain = 0.03;
constexpr double kTimingSeconds = 0.25;  // each timed setting runs at least this long, and three passes
// The direction-count trial (AutoOptions::measure_nav): the counts tried besides none and the variance
// rule's, at most half the dimensions; the sample each is built on, and the rows held out of it to calibrate
// and time on when no queries are given. Below kTrialMinRows the trial's builds would be the whole build.
constexpr std::size_t kTrialCounts[] = {128, 192, 256, 384, 512};
constexpr std::size_t kTrialRows = 100'000;
constexpr std::size_t kTrialHeld = 1000;
constexpr std::size_t kTrialMinRows = 20'000;
// check_walk decides between a walk and none, a far coarser choice (3-28x apart on COCO), so a smaller sample.
constexpr std::size_t kWalkTrialRows = 50'000;
// check_walk's trial runs only for queries unlike the data, judged as check_queries judges them, on this many
// rows sampled before the build (a brute-force pass of well under a second).
constexpr std::size_t kShiftSampleRows = 20'000;
// Advice thresholds: tombstones still cost search time; growth makes a calibrated beam stale; a fall in
// the variance the directions hold means the data moved away from what the walk was built for.
constexpr double kCompactShare = 0.2;
constexpr double kRecalibrateGrowth = 0.25;
constexpr double kNavVarianceDrop = 0.05;
constexpr std::size_t kHealthSample = 5000;
constexpr double kCodeErrorLimit = 0.01;
constexpr std::size_t kQueryCheckSample = 200;
constexpr double kShiftRatio = 1.5;  // between 1.09 (queries like the data) and 2.26 (text against images)

std::string percent(double share) { return std::to_string(static_cast<int>(share * 100.0 + 0.5)) + "%"; }

void note(IndexProfile& profile, std::string setting, std::string why) {
  profile.notes.emplace_back(std::move(setting), std::move(why));
}

// Rows `rows` of `data`, copied into one block.
std::vector<float> gather(const float* data, std::size_t dim, const std::vector<std::size_t>& rows) {
  std::vector<float> out(rows.size() * dim);
  for (std::size_t i = 0; i < rows.size(); ++i) std::copy_n(data + rows[i] * dim, dim, out.data() + i * dim);
  return out;
}

// `count` distinct rows of [0, n), in increasing order.
std::vector<std::size_t> sample_rows(std::size_t n, std::size_t count, std::uint64_t seed) {
  count = std::min(count, n);
  std::vector<std::size_t> rows;
  if (count * 2 > n) {  // most of them: take every row and drop at random
    rows.resize(n);
    for (std::size_t i = 0; i < n; ++i) rows[i] = i;
    std::shuffle(rows.begin(), rows.end(), std::mt19937_64(seed));
    rows.resize(count);
  } else {
    std::unordered_set<std::size_t> picked;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, n - 1);
    while (picked.size() < count) picked.insert(pick(rng));
    rows.assign(picked.begin(), picked.end());
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

// The smallest beam width reaching the recall target on `queries`, which must not be in the graph:
// a row already in it is linked to its own nearest neighbours, which makes its search far easier
// than a real query's (GloVe: 0.95 that way, 0.87 on rows held out). Returns (ef, recall reached).
using Truth = std::vector<std::vector<std::uint64_t>>;

// Each query's exact top k (sorted labels): a scan of every vector, so a build that calibrates twice on the
// same queries (8-bit and 4-bit codes) computes it once.
Truth exact_truth(const HnswIndex& index, const float* queries, std::size_t count, std::size_t k, std::size_t threads) {
  const std::size_t dim = index.params().dim;
  Truth truth(count);
  parallel_for(0, count, threads, [&](std::size_t q) {
    for (const auto& r : index.exact_search(queries + q * dim, k)) truth[q].push_back(r.label);
    std::sort(truth[q].begin(), truth[q].end());
  });
  return truth;
}

std::pair<std::size_t, float> calibrate(const HnswIndex& index, const float* queries, std::size_t count,
                                        const IndexProfile& profile, float target, std::size_t threads,
                                        const Truth* known = nullptr) {
  const std::size_t dim = index.params().dim;
  const std::size_t k = profile.k;
  const Truth truth = known != nullptr ? *known : exact_truth(index, queries, count, k, threads);
  const auto recall = [&](std::size_t ef) {
    std::atomic<std::size_t> hits{0};
    std::size_t expected = 0;
    for (const auto& t : truth) expected += t.size();
    parallel_for(0, count, threads, [&](std::size_t q) {
      const auto found = index.search(queries + q * dim, k,
                                      {.ef = std::max(ef, k), .rerank = profile.rerank, .rerank_z = profile.rerank_z});
      std::size_t h = 0;
      for (const auto& r : found) h += std::binary_search(truth[q].begin(), truth[q].end(), r.label) ? 1 : 0;
      hits += h;
    });
    return expected == 0 ? 1.0f : static_cast<float>(hits.load()) / static_cast<float>(expected);
  };

  // A beam below k measures as k (the search widens it), so the answer is at least k: a lower one would
  // run unmeasured at a smaller k (a beam of 2 for k = 1 missed rows added since).
  std::size_t low = k - 1;
  std::size_t ef = std::max<std::size_t>(16, k);
  float reached = recall(ef);
  while (reached < target && ef < kMaxEf) {
    low = ef;
    ef *= 2;
    reached = recall(ef);
  }
  if (reached < target) return {ef, reached};  // out of reach at these graph settings: the widest beam
  while (ef - low > std::max<std::size_t>(2, ef / 16)) {  // narrow to within about 6%
    const std::size_t middle = (low + ef) / 2;
    const float r = recall(middle);
    if (r >= target) {
      ef = middle;
      reached = r;
    } else {
      low = middle;
    }
  }
  return {ef, reached};
}

// Seconds per query of searching `queries` on one thread with `options` (the fastest of at least three
// passes and kTimingSeconds), for comparing two settings of the same index.
double time_search(const HnswIndex& index, const float* queries, std::size_t count, std::size_t k,
                   const SearchOptions& options) {
  const std::size_t dim = index.params().dim;
  using clock = std::chrono::steady_clock;
  double best = std::numeric_limits<double>::max();
  double total = 0.0;
  for (int pass = 0; (pass < 3 || total < kTimingSeconds) && pass < 50; ++pass) {
    const auto start = clock::now();
    for (std::size_t q = 0; q < count; ++q) (void)index.search(queries + q * dim, k, options);
    const double seconds = std::chrono::duration<double>(clock::now() - start).count();
    best = std::min(best, seconds);
    total += seconds;
  }
  return best / static_cast<double>(std::max<std::size_t>(count, 1));
}

// check_queries's ratio before any index exists: the median distance from a query to its nearest row of a sample,
// over the median distance from a sampled row to its nearest other row of it, by brute force.
double query_shift(const float* data, std::size_t n, std::size_t dim, Metric metric, const float* queries, std::size_t count,
                   std::uint64_t seed, std::size_t threads) {
  const std::vector<std::size_t> picked = sample_rows(n, std::min(n, kShiftSampleRows), seed + 5);
  std::vector<float> sample = gather(data, dim, picked);
  const std::size_t q = std::min(count, kQueryCheckSample);
  std::vector<float> probes(queries, queries + q * dim);
  if (metric == Metric::Cosine) {
    for (std::size_t i = 0; i < picked.size(); ++i) normalize(sample.data() + i * dim, dim);
    for (std::size_t i = 0; i < q; ++i) normalize(probes.data() + i * dim, dim);
  }
  const DistanceFn distance = get_distance_fn(metric, active_simd_level());
  const auto nearest = [&](const float* v, std::size_t skip) {
    double best = std::numeric_limits<double>::max();
    for (std::size_t r = 0; r < picked.size(); ++r) {
      if (r != skip) best = std::min(best, static_cast<double>(distance(v, sample.data() + r * dim, dim)));
    }
    return best;
  };
  const std::size_t rows = std::min(picked.size(), kQueryCheckSample);
  std::vector<double> query_distance(q), row_distance(rows);
  parallel_for(0, q, threads, [&](std::size_t i) { query_distance[i] = nearest(probes.data() + i * dim, picked.size()); });
  const std::size_t stride = std::max<std::size_t>(1, picked.size() / rows);
  parallel_for(0, rows, threads, [&](std::size_t i) { row_distance[i] = nearest(sample.data() + i * stride * dim, i * stride); });
  const auto median = [](std::vector<double>& v) {
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  const double rd = median(row_distance), qd = median(query_distance);
  return rd > 0.0 ? qd / rd : (qd > 0.0 ? 1e9 : 1.0);
}

struct NavTrial {
  std::size_t nav = 0;
  std::string why;
};

// The direction count whose trial index on a sample searches fastest at the recall target (see
// AutoOptions::measure_nav). Each trial is build_auto on the sample with that count fixed, 8-bit codes and
// the default prefetch, calibrated on the timing queries' first half (or all of them, if they are the
// caller's), then timed on all of them on one thread.
NavTrial nav_by_trial(const float* data, std::size_t n, std::size_t dim, const AutoOptions& options, std::size_t rule,
                      std::vector<std::size_t> counts, std::size_t sample_rows_wanted) {
  std::sort(counts.begin(), counts.end());
  counts.erase(std::unique(counts.begin(), counts.end()), counts.end());

  const bool own = options.queries != nullptr && options.query_count > 0;
  const std::size_t held_count = own ? 0 : std::min(kTrialHeld, n / 5);
  const std::size_t rows = std::min(sample_rows_wanted, n - held_count);
  std::vector<std::size_t> picked = sample_rows(n, rows + held_count, options.seed + 3);
  std::shuffle(picked.begin(), picked.end(), std::mt19937_64(options.seed + 4));
  std::vector<std::size_t> held(picked.begin(), picked.begin() + static_cast<std::ptrdiff_t>(held_count));
  std::vector<std::size_t> kept(picked.begin() + static_cast<std::ptrdiff_t>(held_count), picked.end());
  std::sort(kept.begin(), kept.end());
  const std::vector<float> sample = gather(data, dim, kept);
  const std::vector<float> held_rows = gather(data, dim, held);
  const float* timing = own ? options.queries : held_rows.data();
  const std::size_t timing_count = own ? std::min<std::size_t>(options.query_count, kTrialHeld) : held.size();
  const std::size_t calibration_count = own ? timing_count : std::max<std::size_t>(1, timing_count / 2);

  NavTrial best{rule, ""};
  double best_seconds = std::numeric_limits<double>::max();
  std::string tried;
  for (const std::size_t c : counts) {
    AutoOptions o = options;
    o.measure_nav = false;
    o.check_walk = false;
    o.capacity = 0;
    o.fixed.nav_dim = c;
    o.fixed.code_bits = 8;
    o.fixed.prefetch = std::make_pair(std::numeric_limits<std::size_t>::max(), std::size_t{1});
    o.queries = timing;
    o.query_count = calibration_count;
    const auto index = build_auto(sample.data(), nullptr, kept.size(), dim, o);
    const IndexProfile& p = index->profile();
    if (!tried.empty()) tried += ", ";
    if (p.recall_measured < options.recall) {
      tried += std::to_string(c) + ": recall " + std::to_string(p.recall_measured).substr(0, 5);
      continue;
    }
    const double seconds = time_search(*index, timing, timing_count, p.k, {.ef = p.ef, .rerank = p.rerank, .rerank_z = p.rerank_z});
    tried += std::to_string(c) + ": " + std::to_string(static_cast<int>(seconds * 1e6 + 0.5)) + " us";
    if (seconds < best_seconds) {
      best_seconds = seconds;
      best.nav = c;
    }
  }
  best.why = (best_seconds == std::numeric_limits<double>::max() ? std::string("no count reached the target on the trial; the rule's ")
                                                                  : std::string("the fastest at the recall target on ")) +
             std::to_string(kept.size()) + " rows, timed on " + (own ? "your queries" : "rows held out") + " (" + tried +
             "; 0 is no walk)";
  return best;
}

}  // namespace

AutoPlan plan_auto(const float* data, std::size_t n, std::size_t dim, const AutoOptions& options) {
  if (data == nullptr || n == 0 || dim == 0) throw std::invalid_argument("build_auto needs at least one vector");
  if (!(options.recall > 0.0f && options.recall <= 1.0f)) throw std::invalid_argument("recall must be in (0, 1]");
  if (options.k == 0) throw std::invalid_argument("k must be at least 1");
  const AutoFixed& fixed = options.fixed;
  AutoPlan plan;
  HnswParams& params = plan.params;
  IndexProfile& profile = plan.profile;
  params.dim = dim;
  params.metric = options.metric;
  params.max_elements = std::max(options.capacity, n);
  params.M = options.M;
  params.ef_construction = options.ef_construction;
  params.seed = options.seed;
  params.keep_vectors = true;
  const std::string yours = "set by you";

  params.quantization = fixed.quantization.value_or(Quantization::SQ8);
  note(profile, "quantization",
       fixed.quantization ? yours : "8-bit codes: a quarter of the memory traffic, re-ranked with the full vectors");

  std::size_t nav = 0;
  if (fixed.nav_dim) {
    nav = *fixed.nav_dim;
    note(profile, "nav_dim", yours);
  } else if (params.quantization != Quantization::SQ8) {
    note(profile, "nav_dim", "off: principal directions need 8-bit codes");
  } else if (dim < kNavMinDim) {
    note(profile, "nav_dim", "off: below " + std::to_string(kNavMinDim) + " dimensions an 8-bit record is already short");
  } else {
    const auto row = [&](std::size_t i, float* out) {
      std::copy_n(data + i * dim, dim, out);
      if (options.metric == Metric::Cosine) normalize(out, dim);
    };
    const std::size_t most = dim / kNavMinSaving;  // the most directions worth walking on
    const auto held = pca::variance_held(row, n, dim, most, options.seed, options.threads);
    const auto enough = std::find_if(held.begin(), held.end(), [](double h) { return h >= kNavVariance; });
    if (enough == held.end()) {
      note(profile, "nav_dim", "off: " + std::to_string(most) + " principal directions hold only " +
                                   percent(held.empty() ? 0.0 : held.back()) + " of the variance");
    } else {
      const std::size_t needed = static_cast<std::size_t>(enough - held.begin()) + 1;
      nav = std::min(dim, (needed + kNavRound - 1) / kNavRound * kNavRound);
      if (nav * kNavMinSaving > dim) nav = std::max<std::size_t>(kNavRound, dim / kNavMinSaving / kNavRound * kNavRound);
      note(profile, "nav_dim", std::to_string(needed) + " principal directions hold " + percent(kNavVariance) +
                                   " of the variance");
    }
  }
  params.nav_dim = nav;
  if (nav != 0 && (params.quantization != Quantization::SQ8 || nav > dim)) {
    throw std::invalid_argument("nav_dim needs 8-bit quantization and at most dim directions");
  }

  // 16-bit full vectors halve the largest part of the index, but were 5-17% slower on GIST1M and lost
  // 0.003-0.004 recall on OpenAI embeddings (laptop), so they are only ever the caller's choice.
  params.half_vectors = fixed.half_vectors.value_or(false);
  note(profile, "half_vectors", fixed.half_vectors ? yours
                                : "off: 16-bit vectors halve their memory but cost speed and some recall; set it to save memory");
  if (params.half_vectors && params.quantization != Quantization::SQ8) {
    throw std::invalid_argument("half_vectors needs 8-bit quantization");
  }
  // 4-bit codes: the graph is built on 8-bit ones either way, then walked on 4-bit codes if they are
  // faster at the recall target (measured when built), so a dataset where 16 levels lose too much (Glass's
  // 4-bit walk never reached recall 0.90 on LAION, in VIBE) keeps 8 bits.
  if (fixed.code_bits) {
    if (*fixed.code_bits != 8 && *fixed.code_bits != 4) throw std::invalid_argument("code_bits must be 8 or 4");
    if (*fixed.code_bits == 4 && params.quantization != Quantization::SQ8) {
      throw std::invalid_argument("4-bit codes need 8-bit quantization (their ranges)");
    }
    plan.code_bits = *fixed.code_bits;
    note(profile, "code_bits", yours);
  } else if (params.quantization != Quantization::SQ8) {
    note(profile, "code_bits", "off: only for 8-bit quantization");
  } else if ((nav != 0 ? nav : dim) < kCodeBitsMinCodes) {
    note(profile, "code_bits", "8: below " + std::to_string(kCodeBitsMinCodes) +
                                   " codes a node's record takes as many cache lines at 4 bits");
  } else if (fixed.ef) {
    note(profile, "code_bits", "8: choosing 4 bits needs the beam width calibrated");
  } else {
    plan.measure_code_bits = true;
    note(profile, "code_bits", "8 or 4, whichever searches faster at the recall target (timed when built)");
  }
  params.fast_build = fixed.fast_build.value_or(params.quantization == Quantization::SQ8);
  note(profile, "fast_build", fixed.fast_build ? yours
                              : params.fast_build ? "built on the codes: faster, and as good a graph in our measurements"
                                                  : "off: only for 8-bit codes");
  profile.built_on_codes = params.fast_build;
  params.extra_backlinks = fixed.extra_backlinks.value_or(0);
  note(profile, "extra_backlinks", fixed.extra_backlinks ? yours : "off: the standard insertion");
  profile.refined = fixed.refine.value_or(true);
  note(profile, "refine", fixed.refine ? yours : "a second pass gave higher recall at equal speed on every dataset measured");
  profile.filled = fixed.fill_after.value_or(nav != 0);
  note(profile, "fill_after", fixed.fill_after ? yours
                              : nav != 0 ? "free link slots filled after the build (measured with principal directions)"
                                         : "off: not measured to help without principal directions");
  profile.rerank = fixed.rerank.value_or(nav != 0 ? kNavRerank : 0);
  note(profile, "rerank", fixed.rerank ? yours
                          : nav != 0 ? "up to " + std::to_string(kNavRerank) + " or ef, whichever is larger: the walk on directions is approximate"
                                     : "twice k: enough when the walk uses every dimension");
  profile.rerank_z = fixed.rerank_z.value_or(nav != 0 ? kNavRerankZ : 0.0f);
  note(profile, "rerank_z", fixed.rerank_z ? yours
                            : nav != 0 ? "stops the re-rank once the rest are 3 standard deviations past the k-th best"
                                       : "off: the re-rank is short");
  profile.k = options.k;
  plan.calibrate = !fixed.ef.has_value();
  profile.ef = fixed.ef.value_or(0);
  profile.recall_target = plan.calibrate ? options.recall : 0.0f;
  note(profile, "ef", fixed.ef ? yours
                      : "calibrated for recall " + std::to_string(options.recall).substr(0, 4) + " at k = " +
                            std::to_string(options.k) +
                            (options.queries != nullptr ? " on your queries" : " on rows held out of the graph"));
  plan.tune = !fixed.prefetch.has_value();
  if (fixed.prefetch) {
    profile.prefetch_depth = fixed.prefetch->first;
    profile.prefetch_lines = std::max<std::size_t>(1, fixed.prefetch->second);
  }
  note(profile, "prefetch", fixed.prefetch ? yours : "timed on this machine");
  return plan;
}

std::unique_ptr<HnswIndex> build_auto(const float* data, const std::uint64_t* labels, std::size_t n, std::size_t dim,
                                      const AutoOptions& options) {
  const bool own_queries_given = options.queries != nullptr && options.query_count > 0;
  const bool trial_allowed = !options.fixed.nav_dim && dim >= kNavMinDim && n >= kTrialMinRows &&
                             options.fixed.quantization.value_or(Quantization::SQ8) == Quantization::SQ8;
  const std::size_t rule = trial_allowed && (options.measure_nav || (options.check_walk && own_queries_given))
                               ? plan_auto(data, n, dim, options).params.nav_dim : 0;
  // check_walk pays for its trial only when the queries look unlike the data (text against images): queries
  // like it walk well on its directions (AG News: 37 us walking, 54 us without), and the trial cost them 41%
  // of the build for nothing.
  const double shift = !options.measure_nav && options.check_walk && own_queries_given && rule != 0 && trial_allowed
                           ? query_shift(data, n, dim, options.metric, options.queries, options.query_count, options.seed,
                                         options.threads)
                           : 0.0;
  if (trial_allowed && (options.measure_nav || (options.check_walk && own_queries_given && rule != 0 && shift > kShiftRatio))) {
    std::vector<std::size_t> counts = {0, rule};
    if (options.measure_nav) {
      for (const std::size_t c : kTrialCounts) {
        if (c <= dim / 2) counts.push_back(c);
      }
    }
    const NavTrial trial = nav_by_trial(data, n, dim, options, rule, counts, options.measure_nav ? kTrialRows : kWalkTrialRows);
    AutoOptions o = options;
    o.measure_nav = false;
    o.check_walk = false;
    o.fixed.nav_dim = trial.nav;
    auto index = build_auto(data, labels, n, dim, o);
    IndexProfile profile = index->profile();
    for (auto& [setting, text] : profile.notes) {
      if (setting == "nav_dim") text = trial.why;
    }
    index->set_profile(std::move(profile));
    return index;
  }
  AutoPlan plan = plan_auto(data, n, dim, options);
  IndexProfile& profile = plan.profile;
  const std::size_t threads = options.threads;
  const auto label = [&](std::size_t i) { return labels != nullptr ? labels[i] : static_cast<std::uint64_t>(i); };

  // Rows to calibrate on go in last, so the graph has never seen them when they are searched.
  std::vector<std::size_t> held;
  const bool own_queries = options.queries != nullptr && options.query_count > 0;
  if (plan.calibrate && !own_queries && n >= 4 * options.k) {
    held = sample_rows(n, std::min(kCalibrationQueries, n / 4), options.seed + 1);
  }
  std::vector<char> is_held(n, 0);
  for (const std::size_t r : held) is_held[r] = 1;

  auto index = std::make_unique<HnswIndex>(plan.params);
  index->train(data, n);
  parallel_for(0, n, threads, [&](std::size_t i) {
    if (!is_held[i]) index->add(label(i), data + i * dim);
  });
  if (profile.refined) {
    // Without principal directions the second pass is most of the build, and half the beam gives the
    // same graph: GloVe-100 at 200k, recall within 0.001 at every ef, 7.7 s against 10.5 s. With them
    // it saved 5% and cost 0.002-0.004 recall (GIST1M), so they keep the full beam.
    const std::size_t ef = options.fixed.refine_ef.value_or(
        plan.params.nav_dim == 0 ? std::max<std::size_t>(plan.params.ef_construction / 2, 2 * plan.params.M)
                                 : plan.params.ef_construction);
    index->refine_links(threads, ef);
  }
  if (profile.filled) index->fill_free_links(threads);
  const auto four_bit_rerank = [&](IndexProfile& p) {
    if (options.fixed.rerank_z || p.rerank_z != 0.0f) return;
    p.rerank_z = kNavRerankZ;  // the 4-bit walk ranks less exactly: re-rank up to max(rerank, ef), stopping early
    for (auto& [setting, text] : p.notes) {
      if (setting == "rerank_z") text = "stops the re-rank once the rest are 3 standard deviations past the k-th best (4-bit walk)";
    }
  };
  if (plan.code_bits == 4) {
    index->set_code_bits(4, threads);
    four_bit_rerank(profile);
  }

  if (plan.calibrate) {
    const std::vector<float> rows = gather(data, dim, held);
    const float* queries = own_queries ? options.queries : rows.data();
    const std::size_t count = own_queries ? options.query_count : held.size();
    Truth truth;
    if (count > 0) {
      truth = exact_truth(*index, queries, count, profile.k, threads);
      const auto [ef, reached] = calibrate(*index, queries, count, profile, options.recall, threads, &truth);
      profile.ef = ef;
      profile.recall_measured = reached;
    } else {
      profile.ef = std::max<std::size_t>(64, options.k);  // too few rows to hold any out
    }
    if (count > 0 && plan.measure_code_bits) {
      // The same graph walked on 4-bit codes, calibrated to the same target and timed against the
      // 8-bit walk on one thread.
      const double seconds8 = time_search(*index, queries, count, profile.k,
                                          {.ef = profile.ef, .rerank = profile.rerank, .rerank_z = profile.rerank_z});
      IndexProfile four = profile;
      four_bit_rerank(four);
      index->set_code_bits(4, threads);
      const auto [ef4, reached4] = calibrate(*index, queries, count, four, options.recall, threads, &truth);
      const bool reaches = reached4 >= options.recall;
      const double seconds4 =
          reaches ? time_search(*index, queries, count, four.k, {.ef = ef4, .rerank = four.rerank, .rerank_z = four.rerank_z}) : 0.0;
      const bool keep = reaches && seconds4 < seconds8 * (1.0 - kCodeBitsMinGain);
      const auto micros = [](double s) { return std::to_string(static_cast<int>(s * 1e6 + 0.5)) + " us"; };
      const std::string why = !reaches ? "8: 4-bit codes reached only " + std::to_string(reached4).substr(0, 5) + " recall"
                              : keep   ? "4: " + micros(seconds4) + " a query at the recall target, against " + micros(seconds8) + " with 8 bits"
                                       : "8: " + micros(seconds8) + " a query at the recall target, against " + micros(seconds4) + " with 4 bits";
      if (keep) {
        four.ef = ef4;
        four.recall_measured = reached4;
        profile = std::move(four);
      } else {
        index->set_code_bits(8, threads);
      }
      for (auto& [setting, text] : profile.notes) {
        if (setting == "code_bits") text = why;
      }
    }
  }
  parallel_for(0, held.size(), threads, [&](std::size_t i) { index->add(label(held[i]), data + held[i] * dim); });

  if (plan.tune) {
    const std::size_t queries = std::max(kMinTuneQueries, kTuneQueries * kTuneEf / std::max(kTuneEf, profile.ef));
    const std::vector<float> rows = gather(data, dim, sample_rows(n, queries, options.seed + 2));
    const auto [depth, lines] = index->tune_prefetch(rows.data(), rows.size() / dim, profile.k, profile.ef);
    profile.prefetch_depth = depth;
    profile.prefetch_lines = lines;
  } else {
    index->set_prefetch(profile.prefetch_depth, profile.prefetch_lines);
  }
  profile.machine = machine_name();
  profile.calibrated_size = plan.calibrate ? n : 0;
  index->set_profile(std::move(profile));
  if (plan.params.nav_dim != 0) {
    IndexProfile p = index->profile();
    p.nav_variance = static_cast<float>(nav_variance(*index, threads));
    index->set_profile(std::move(p));
  }
  return index;
}

std::pair<std::size_t, float> recalibrate(HnswIndex& index, const float* queries, std::size_t count, float recall,
                                          std::size_t threads) {
  if (queries == nullptr || count == 0) throw std::invalid_argument("recalibrate needs queries");
  IndexProfile profile = index.profile();
  const float target = recall > 0.0f ? recall : profile.recall_target > 0.0f ? profile.recall_target : 0.95f;
  if (!(target > 0.0f && target <= 1.0f)) throw std::invalid_argument("recall must be in (0, 1]");
  const auto [ef, reached] = calibrate(index, queries, count, profile, target, threads);
  profile.ef = ef;
  profile.recall_target = target;
  profile.recall_measured = reached;
  profile.calibrated_size = index.live_count();
  std::erase_if(profile.notes, [](const auto& note) { return note.first == "ef"; });
  note(profile, "ef", "recalibrated for recall " + std::to_string(target).substr(0, 4) + " on your queries");
  index.set_profile(std::move(profile));
  return {ef, reached};
}

double nav_variance(const HnswIndex& index, std::size_t threads) {
  const std::vector<float>& basis = index.nav_basis();
  const std::size_t dim = index.params().dim;
  const std::size_t directions = index.params().nav_dim;
  if (basis.empty() || directions == 0) return 0.0;
  auto labels = index.live_labels();
  if (labels.size() < 2) return 0.0;
  std::shuffle(labels.begin(), labels.end(), std::mt19937_64(7));
  labels.resize(std::min(labels.size(), kHealthSample));
  std::vector<float> rows(labels.size() * dim);
  parallel_for(0, labels.size(), threads, [&](std::size_t i) { index.get_vector(labels[i], rows.data() + i * dim); });
  std::vector<double> mean(dim, 0.0);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    for (std::size_t d = 0; d < dim; ++d) mean[d] += rows[i * dim + d];
  }
  for (double& m : mean) m /= static_cast<double>(labels.size());
  // Per row: its squared distance from the mean, and the part of it along the directions (orthonormal).
  std::vector<double> total(labels.size()), held(labels.size());
  parallel_for(0, labels.size(), threads, [&](std::size_t i) {
    const float* v = rows.data() + i * dim;
    double t = 0.0, h = 0.0;
    for (std::size_t d = 0; d < dim; ++d) t += (v[d] - mean[d]) * (v[d] - mean[d]);
    for (std::size_t r = 0; r < directions; ++r) {
      const float* q = basis.data() + r * dim;
      double dot = 0.0;
      for (std::size_t d = 0; d < dim; ++d) dot += q[d] * (v[d] - mean[d]);
      h += dot * dot;
    }
    total[i] = t;
    held[i] = h;
  });
  double t = 0.0, h = 0.0;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    t += total[i];
    h += held[i];
  }
  return t > 0.0 ? std::min(1.0, h / t) : 0.0;
}

QueryCheck check_queries(const HnswIndex& index, const float* queries, std::size_t count, std::size_t threads) {
  if (queries == nullptr || count == 0) throw std::invalid_argument("check_queries needs queries");
  auto labels = index.live_labels();
  if (labels.size() < 3) throw std::invalid_argument("check_queries needs an index of at least 3 vectors");
  const std::size_t dim = index.params().dim;
  const std::size_t q = std::min(count, kQueryCheckSample);
  std::shuffle(labels.begin(), labels.end(), std::mt19937_64(11));
  labels.resize(std::min(labels.size(), kQueryCheckSample));
  std::vector<float> rows(labels.size() * dim);
  for (std::size_t i = 0; i < labels.size(); ++i) index.get_vector(labels[i], rows.data() + i * dim);
  std::vector<double> query_distance(q), row_distance(labels.size());
  parallel_for(0, q, threads, [&](std::size_t i) {
    query_distance[i] = index.exact_search(queries + i * dim, 1).front().distance;
  });
  // A row's nearest neighbour other than itself: the second result (the first is the row, at distance 0).
  parallel_for(0, labels.size(), threads, [&](std::size_t i) {
    const auto found = index.exact_search(rows.data() + i * dim, 2);
    row_distance[i] = found.size() > 1 ? found[1].distance : found.front().distance;
  });
  const auto median = [](std::vector<double>& v) {
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  QueryCheck out;
  out.query_distance = median(query_distance);
  out.row_distance = median(row_distance);
  out.ratio = out.row_distance > 0.0 ? out.query_distance / out.row_distance : (out.query_distance > 0.0 ? 1e9 : 1.0);
  out.shifted = out.ratio > kShiftRatio;
  const IndexProfile& profile = index.profile();
  const bool calibrated_on_queries = std::any_of(profile.notes.begin(), profile.notes.end(), [](const auto& note) {
    return note.first == "ef" && note.second.find("your queries") != std::string::npos;
  });
  if (out.shifted && !calibrated_on_queries) {
    char ratio[16];
    std::snprintf(ratio, sizeof(ratio), "%.1f", out.ratio);
    out.advice.push_back(std::string("recalibrate(queries): these queries sit ") + ratio +
                         "x farther from their nearest neighbours than rows of the data do, so recall calibrated on "
                         "the data overestimates theirs");
  }
  return out;
}

IndexHealth check_health(const HnswIndex& index, std::size_t threads) {
  IndexHealth out;
  const IndexProfile& profile = index.profile();
  out.live = index.live_count();
  out.deleted = index.deleted_count();
  const auto percent_of = [](double share) { return std::to_string(static_cast<long long>(std::lround(share * 100.0))) + "%"; };
  const double slots = static_cast<double>(out.live + out.deleted);
  if (slots > 0 && out.deleted / slots > kCompactShare) {
    out.advice.push_back("compact(): " + percent_of(out.deleted / slots) +
                         " of the index is deleted vectors, which still cost search time");
  }
  if (profile.calibrated_size > 0) {
    out.growth = static_cast<double>(out.live) / static_cast<double>(profile.calibrated_size) - 1.0;
    if (std::abs(out.growth) > kRecalibrateGrowth) {
      out.advice.push_back(std::string("recalibrate(queries): the index has ") + (out.growth > 0 ? "grown " : "shrunk ") +
                           percent_of(std::abs(out.growth)) + " since its beam width was calibrated");
    }
  } else if (profile.recall_target == 0.0f) {
    out.advice.push_back("recalibrate(queries): the beam width was never calibrated for a recall target");
  }
  out.code_error = index.code_error(kHealthSample, threads);
  if (out.code_error > kCodeErrorLimit) {
    out.advice.push_back("requantize(): vectors added since the build lie outside the ranges the 8-bit codes were "
                         "fitted on, and their codes miss " + percent_of(out.code_error) + " of the variance");
  }
  if (index.params().nav_dim != 0) {
    out.nav_variance_built = profile.nav_variance;
    out.nav_variance_now = nav_variance(index, threads);
    if (profile.nav_variance > 0.0f && out.nav_variance_now < out.nav_variance_built - kNavVarianceDrop) {
      out.advice.push_back("rebuild with build_auto: the principal directions now hold " + percent_of(out.nav_variance_now) +
                           " of the variance, down from " + percent_of(out.nav_variance_built) + " when built");
    }
  }
  return out;
}

}  // namespace epzero
