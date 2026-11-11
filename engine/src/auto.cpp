#include "epzero/auto.h"

#include <algorithm>
#include <atomic>
#include <cmath>
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
// Advice thresholds: tombstones still cost search time; growth makes a calibrated beam stale; a fall in
// the variance the directions hold means the data moved away from what the walk was built for.
constexpr double kCompactShare = 0.2;
constexpr double kRecalibrateGrowth = 0.25;
constexpr double kNavVarianceDrop = 0.05;
constexpr std::size_t kHealthSample = 5000;

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
std::pair<std::size_t, float> calibrate(const HnswIndex& index, const float* queries, std::size_t count,
                                        const IndexProfile& profile, float target, std::size_t threads) {
  const std::size_t dim = index.params().dim;
  const std::size_t k = profile.k;
  std::vector<std::vector<std::uint64_t>> truth(count);
  parallel_for(0, count, threads, [&](std::size_t q) {
    for (const auto& r : index.exact_search(queries + q * dim, k)) truth[q].push_back(r.label);
    std::sort(truth[q].begin(), truth[q].end());
  });
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

  std::size_t low = 0;
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

  params.fast_build = fixed.fast_build.value_or(params.quantization == Quantization::SQ8);
  note(profile, "fast_build", fixed.fast_build ? yours
                              : params.fast_build ? "built on the codes: faster, and as good a graph in our measurements"
                                                  : "off: only for 8-bit codes");
  profile.built_on_codes = params.fast_build;
  profile.refined = fixed.refine.value_or(true);
  note(profile, "refine", fixed.refine ? yours : "a second pass gave higher recall at equal speed on every dataset measured");
  profile.filled = fixed.fill_after.value_or(nav != 0);
  note(profile, "fill_after", fixed.fill_after ? yours
                              : nav != 0 ? "free link slots filled after the build (measured with principal directions)"
                                         : "off: not measured to help without principal directions");
  profile.rerank = fixed.rerank.value_or(nav != 0 ? kNavRerank : 0);
  note(profile, "rerank", fixed.rerank ? yours
                          : nav != 0 ? "up to " + std::to_string(kNavRerank) + ": the walk on directions is approximate"
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
    const std::size_t ef = plan.params.nav_dim == 0 ? std::max<std::size_t>(plan.params.ef_construction / 2, 2 * plan.params.M)
                                                    : plan.params.ef_construction;
    index->refine_links(threads, ef);
  }
  if (profile.filled) index->fill_free_links(threads);

  if (plan.calibrate) {
    const std::vector<float> rows = gather(data, dim, held);
    const float* queries = own_queries ? options.queries : rows.data();
    const std::size_t count = own_queries ? options.query_count : held.size();
    if (count > 0) {
      const auto [ef, reached] = calibrate(*index, queries, count, profile, options.recall, threads);
      profile.ef = ef;
      profile.recall_measured = reached;
    } else {
      profile.ef = std::max<std::size_t>(64, options.k);  // too few rows to hold any out
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
