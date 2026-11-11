#include <gtest/gtest.h>

#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "epzero/auto.h"
#include "epzero/brute_force.h"
#include "epzero/dataset.h"

namespace epzero {
namespace {

// Rows near a `latent`-dimensional subspace: most of the variance in few directions.
std::vector<float> low_rank(std::size_t n, std::size_t dim, std::size_t latent, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> normal;
  std::vector<float> basis(latent * dim), centers(30 * latent), out(n * dim);
  for (float& v : basis) v = normal(rng);
  for (float& v : centers) v = normal(rng);
  std::vector<float> z(latent);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t c = rng() % 30;
    for (std::size_t l = 0; l < latent; ++l) z[l] = centers[c * latent + l] + 0.4f * normal(rng);
    for (std::size_t d = 0; d < dim; ++d) {
      float sum = 0.05f * normal(rng);
      for (std::size_t l = 0; l < latent; ++l) sum += z[l] * basis[l * dim + d];
      out[i * dim + d] = sum;
    }
  }
  return out;
}

std::string why(const IndexProfile& profile, const std::string& setting) {
  for (const auto& [name, reason] : profile.notes) {
    if (name == setting) return reason;
  }
  return "";
}

double recall_on(const HnswIndex& index, const std::vector<float>& base, const std::vector<float>& queries,
                 std::size_t dim, Metric metric) {
  const std::size_t n = base.size() / dim, q = queries.size() / dim;
  const auto& p = index.profile();
  std::vector<std::vector<SearchResult>> results;
  for (std::size_t i = 0; i < q; ++i) {
    results.push_back(index.search(queries.data() + i * dim, p.k, {.ef = p.ef, .rerank = p.rerank, .rerank_z = p.rerank_z}));
  }
  // compute_ground_truth expects cosine data already normalized, as the index stores it.
  std::vector<float> stored = base;
  if (metric == Metric::Cosine) {
    for (std::size_t i = 0; i < n; ++i) normalize(stored.data() + i * dim, dim);
  }
  return recall_at_k(results, compute_ground_truth(stored.data(), n, dim, queries.data(), q, p.k, metric), p.k);
}

TEST(Auto, ShortVectorsWalkOnEveryDimension) {
  const auto data = make_clustered(3000, 0, 64, 20, 1);
  const AutoPlan plan = plan_auto(data.base.data.data(), 3000, 64);
  EXPECT_EQ(plan.params.nav_dim, 0u);
  EXPECT_EQ(plan.params.quantization, Quantization::SQ8);
  EXPECT_TRUE(plan.params.fast_build);
  EXPECT_TRUE(plan.profile.refined);
  EXPECT_FALSE(plan.profile.filled);
  EXPECT_NE(why(plan.profile, "nav_dim").find("below"), std::string::npos);
}

TEST(Auto, ConcentratedSpectrumWalksOnDirections) {
  const auto base = low_rank(3000, 512, 16, 2);
  const AutoPlan plan = plan_auto(base.data(), 3000, 512);
  EXPECT_GT(plan.params.nav_dim, 0u);
  EXPECT_LE(plan.params.nav_dim, 512u / 4);
  EXPECT_EQ(plan.params.nav_dim % 32, 0u);
  EXPECT_GT(plan.profile.rerank, 0u);
  EXPECT_GT(plan.profile.rerank_z, 0.0f);
  EXPECT_TRUE(plan.profile.filled);
}

TEST(Auto, SpreadSpectrumKeepsEveryDimension) {
  std::mt19937_64 rng(3);
  std::normal_distribution<float> normal;
  std::vector<float> base(3000 * 512);
  for (float& v : base) v = normal(rng);
  const AutoPlan plan = plan_auto(base.data(), 3000, 512);
  EXPECT_EQ(plan.params.nav_dim, 0u);
  EXPECT_NE(why(plan.profile, "nav_dim").find("only"), std::string::npos);
}

TEST(Auto, FixedSettingsWin) {
  const auto base = low_rank(2000, 512, 16, 4);
  AutoOptions options;
  options.fixed.nav_dim = 0;
  options.fixed.refine = false;
  options.fixed.rerank = 40;
  const AutoPlan plan = plan_auto(base.data(), 2000, 512, options);
  EXPECT_EQ(plan.params.nav_dim, 0u);
  EXPECT_FALSE(plan.profile.refined);
  EXPECT_EQ(plan.profile.rerank, 40u);
  EXPECT_EQ(why(plan.profile, "nav_dim"), "set by you");
  options.fixed.nav_dim = 600;
  EXPECT_THROW(plan_auto(base.data(), 2000, 512, options), std::invalid_argument);
}

TEST(Auto, RejectsBadInput) {
  const std::vector<float> one(8, 1.0f);
  EXPECT_THROW(plan_auto(nullptr, 0, 8), std::invalid_argument);
  AutoOptions options;
  options.recall = 1.5f;
  EXPECT_THROW(plan_auto(one.data(), 1, 8, options), std::invalid_argument);
}

TEST(Auto, BuildReachesTheRecallTargetOnUnseenQueries) {
  const auto data = make_clustered(6000, 200, 48, 30, 5);
  AutoOptions options;
  options.threads = 4;
  const auto index = build_auto(data.base.data.data(), nullptr, 6000, 48, options);
  const auto& p = index->profile();
  EXPECT_EQ(index->live_count(), 6000u);
  EXPECT_GE(p.recall_measured, 0.95f);
  EXPECT_GE(p.ef, 10u);
  EXPECT_FALSE(p.machine.empty());
  EXPECT_GE(recall_on(*index, data.base.data, data.queries.data, 48, Metric::L2), 0.92);
}

TEST(Auto, CalibratedBeamIsNeverBelowK) {
  // Easy data reaches the target at any beam; one below k was never measured (the search widens it to k).
  const auto data = make_clustered(300, 20, 32, 20, 2);
  const auto index = build_auto(data.base.data.data(), nullptr, 300, 32);
  EXPECT_GE(index->profile().ef, index->profile().k);
}

TEST(Auto, BuildWithDirectionsAndCosine) {
  const auto base = low_rank(3000, 512, 16, 6);
  const auto queries = low_rank(100, 512, 16, 6);  // same seed: the same subspace and clusters
  AutoOptions options;
  options.metric = Metric::Cosine;
  options.threads = 4;
  const auto index = build_auto(base.data(), nullptr, 3000, 512, options);
  EXPECT_GT(index->params().nav_dim, 0u);
  EXPECT_GE(index->profile().recall_measured, 0.95f);
  EXPECT_GE(recall_on(*index, base, queries, 512, Metric::Cosine), 0.9);
}

TEST(Auto, CalibratesOnGivenQueriesAndKeepsLabels) {
  const auto data = make_clustered(3000, 100, 32, 20, 7);
  std::vector<std::uint64_t> labels(3000);
  for (std::size_t i = 0; i < labels.size(); ++i) labels[i] = 1000 + 3 * i;
  AutoOptions options;
  options.queries = data.queries.data.data();
  options.query_count = 100;
  options.recall = 0.9f;
  options.threads = 2;
  const auto index = build_auto(data.base.data.data(), labels.data(), 3000, 32, options);
  EXPECT_GE(index->profile().recall_measured, 0.9f);
  const auto found = index->search(data.base.row(17), 1, {.ef = 64});
  ASSERT_EQ(found.size(), 1u);
  EXPECT_EQ(found[0].label, 1000u + 3 * 17);
}

TEST(Auto, FixedEfAndPrefetchSkipMeasurement) {
  const auto data = make_clustered(2000, 0, 32, 20, 8);
  AutoOptions options;
  options.fixed.ef = 40;
  options.fixed.prefetch = std::make_pair<std::size_t, std::size_t>(4, 1);
  const auto index = build_auto(data.base.data.data(), nullptr, 2000, 32, options);
  EXPECT_EQ(index->profile().ef, 40u);
  EXPECT_EQ(index->profile().recall_target, 0.0f);
  EXPECT_EQ(index->prefetch_settings(), (std::pair<std::size_t, std::size_t>(4, 1)));
}

TEST(Auto, ProfileSurvivesSaveLoadAndCompaction) {
  const auto data = make_clustered(3000, 0, 32, 20, 9);
  AutoOptions options;
  options.threads = 2;
  const auto index = build_auto(data.base.data.data(), nullptr, 3000, 32, options);
  const auto path = (std::filesystem::temp_directory_path() / "epzero_auto_profile.idx").string();
  index->save(path);
  const auto loaded = HnswIndex::load(path);
  std::filesystem::remove(path);
  const auto& a = index->profile();
  const auto& b = loaded->profile();
  EXPECT_EQ(a.ef, b.ef);
  EXPECT_EQ(a.k, b.k);
  EXPECT_EQ(a.rerank, b.rerank);
  EXPECT_EQ(a.recall_measured, b.recall_measured);
  EXPECT_EQ(a.calibrated_size, b.calibrated_size);
  EXPECT_EQ(a.machine, b.machine);
  EXPECT_EQ(a.notes, b.notes);
  EXPECT_TRUE(b.refined);
  // The same machine keeps the timed setting; where the CPU can't be named (WebAssembly: "/wasm") a
  // loaded index can't tell it's the same machine, and keeps the default.
  const bool named = !a.machine.empty() && a.machine.front() != '/';
  const HnswIndex defaults(index->params());
  EXPECT_EQ(loaded->prefetch_settings(), named ? index->prefetch_settings() : defaults.prefetch_settings());
  index->remove(5);
  const auto compact = index->compacted(2);
  EXPECT_EQ(compact->profile().ef, a.ef);
  EXPECT_EQ(compact->prefetch_settings(), index->prefetch_settings());
}

TEST(Auto, RecalibrateAfterGrowthAndHealthAdvice) {
  const auto data = make_clustered(8000, 300, 32, 30, 11);
  AutoOptions options;
  options.threads = 2;
  options.capacity = 8000;
  auto index = build_auto(data.base.data.data(), nullptr, 2000, 32, options);
  // A quarter grown is fine; four times the vectors is not, and health says so.
  for (std::size_t i = 2000; i < 8000; ++i) index->add(i, data.base.row(i));
  auto health = check_health(*index);
  EXPECT_EQ(health.live, 8000u);
  EXPECT_NEAR(health.growth, 3.0, 1e-9);
  ASSERT_EQ(health.advice.size(), 1u);
  EXPECT_NE(health.advice[0].find("recalibrate"), std::string::npos);

  const auto [ef, reached] = recalibrate(*index, data.queries.data.data(), 300, 0.95f, 2);
  EXPECT_EQ(index->profile().ef, ef);
  EXPECT_GE(reached, 0.95f);
  EXPECT_EQ(index->profile().calibrated_size, 8000u);
  EXPECT_TRUE(check_health(*index).advice.empty());

  for (std::uint64_t i = 0; i < 2500; ++i) index->remove(i);
  health = check_health(*index);
  ASSERT_FALSE(health.advice.empty());
  EXPECT_NE(health.advice[0].find("compact"), std::string::npos);
}

TEST(Auto, HealthNoticesDataTheDirectionsDontFit) {
  // Built on data from one subspace, then flooded with data from another.
  auto first = low_rank(3000, 512, 16, 12);
  const auto second = low_rank(9000, 512, 16, 13);
  AutoOptions options;
  options.threads = 4;
  options.capacity = 12000;
  auto index = build_auto(first.data(), nullptr, 3000, 512, options);
  ASSERT_GT(index->params().nav_dim, 0u);
  EXPECT_GT(index->profile().nav_variance, 0.8f);
  EXPECT_NEAR(nav_variance(*index), index->profile().nav_variance, 0.03);
  for (std::size_t i = 0; i < 9000; ++i) index->add(3000 + i, second.data() + i * 512);
  const auto health = check_health(*index);
  EXPECT_LT(health.nav_variance_now, health.nav_variance_built - 0.05);
  EXPECT_TRUE(std::any_of(health.advice.begin(), health.advice.end(),
                          [](const std::string& a) { return a.find("rebuild") != std::string::npos; }));
}

TEST(Auto, CheckQueriesSeesAShift) {
  const auto data = make_clustered(4000, 200, 32, 20, 16);
  AutoOptions options;
  options.threads = 2;
  const auto index = build_auto(data.base.data.data(), nullptr, 4000, 32, options);
  const auto same = check_queries(*index, data.queries.data.data(), 200);
  EXPECT_FALSE(same.shifted);
  EXPECT_TRUE(same.advice.empty());
  std::vector<float> far = data.queries.data;
  std::mt19937_64 rng(4);
  std::normal_distribution<float> noise(0.0f, 4.0f);
  for (float& v : far) v += noise(rng);
  const auto shifted = check_queries(*index, far.data(), 200);
  EXPECT_TRUE(shifted.shifted);
  EXPECT_GT(shifted.ratio, 1.5);
  ASSERT_EQ(shifted.advice.size(), 1u);
  EXPECT_THROW(check_queries(*index, nullptr, 0), std::invalid_argument);
}

TEST(Auto, GetVectorReturnsWhatIsStored) {
  const auto data = make_clustered(500, 0, 16, 5, 14);
  HnswParams params;
  params.dim = 16;
  params.max_elements = 500;
  params.metric = Metric::Cosine;
  HnswIndex index(params);
  index.add(7, data.base.row(3));
  std::vector<float> out(16), expected(data.base.row(3), data.base.row(3) + 16);
  normalize(expected.data(), 16);
  ASSERT_TRUE(index.get_vector(7, out.data()));
  for (std::size_t d = 0; d < 16; ++d) EXPECT_FLOAT_EQ(out[d], expected[d]);
  EXPECT_FALSE(index.get_vector(8, out.data()));
  EXPECT_EQ(index.live_labels(), std::vector<std::uint64_t>{7});
}

// A file the released 0.4.0 wrote (format v4, before the storage flags and the v5 profile fields):
// every release must keep reading it. Written by epzero 0.4.0 from PyPI:
//   x = np.random.default_rng(7).normal(size=(300, 16)).astype(np.float32)
//   epzero.build(x, metric="cosine", ef=24, prefetch=(8, 2), threads=1, seed=5).save(...)
// and its search(x[:3], k=3) returned [[0, 6, 23], [1, 162, 101], [2, 118, 36]].
TEST(Auto, ReadsTheFormatThatVersion040Wrote) {
  const auto index = HnswIndex::load(std::string(EPZERO_TEST_DATA) + "/v4-from-0.4.0.epzero");
  const auto& p = index->profile();
  EXPECT_EQ(index->live_count(), 300u);
  EXPECT_EQ(index->params().metric, Metric::Cosine);
  EXPECT_EQ(p.ef, 24u);
  EXPECT_TRUE(p.refined);
  EXPECT_EQ(p.calibrated_size, 0u);  // not in v4
  EXPECT_FALSE(index->params().half_vectors);
  const std::vector<std::vector<std::uint64_t>> expected = {{0, 6, 23}, {1, 162, 101}, {2, 118, 36}};
  std::vector<float> query(16);
  for (std::uint64_t q = 0; q < 3; ++q) {
    ASSERT_TRUE(index->get_vector(q, query.data()));
    const auto found = index->search(query.data(), 3, {.ef = p.ef, .rerank = p.rerank, .rerank_z = p.rerank_z});
    ASSERT_EQ(found.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) EXPECT_EQ(found[i].label, expected[q][i]);
  }
}

TEST(Auto, ProfiledIndexesWriteVersion5) {
  const auto data = make_clustered(500, 0, 16, 5, 15);
  AutoOptions options;
  options.threads = 1;
  const auto index = build_auto(data.base.data.data(), nullptr, 500, 16, options);
  const auto path = (std::filesystem::temp_directory_path() / "epzero_v5.idx").string();
  index->save(path);
  std::ifstream in(path, std::ios::binary);
  char header[12];
  in.read(header, sizeof(header));
  in.close();
  std::filesystem::remove(path);
  std::uint32_t version;
  std::memcpy(&version, header + 8, sizeof(version));
  EXPECT_EQ(version, 5u);
}

TEST(Auto, PlainIndexesKeepTheOlderFormat) {
  HnswParams params;
  params.dim = 8;
  params.max_elements = 10;
  HnswIndex index(params);
  const std::vector<float> v(8, 1.0f);
  index.add(1, v.data());
  const auto path = (std::filesystem::temp_directory_path() / "epzero_plain_format.idx").string();
  index.save(path);
  std::ifstream in(path, std::ios::binary);
  char header[12];
  in.read(header, sizeof(header));
  in.close();
  std::filesystem::remove(path);
  std::uint32_t version;
  std::memcpy(&version, header + 8, sizeof(version));
  EXPECT_EQ(version, 2u);  // readable by older builds
}

}  // namespace
}  // namespace epzero
