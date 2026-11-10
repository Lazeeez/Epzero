#include <gtest/gtest.h>

#include <cstring>
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
  EXPECT_EQ(a.machine, b.machine);
  EXPECT_EQ(a.notes, b.notes);
  EXPECT_TRUE(b.refined);
  EXPECT_EQ(loaded->prefetch_settings(), index->prefetch_settings());  // the same machine
  index->remove(5);
  const auto compact = index->compacted(2);
  EXPECT_EQ(compact->profile().ef, a.ef);
  EXPECT_EQ(compact->prefetch_settings(), index->prefetch_settings());
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
