#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

#include "../src/half.h"
#include "epzero/brute_force.h"
#include "epzero/dataset.h"
#include "epzero/distance.h"
#include "epzero/hnsw.h"

namespace epzero {
namespace {

TEST(Half, EveryHalfRoundTrips) {
  for (std::uint32_t h = 0; h <= 0xffffu; ++h) {
    const auto bits = static_cast<std::uint16_t>(h);
    const float f = half::to_float(bits);
    if (std::isnan(f)) {
      EXPECT_TRUE(std::isnan(half::to_float(half::from_float(f))));
    } else {
      EXPECT_EQ(half::from_float(f), bits) << h;
    }
  }
}

TEST(Half, FloatsRoundToTheNearestHalf) {
  std::mt19937_64 rng(1);
  std::uniform_real_distribution<float> exponent(-26.0f, 16.5f);
  std::uniform_int_distribution<int> sign(0, 1);
  for (int i = 0; i < 200000; ++i) {
    const float f = (sign(rng) ? -1.0f : 1.0f) * std::exp2(exponent(rng));
    const std::uint16_t h = half::from_float(f);
    const float back = half::to_float(h);
    if (std::abs(f) >= 65520.0f) {
      EXPECT_TRUE(std::isinf(back));
      continue;
    }
    // No other half is closer: the neighbours on either side are at least as far.
    const float error = std::abs(back - f);
    for (const int step : {-1, 1}) {
      const auto neighbour = static_cast<std::uint16_t>((h & 0x7fffu) == 0 && step < 0 ? h ^ 0x8000u : h + step);
      const float other = half::to_float(neighbour);
      if (!std::isfinite(other)) continue;
      EXPECT_LE(error, std::abs(other - f)) << f;
    }
  }
  EXPECT_EQ(half::from_float(0.0f), 0u);
  EXPECT_EQ(half::from_float(-0.0f), 0x8000u);
  EXPECT_EQ(half::from_float(std::numeric_limits<float>::infinity()), 0x7c00u);
  EXPECT_EQ(half::from_float(65504.0f), 0x7bffu);
}

TEST(Half, KernelsMatchTheFloatKernels) {
  std::mt19937_64 rng(2);
  std::normal_distribution<float> normal;
  for (const std::size_t dim : {1u, 7u, 8u, 15u, 16u, 17u, 96u, 100u, 960u, 1536u}) {
    std::vector<float> a(dim), b(dim), rounded(dim);
    std::vector<std::uint16_t> h(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      a[i] = normal(rng);
      b[i] = normal(rng);
      h[i] = half::from_float(b[i]);
      rounded[i] = half::to_float(h[i]);
    }
    for (const Metric metric : {Metric::L2, Metric::InnerProduct}) {
      const float expected = get_distance_fn(metric, SimdLevel::Scalar)(a.data(), rounded.data(), dim);
      for (const SimdLevel level : {SimdLevel::Scalar, SimdLevel::AVX2, SimdLevel::AVX512, SimdLevel::NEON}) {
        if (!simd_level_supported(level)) continue;
        const float got = get_half_distance_fn(metric, level)(a.data(), h.data(), dim);
        EXPECT_NEAR(got, expected, 1e-4f * (1.0f + std::abs(expected))) << dim << " " << to_string(level);
      }
    }
  }
}

class HalfIndexTest : public ::testing::TestWithParam<Metric> {};

TEST_P(HalfIndexTest, SameRecallAtHalfTheVectorMemory) {
  const Metric metric = GetParam();
  const auto data = make_clustered(6000, 200, 64, 30, 3);
  auto build = [&](bool half) {
    HnswParams p;
    p.dim = 64;
    p.metric = metric;
    p.max_elements = 6000;
    p.quantization = Quantization::SQ8;
    p.half_vectors = half;
    auto index = std::make_unique<HnswIndex>(p);
    index->train(data.base.data.data(), 6000);
    for (std::size_t i = 0; i < 6000; ++i) index->add(i, data.base.row(i));
    return index;
  };
  const auto full = build(false);
  const auto half = build(true);
  std::vector<float> base = data.base.data;
  if (metric == Metric::Cosine) {
    for (std::size_t i = 0; i < 6000; ++i) normalize(base.data() + i * 64, 64);
  }
  const auto truth = compute_ground_truth(base.data(), 6000, 64, data.queries.data.data(), 200, 10, metric);
  auto recall = [&](const HnswIndex& index) {
    std::vector<std::vector<SearchResult>> results;
    for (std::size_t q = 0; q < 200; ++q) results.push_back(index.search(data.queries.row(q), 10, {.ef = 64}));
    return recall_at_k(results, truth, 10);
  };
  EXPECT_NEAR(recall(*half), recall(*full), 0.01);
  EXPECT_GT(recall(*half), 0.9);
  EXPECT_EQ(full->memory_bytes() - half->memory_bytes(), 6000u * 64 * 2);
}

INSTANTIATE_TEST_SUITE_P(Metrics, HalfIndexTest, ::testing::Values(Metric::L2, Metric::Cosine));

TEST(Half, SaveLoadCompactAndGetVector) {
  const auto data = make_clustered(2000, 0, 32, 20, 4);
  HnswParams p;
  p.dim = 32;
  p.max_elements = 2000;
  p.quantization = Quantization::SQ8;
  p.half_vectors = true;
  HnswIndex index(p);
  index.train(data.base.data.data(), 2000);
  for (std::size_t i = 0; i < 2000; ++i) index.add(i, data.base.row(i));
  const auto path = (std::filesystem::temp_directory_path() / "epzero_half.idx").string();
  index.save(path);
  const auto loaded = HnswIndex::load(path);
  std::filesystem::remove(path);
  EXPECT_TRUE(loaded->params().half_vectors);
  std::vector<float> a(32), b(32);
  ASSERT_TRUE(index.get_vector(9, a.data()));
  ASSERT_TRUE(loaded->get_vector(9, b.data()));
  EXPECT_EQ(a, b);
  for (std::size_t d = 0; d < 32; ++d) EXPECT_NEAR(a[d], data.base.row(9)[d], 1e-3f * (1.0f + std::abs(a[d])));
  const auto query = data.base.row(11);
  EXPECT_EQ(loaded->search(query, 5, {.ef = 64})[0].label, 11u);
  index.remove(3);
  const auto compact = index.compacted(2);
  EXPECT_TRUE(compact->params().half_vectors);
  EXPECT_EQ(compact->live_count(), 1999u);
  EXPECT_EQ(compact->search(query, 1, {.ef = 64})[0].label, 11u);
}

TEST(Half, NeedsQuantizedIndexWithVectors) {
  HnswParams p;
  p.dim = 8;
  p.max_elements = 10;
  p.half_vectors = true;
  EXPECT_THROW(HnswIndex{p}, std::invalid_argument);  // no quantization: the vectors are the graph's records
  p.quantization = Quantization::SQ8;
  p.keep_vectors = false;
  EXPECT_THROW(HnswIndex{p}, std::invalid_argument);
}

}  // namespace
}  // namespace epzero
