#include <gtest/gtest.h>

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "oryx/brute_force.h"
#include "oryx/dataset.h"
#include "oryx/hnsw.h"

namespace oryx {
namespace {

constexpr std::size_t kDim = 32;
constexpr std::size_t kBase = 5000;
constexpr std::size_t kQueries = 200;
constexpr std::size_t kK = 10;

HnswParams params_for(Metric metric, std::size_t max_elements = kBase) {
  HnswParams p;
  p.dim = kDim;
  p.metric = metric;
  p.max_elements = max_elements;
  p.M = 16;
  p.ef_construction = 100;
  return p;
}

double measure_recall(const HnswIndex& index, const SyntheticData& data, std::size_t ef, Metric metric) {
  std::vector<std::vector<SearchResult>> results;
  for (std::size_t q = 0; q < data.queries.n; ++q) {
    results.push_back(index.search(data.queries.row(q), kK, {.ef = ef}));
  }
  const auto truth = compute_ground_truth(data.base.data.data(), data.base.n, kDim, data.queries.data.data(),
                                          data.queries.n, kK, metric);
  return recall_at_k(results, truth, kK);
}

class HnswTest : public ::testing::Test {
 protected:
  void SetUp() override { data_ = make_clustered(kBase, kQueries, kDim, 50, 1); }
  SyntheticData data_;
};

TEST_F(HnswTest, HighRecallOnL2) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  EXPECT_EQ(index.size(), kBase);
  EXPECT_GE(measure_recall(index, data_, 100, Metric::L2), 0.95);
}

TEST_F(HnswTest, RecallImprovesWithEf) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  const double low = measure_recall(index, data_, 10, Metric::L2);
  const double high = measure_recall(index, data_, 200, Metric::L2);
  EXPECT_GE(high, low);
  EXPECT_GE(high, 0.98);
}

TEST_F(HnswTest, CosineFindsItselfFirst) {
  HnswIndex index(params_for(Metric::Cosine));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  for (std::size_t i = 0; i < 100; ++i) {
    // Scaling must not matter for cosine.
    std::vector<float> scaled(data_.base.row(i), data_.base.row(i) + kDim);
    for (float& x : scaled) x *= 3.5f;
    const auto results = index.search(scaled.data(), 1, {.ef = 50});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].label, i);
    EXPECT_NEAR(results[0].distance, 0.0f, 1e-4f);
  }
}

TEST_F(HnswTest, ResultsAreSortedAndBounded) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < 20; ++i) index.add(i, data_.base.row(i));
  const auto results = index.search(data_.queries.row(0), 50, {.ef = 64});
  ASSERT_EQ(results.size(), 20u);  // k larger than the index returns everything
  for (std::size_t i = 1; i < results.size(); ++i) EXPECT_LE(results[i - 1].distance, results[i].distance);
}

TEST(Hnsw, EmptyIndexAndZeroK) {
  HnswIndex index(params_for(Metric::L2, 10));
  const std::vector<float> q(kDim, 0.5f);
  EXPECT_TRUE(index.search(q.data(), 5).empty());
  EXPECT_EQ(index.max_level(), -1);
  index.add(7, q.data());
  EXPECT_TRUE(index.search(q.data(), 0).empty());
  const auto one = index.search(q.data(), 5);
  ASSERT_EQ(one.size(), 1u);
  EXPECT_EQ(one[0].label, 7u);
}

TEST(Hnsw, RejectsDuplicatesAndOverflow) {
  HnswIndex index(params_for(Metric::L2, 2));
  const std::vector<float> v(kDim, 1.0f);
  index.add(1, v.data());
  EXPECT_THROW(index.add(1, v.data()), std::invalid_argument);
  index.add(2, v.data());
  EXPECT_THROW(index.add(3, v.data()), std::length_error);
}

TEST(Hnsw, RejectsInvalidParams) {
  HnswParams p = params_for(Metric::L2);
  p.dim = 0;
  EXPECT_THROW(HnswIndex{p}, std::invalid_argument);
  p = params_for(Metric::L2);
  p.M = 1;
  EXPECT_THROW(HnswIndex{p}, std::invalid_argument);
}

TEST_F(HnswTest, GraphInvariantsHold) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  for (std::size_t i = 0; i < kBase; ++i) {
    const int level = index.level_of(i);
    ASSERT_LE(level, index.max_level());
    for (int layer = 0; layer <= level; ++layer) {
      const auto links = index.neighbors_of(i, layer);
      EXPECT_LE(links.size(), layer == 0 ? 32u : 16u);
      for (std::uint64_t n : links) {
        EXPECT_NE(n, i) << "self loop";
        EXPECT_GE(index.level_of(n), layer) << "link to a node that does not exist on this layer";
      }
    }
  }
}

TEST_F(HnswTest, TraceDescendsLayersAndMatchesStats) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  SearchTrace trace;
  SearchStats stats;
  const auto results = index.search(data_.queries.row(0), kK, {.ef = 64, .stats = &stats, .trace = &trace});
  ASSERT_FALSE(trace.events.empty());
  EXPECT_EQ(trace.entry_layer, static_cast<std::uint32_t>(index.max_level()));
  for (std::size_t i = 1; i < trace.events.size(); ++i) {
    EXPECT_LE(trace.events[i].layer, trace.events[i - 1].layer) << "search must never climb back up";
  }
  EXPECT_EQ(trace.events.back().layer, 0u);
  // Every scored neighbor is one event; the entry point of each phase is scored but not an event.
  EXPECT_EQ(stats.distance_computations, trace.events.size() + 2);

  // Tracing must not change the answer.
  const auto untraced = index.search(data_.queries.row(0), kK, {.ef = 64});
  ASSERT_EQ(untraced.size(), results.size());
  for (std::size_t i = 0; i < results.size(); ++i) EXPECT_EQ(untraced[i].label, results[i].label);
}

TEST_F(HnswTest, ConcurrentInsertAndSearch) {
  HnswIndex index(params_for(Metric::L2));
  std::atomic<std::size_t> next{0};
  std::atomic<bool> done{false};
  std::vector<std::thread> writers;
  for (int t = 0; t < 8; ++t) {
    writers.emplace_back([&] {
      for (std::size_t i = next++; i < kBase; i = next++) index.add(i, data_.base.row(i));
    });
  }
  // Readers run while the graph is being built; they must never crash or see torn state.
  std::thread reader([&] {
    while (!done) (void)index.search(data_.queries.row(0), kK, {.ef = 32});
  });
  for (auto& w : writers) w.join();
  done = true;
  reader.join();

  EXPECT_EQ(index.size(), kBase);
  EXPECT_GE(measure_recall(index, data_, 100, Metric::L2), 0.95);
}

TEST_F(HnswTest, SaveLoadRoundTrip) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i * 10, data_.base.row(i));
  const auto path = (std::filesystem::temp_directory_path() / "oryx_roundtrip.idx").string();
  index.save(path);
  const auto loaded = HnswIndex::load(path);
  std::filesystem::remove(path);

  EXPECT_EQ(loaded->size(), index.size());
  EXPECT_EQ(loaded->max_level(), index.max_level());
  for (std::size_t q = 0; q < 50; ++q) {
    const auto a = index.search(data_.queries.row(q), kK, {.ef = 64});
    const auto b = loaded->search(data_.queries.row(q), kK, {.ef = 64});
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      EXPECT_EQ(a[i].label, b[i].label);
      EXPECT_EQ(a[i].distance, b[i].distance);
    }
  }
  EXPECT_THROW(loaded->add(0, data_.base.row(0)), std::invalid_argument);  // label map restored
}

TEST(Hnsw, LoadRejectsGarbage) {
  const auto path = (std::filesystem::temp_directory_path() / "oryx_garbage.idx").string();
  {
    std::ofstream out(path, std::ios::binary);
    out << "definitely not an index";
  }
  EXPECT_THROW(HnswIndex::load(path), std::runtime_error);
  std::filesystem::remove(path);
}

}  // namespace
}  // namespace oryx
