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

// ---------- SQ8 quantization ----------

HnswParams sq8_params(Metric metric, bool keep_vectors, std::size_t max_elements = kBase) {
  HnswParams p = params_for(metric, max_elements);
  p.quantization = Quantization::SQ8;
  p.keep_vectors = keep_vectors;
  return p;
}

TEST_F(HnswTest, Sq8RequiresTraining) {
  HnswIndex index(sq8_params(Metric::L2, true));
  EXPECT_FALSE(index.trained());
  EXPECT_THROW(index.add(0, data_.base.row(0)), std::logic_error);
  index.train(data_.base.data.data(), 500);
  EXPECT_TRUE(index.trained());
  index.add(0, data_.base.row(0));
  EXPECT_THROW(index.train(data_.base.data.data(), 500), std::logic_error);  // only before adds
}

TEST_F(HnswTest, Sq8WithRerankKeepsHighRecall) {
  HnswIndex index(sq8_params(Metric::L2, true));
  index.train(data_.base.data.data(), kBase);
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  EXPECT_GE(measure_recall(index, data_, 100, Metric::L2), 0.95);
  // Re-ranked distances are exact.
  const auto results = index.search(data_.queries.row(0), kK, {.ef = 100});
  const auto exact = brute_force_search(data_.base.data.data(), kBase, kDim, data_.queries.row(0), 1, Metric::L2);
  EXPECT_EQ(results[0].label, exact[0].label);
  EXPECT_NEAR(results[0].distance, exact[0].distance, 1e-3f);
}

TEST_F(HnswTest, Sq8WithoutVectorsStillSearchesWell) {
  HnswIndex index(sq8_params(Metric::L2, false));
  index.train(data_.base.data.data(), kBase);
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  EXPECT_GE(measure_recall(index, data_, 100, Metric::L2), 0.85);
  HnswIndex full(params_for(Metric::L2));
  EXPECT_LT(index.memory_bytes(), full.memory_bytes());
}

TEST_F(HnswTest, Sq8CosineFindsItself) {
  HnswIndex index(sq8_params(Metric::Cosine, true));
  index.train(data_.base.data.data(), kBase);
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  for (std::size_t i = 0; i < 50; ++i) {
    EXPECT_EQ(index.search(data_.base.row(i), 1, {.ef = 64})[0].label, i);
  }
}

TEST_F(HnswTest, Sq8SaveLoadRoundTrip) {
  HnswIndex index(sq8_params(Metric::L2, true));
  index.train(data_.base.data.data(), kBase);
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  const auto path = (std::filesystem::temp_directory_path() / "oryx_sq8.idx").string();
  index.save(path);
  const auto loaded = HnswIndex::load(path);
  std::filesystem::remove(path);
  EXPECT_EQ(loaded->params().quantization, Quantization::SQ8);
  for (std::size_t q = 0; q < 50; ++q) {
    const auto a = index.search(data_.queries.row(q), kK, {.ef = 64});
    const auto b = loaded->search(data_.queries.row(q), kK, {.ef = 64});
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].label, b[i].label);
  }
}

// ---------- Deletes ----------

TEST_F(HnswTest, RemovedLabelsAreNeverReturned) {
  HnswIndex index(params_for(Metric::L2, kBase + 10));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  // Delete every third vector, then check results against exact search over the survivors.
  std::vector<float> live;
  std::vector<std::uint64_t> live_labels;
  for (std::size_t i = 0; i < kBase; ++i) {
    if (i % 3 == 0) {
      EXPECT_TRUE(index.remove(i));
    } else {
      live.insert(live.end(), data_.base.row(i), data_.base.row(i) + kDim);
      live_labels.push_back(i);
    }
  }
  EXPECT_FALSE(index.remove(0));  // already gone
  EXPECT_FALSE(index.contains(0));
  EXPECT_EQ(index.live_count(), live_labels.size());

  std::vector<std::vector<SearchResult>> results;
  std::vector<std::vector<std::uint32_t>> truth;
  for (std::size_t q = 0; q < kQueries; ++q) {
    auto r = index.search(data_.queries.row(q), kK, {.ef = 100});
    ASSERT_EQ(r.size(), kK);
    for (const auto& hit : r) EXPECT_NE(hit.label % 3, 0u) << "deleted label returned";
    results.push_back(std::move(r));
    std::vector<std::uint32_t> t;
    for (const auto& e : brute_force_search(live.data(), live_labels.size(), kDim, data_.queries.row(q), kK, Metric::L2)) {
      t.push_back(static_cast<std::uint32_t>(live_labels[e.label]));
    }
    truth.push_back(std::move(t));
  }
  EXPECT_GE(recall_at_k(results, truth, kK), 0.95);

  // A removed label can be inserted again (as a new node).
  index.add(0, data_.base.row(0));
  EXPECT_EQ(index.search(data_.base.row(0), 1, {.ef = 64})[0].label, 0u);
}

TEST_F(HnswTest, TombstonesSurviveSaveAndCompaction) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  for (std::size_t i = 0; i < kBase; i += 2) index.remove(i);

  const auto path = (std::filesystem::temp_directory_path() / "oryx_tombstones.idx").string();
  index.save(path);
  const auto loaded = HnswIndex::load(path);
  std::filesystem::remove(path);
  EXPECT_EQ(loaded->deleted_count(), index.deleted_count());
  EXPECT_FALSE(loaded->contains(0));
  EXPECT_TRUE(loaded->contains(1));

  const auto compact = loaded->compacted(4);
  EXPECT_EQ(compact->size(), kBase / 2);
  EXPECT_EQ(compact->deleted_count(), 0u);
  for (std::size_t q = 0; q < 20; ++q) {
    for (const auto& hit : compact->search(data_.queries.row(q), kK, {.ef = 64})) EXPECT_EQ(hit.label % 2, 1u);
  }
}

TEST_F(HnswTest, ConcurrentInsertRemoveAndSearch) {
  HnswIndex index(params_for(Metric::L2));
  std::atomic<std::size_t> next{0};
  std::atomic<bool> done{false};
  std::vector<std::thread> threads;
  for (int t = 0; t < 6; ++t) {
    threads.emplace_back([&] {
      for (std::size_t i = next++; i < kBase; i = next++) {
        index.add(i, data_.base.row(i));
        if (i % 5 == 0 && i > 0) index.remove(i - 1);
      }
    });
  }
  std::thread reader([&] {
    while (!done) {
      for (const auto& hit : index.search(data_.queries.row(0), kK, {.ef = 32})) (void)hit;
    }
  });
  for (auto& t : threads) t.join();
  done = true;
  reader.join();
  EXPECT_EQ(index.size(), kBase);
  EXPECT_GT(index.deleted_count(), 0u);
  for (const auto& hit : index.search(data_.queries.row(1), kK, {.ef = 64})) EXPECT_TRUE(index.contains(hit.label));
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
