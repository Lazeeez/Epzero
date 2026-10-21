#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
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

// Prefetching only changes when memory is read, never what a search returns.
TEST_F(HnswTest, PrefetchSettingsDontChangeResults) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  std::vector<std::vector<SearchResult>> before;
  for (std::size_t q = 0; q < kQueries; ++q) before.push_back(index.search(data_.queries.row(q), 10, {.ef = 64}));
  const auto [depth, lines] = index.tune_prefetch(data_.queries.row(0), 50);
  EXPECT_GE(lines, 1u);
  EXPECT_EQ(index.prefetch_settings(), std::make_pair(depth, lines));
  for (const auto& setting : {std::pair<std::size_t, std::size_t>{2, 4}, {1, 1}, {8, 2}, {depth, lines}}) {
    index.set_prefetch(setting.first, setting.second);
    for (std::size_t q = 0; q < kQueries; ++q) {
      const auto after = index.search(data_.queries.row(q), 10, {.ef = 64});
      ASSERT_EQ(after.size(), before[q].size());
      for (std::size_t i = 0; i < after.size(); ++i) EXPECT_EQ(after[i].label, before[q][i].label);
    }
  }
}

// A short patience stops early; an unbounded one is exactly the plain search.
TEST_F(HnswTest, PatienceStopsEarlyAndConvergesToThePlainSearch) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  double cost_patient = 0, cost_plain = 0;
  for (std::size_t q = 0; q < kQueries; ++q) {
    SearchStats plain_stats, patient_stats;
    const auto plain = index.search(data_.queries.row(q), 10, {.ef = 128, .stats = &plain_stats});
    const auto patient = index.search(data_.queries.row(q), 10, {.ef = 128, .patience = 4, .stats = &patient_stats});
    const auto unbounded = index.search(data_.queries.row(q), 10, {.ef = 128, .patience = 1u << 30});
    EXPECT_LE(patient.size(), 10u);
    EXPECT_LE(patient_stats.distance_computations, plain_stats.distance_computations);
    ASSERT_EQ(unbounded.size(), plain.size());
    for (std::size_t i = 0; i < plain.size(); ++i) EXPECT_EQ(unbounded[i].label, plain[i].label);
    cost_patient += static_cast<double>(patient_stats.distance_computations);
    cost_plain += static_cast<double>(plain_stats.distance_computations);
  }
  EXPECT_LT(cost_patient, 0.9 * cost_plain);
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

TEST_F(HnswTest, GraphExportImportRoundTrip) {
  // Cosine, inserted on several threads so node ids differ from labels (the web demo's case).
  HnswIndex index(params_for(Metric::Cosine));
  std::vector<std::thread> workers;
  for (std::size_t t = 0; t < 4; ++t) {
    workers.emplace_back([&, t] {
      for (std::size_t i = t; i < kBase; i += 4) index.add(i, data_.base.row(i));
    });
  }
  for (auto& w : workers) w.join();

  const auto bytes = index.export_graph();
  EXPECT_LT(bytes.size(), kBase * kDim * sizeof(float));  // links only: smaller than the vectors alone
  const auto imported =
      HnswIndex::import_graph(params_for(Metric::Cosine), data_.base.data.data(), kBase, bytes.data(), bytes.size());
  EXPECT_EQ(imported->size(), index.size());
  EXPECT_EQ(imported->max_level(), index.max_level());
  for (std::size_t q = 0; q < 50; ++q) {
    const auto a = index.search(data_.queries.row(q), kK, {.ef = 64});
    const auto b = imported->search(data_.queries.row(q), kK, {.ef = 64});
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      EXPECT_EQ(a[i].label, b[i].label);
      EXPECT_EQ(a[i].distance, b[i].distance);
    }
  }
  EXPECT_THROW(imported->add(0, data_.base.row(0)), std::invalid_argument);  // label map restored
}

TEST_F(HnswTest, GraphImportRejectsCorruptOrMismatchedInput) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  auto bytes = index.export_graph();
  const float* vectors = data_.base.data.data();

  auto truncated = bytes;
  truncated.pop_back();
  EXPECT_THROW(HnswIndex::import_graph(params_for(Metric::L2), vectors, kBase, truncated.data(), truncated.size()),
               std::runtime_error);
  auto garbage = bytes;
  garbage[0] = 'X';
  EXPECT_THROW(HnswIndex::import_graph(params_for(Metric::L2), vectors, kBase, garbage.data(), garbage.size()),
               std::runtime_error);
  // Every link must point at a loaded node: overwrite the tail (links) with 0xff.
  auto bad_links = bytes;
  for (std::size_t i = bad_links.size() - 64; i < bad_links.size(); ++i) bad_links[i] = 0xff;
  EXPECT_THROW(HnswIndex::import_graph(params_for(Metric::L2), vectors, kBase, bad_links.data(), bad_links.size()),
               std::runtime_error);
  EXPECT_THROW(HnswIndex::import_graph(params_for(Metric::Cosine), vectors, kBase, bytes.data(), bytes.size()),
               std::runtime_error);  // built for L2
  EXPECT_THROW(HnswIndex::import_graph(params_for(Metric::L2), vectors, kBase - 1, bytes.data(), bytes.size()),
               std::runtime_error);  // fewer vectors than nodes
}

// Found by fuzzing: an upper-layer link to a node that only exists on layer 0. A search following
// it would read that node's (non-existent) upper-layer list.
TEST_F(HnswTest, GraphImportRejectsLinksToNodesNotOnThatLayer) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  auto bytes = index.export_graph();
  // export_graph's layout: a 57-byte header, then per node its label (u64), level (i8) and, for
  // each of its layers, a u8 link count and that many u16 links (fewer than 65,536 nodes).
  std::size_t pos = 57;
  std::size_t upper_link = 0;  // offset of a layer-1 link
  std::vector<int> levels;
  for (std::size_t id = 0; id < kBase; ++id) {
    pos += 8;
    const int level = static_cast<std::int8_t>(bytes[pos++]);
    levels.push_back(level);
    for (int layer = 0; layer <= level; ++layer) {
      const std::size_t count = bytes[pos++];
      if (layer == 1 && count > 0 && upper_link == 0) upper_link = pos;
      pos += 2 * count;
    }
  }
  ASSERT_EQ(pos, bytes.size());
  ASSERT_NE(upper_link, 0u);
  const auto base_only = static_cast<std::uint16_t>(std::find(levels.begin(), levels.end(), 0) - levels.begin());
  bytes[upper_link] = static_cast<std::uint8_t>(base_only & 0xff);
  bytes[upper_link + 1] = static_cast<std::uint8_t>(base_only >> 8);
  EXPECT_THROW(HnswIndex::import_graph(params_for(Metric::L2), data_.base.data.data(), kBase, bytes.data(), bytes.size()),
               std::runtime_error);
}

TEST_F(HnswTest, LoadRespectsAMemoryLimit) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < 100; ++i) index.add(i, data_.base.row(i));
  const auto path = (std::filesystem::temp_directory_path() / "oryx_limit.idx").string();
  index.save(path);
  // The index was made with room for kBase vectors: what loading it would allocate.
  const std::size_t needed = HnswIndex::estimated_bytes(params_for(Metric::L2));
  EXPECT_THROW(HnswIndex::load(path, needed / 2), std::runtime_error);
  EXPECT_EQ(HnswIndex::load(path, needed)->size(), 100u);
  EXPECT_EQ(HnswIndex::load(path)->size(), 100u);  // no limit
  std::filesystem::remove(path);
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

// ---------- RaBitQ quantization ----------

HnswParams rabitq_params(Metric metric, std::size_t max_elements = kBase) {
  HnswParams p = params_for(metric, max_elements);
  p.quantization = Quantization::RaBitQ;
  return p;
}

TEST_F(HnswTest, RabitqRequiresTrainingAndFullVectors) {
  HnswIndex index(rabitq_params(Metric::L2));
  EXPECT_FALSE(index.trained());
  EXPECT_THROW(index.add(0, data_.base.row(0)), std::logic_error);
  index.train(data_.base.data.data(), 500);
  EXPECT_TRUE(index.trained());
  index.add(0, data_.base.row(0));
  HnswParams without = rabitq_params(Metric::L2);
  without.keep_vectors = false;
  EXPECT_THROW(HnswIndex{without}, std::invalid_argument);
}

// The estimator is unbiased: averaged over many pairs, estimated distances track the true ones.
TEST_F(HnswTest, RabitqEstimatesDistances) {
  HnswIndex index(rabitq_params(Metric::L2));
  index.train(data_.base.data.data(), kBase);
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  // With re-ranking off (rerank = k), the top result's distance is still re-scored exactly, so
  // compare the beam's order instead: the exact nearest neighbour is among RaBitQ's top 10.
  std::size_t hits = 0;
  for (std::size_t q = 0; q < kQueries; ++q) {
    const auto exact = brute_force_search(data_.base.data.data(), kBase, kDim, data_.queries.row(q), 1, Metric::L2);
    for (const auto& r : index.search(data_.queries.row(q), 10, {.ef = 100})) hits += r.label == exact[0].label;
  }
  EXPECT_GE(static_cast<double>(hits) / kQueries, 0.97);
}

TEST_F(HnswTest, RabitqWithRerankKeepsHighRecall) {
  for (const Metric metric : {Metric::L2, Metric::InnerProduct}) {
    HnswIndex index(rabitq_params(metric));
    index.train(data_.base.data.data(), kBase);
    for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
    EXPECT_GE(measure_recall(index, data_, 160, metric), 0.95) << to_string(metric);
  }
  {
    // Cosine (the ground-truth helper wants normalized data, so check that vectors find themselves).
    HnswIndex index(rabitq_params(Metric::Cosine));
    index.train(data_.base.data.data(), kBase);
    for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
    for (std::size_t i = 0; i < 100; ++i) EXPECT_EQ(index.search(data_.base.row(i), 1, {.ef = 64})[0].label, i);
  }
  // Re-ranked distances are exact.
  HnswIndex index(rabitq_params(Metric::L2));
  index.train(data_.base.data.data(), kBase);
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  const auto results = index.search(data_.queries.row(0), kK, {.ef = 100});
  const auto exact = brute_force_search(data_.base.data.data(), kBase, kDim, data_.queries.row(0), 1, Metric::L2);
  EXPECT_EQ(results[0].label, exact[0].label);
  EXPECT_NEAR(results[0].distance, exact[0].distance, 1e-3f);
}

TEST_F(HnswTest, RabitqSaveLoadAndCompaction) {
  HnswIndex index(rabitq_params(Metric::Cosine));
  index.train(data_.base.data.data(), kBase);
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  for (std::size_t i = 0; i < kBase; i += 4) index.remove(i);
  const auto path = (std::filesystem::temp_directory_path() / "oryx_rabitq.idx").string();
  index.save(path);
  const auto loaded = HnswIndex::load(path);
  std::filesystem::remove(path);
  EXPECT_EQ(loaded->params().quantization, Quantization::RaBitQ);
  for (std::size_t q = 0; q < 50; ++q) {
    const auto a = index.search(data_.queries.row(q), kK, {.ef = 64});
    const auto b = loaded->search(data_.queries.row(q), kK, {.ef = 64});
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].label, b[i].label);
  }
  const auto compact = loaded->compacted(4);
  EXPECT_EQ(compact->size(), kBase - kBase / 4);
  for (std::size_t i = 1; i < 60; i += 4) EXPECT_EQ(compact->search(data_.base.row(i), 1, {.ef = 64})[0].label, i);
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

// Regression (found by LangChain's vector-store test suite): with every node deleted, new nodes
// found no live neighbour to link to and were unreachable, so every search came back empty.
TEST_F(HnswTest, InsertsAfterDeletingEverythingAreFound) {
  HnswIndex index(params_for(Metric::L2, 2 * kBase));
  const std::size_t n = 200;
  for (std::size_t i = 0; i < n; ++i) index.add(i, data_.base.row(i));
  for (std::size_t i = 0; i < n; ++i) ASSERT_TRUE(index.remove(i));
  for (std::size_t i = 0; i < n; ++i) index.add(n + i, data_.base.row(n + i));
  std::size_t found = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const auto hits = index.search(data_.base.row(n + i), 1, {.ef = 64});
    ASSERT_FALSE(hits.empty());
    EXPECT_GE(hits[0].label, n) << "deleted label returned";
    found += hits[0].label == n + i;
  }
  EXPECT_GE(found, n * 95 / 100);
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

// ---------------------------------------------------------------------------------------------
// Filtered search

// Exact top-k among the allowed labels, by brute force over just those rows.
std::vector<std::vector<std::uint32_t>> filtered_truth(const SyntheticData& data, const std::vector<std::uint64_t>& allowed,
                                                       Metric metric) {
  std::vector<float> rows;
  for (const auto label : allowed) rows.insert(rows.end(), data.base.row(label), data.base.row(label) + kDim);
  std::vector<std::vector<std::uint32_t>> truth;
  for (std::size_t q = 0; q < data.queries.n; ++q) {
    std::vector<std::uint32_t> t;
    for (const auto& e : brute_force_search(rows.data(), allowed.size(), kDim, data.queries.row(q), kK, metric)) {
      t.push_back(static_cast<std::uint32_t>(allowed[e.label]));
    }
    truth.push_back(std::move(t));
  }
  return truth;
}

// Runs every query under `strategy`, checks that only allowed labels come back, and returns recall.
double filtered_recall(const HnswIndex& index, const SyntheticData& data, const SearchFilter& filter,
                       const std::vector<std::uint64_t>& allowed, FilterStrategy strategy, std::size_t ef,
                       FilterStrategy* used = nullptr) {
  std::vector<bool> is_allowed(kBase, false);
  for (const auto label : allowed) is_allowed[label] = true;
  std::vector<std::vector<SearchResult>> results;
  for (std::size_t q = 0; q < data.queries.n; ++q) {
    SearchStats stats;
    auto r = index.search(data.queries.row(q), kK, {.ef = ef, .stats = &stats, .filter = &filter, .strategy = strategy});
    for (const auto& hit : r) EXPECT_TRUE(is_allowed[hit.label]) << "label " << hit.label << " is filtered out";
    if (used != nullptr) *used = stats.strategy;
    results.push_back(std::move(r));
  }
  return recall_at_k(results, filtered_truth(data, allowed, Metric::L2), kK);
}

std::vector<std::uint64_t> labels_where(const std::function<bool(std::uint64_t)>& keep) {
  std::vector<std::uint64_t> out;
  for (std::uint64_t i = 0; i < kBase; ++i) {
    if (keep(i)) out.push_back(i);
  }
  return out;
}

TEST_F(HnswTest, FilteredSearchHonorsTheFilterUnderEveryStrategy) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  // 50% and 10% of the labels, uncorrelated with position in the vector space.
  for (const std::uint64_t every : {2u, 10u}) {
    const auto allowed = labels_where([&](std::uint64_t l) { return (l * 7919) % every == 0; });
    const SearchFilter filter = index.make_filter([&](std::uint64_t l) { return (l * 7919) % every == 0; });
    ASSERT_EQ(filter.allowed(), allowed.size());
    EXPECT_NEAR(filter.selectivity(), 1.0 / every, 0.01);
    EXPECT_DOUBLE_EQ(filtered_recall(index, data_, filter, allowed, FilterStrategy::Exact, 100), 1.0);
    EXPECT_GE(filtered_recall(index, data_, filter, allowed, FilterStrategy::Inline, 100), 0.95) << "1/" << every;
    EXPECT_GE(filtered_recall(index, data_, filter, allowed, FilterStrategy::TwoHop, 100), 0.95) << "1/" << every;
    EXPECT_GE(filtered_recall(index, data_, filter, allowed, FilterStrategy::Auto, 100), 0.95) << "1/" << every;
  }
}

TEST_F(HnswTest, AutoPicksAStrategyBySelectivity) {
  HnswIndex index(params_for(Metric::L2));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  FilterStrategy used = FilterStrategy::Auto;
  // A handful of labels: scoring them outright is exact and cheapest.
  const auto few = labels_where([](std::uint64_t l) { return l % 250 == 3; });
  filtered_recall(index, data_, index.make_filter(few.data(), few.size()), few, FilterStrategy::Auto, 10, &used);
  EXPECT_EQ(used, FilterStrategy::Exact);
  // Most labels pass: walk the graph and skip the excluded ones.
  const auto most = labels_where([](std::uint64_t l) { return l % 5 != 0; });
  filtered_recall(index, data_, index.make_filter(most.data(), most.size()), most, FilterStrategy::Auto, 10, &used);
  EXPECT_EQ(used, FilterStrategy::Inline);
  // In between (and too many to scan at this ef): the two-hop walk.
  const auto some = labels_where([](std::uint64_t l) { return l % 5 == 0; });
  filtered_recall(index, data_, index.make_filter(some.data(), some.size()), some, FilterStrategy::Auto, 10, &used);
  EXPECT_EQ(used, FilterStrategy::TwoHop);
}

TEST_F(HnswTest, FilterEdgeCases) {
  HnswIndex index(params_for(Metric::L2, kBase + 10));
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  const float* q = data_.queries.row(0);

  // Nothing allowed: no results, under any strategy.
  const SearchFilter none = index.make_filter([](std::uint64_t) { return false; });
  EXPECT_EQ(none.allowed(), 0u);
  for (const auto strategy : {FilterStrategy::Auto, FilterStrategy::Exact, FilterStrategy::Inline, FilterStrategy::TwoHop}) {
    EXPECT_TRUE(index.search(q, kK, {.ef = 50, .filter = &none, .strategy = strategy}).empty());
  }

  // Fewer allowed than k: all of them come back, closest first.
  const std::vector<std::uint64_t> three = {11, 4000, 2500, 11, 999999};  // a duplicate and an unknown label
  const SearchFilter small = index.make_filter(three.data(), three.size());
  EXPECT_EQ(small.allowed(), 3u);
  const auto r = index.search(q, kK, {.ef = 50, .filter = &small});
  ASSERT_EQ(r.size(), 3u);
  EXPECT_LE(r[0].distance, r[1].distance);
  EXPECT_LE(r[1].distance, r[2].distance);

  // Removed labels are not allowed, even if the filter names them.
  index.remove(4000);
  const SearchFilter after = index.make_filter(three.data(), three.size());
  EXPECT_EQ(after.allowed(), 2u);
  for (const auto& hit : index.search(q, kK, {.ef = 50, .filter = &after})) EXPECT_NE(hit.label, 4000u);
  // ...and a filter made before the removal still never returns it.
  for (const auto strategy : {FilterStrategy::Exact, FilterStrategy::Inline, FilterStrategy::TwoHop}) {
    for (const auto& hit : index.search(q, kK, {.ef = 50, .filter = &small, .strategy = strategy})) {
      EXPECT_NE(hit.label, 4000u);
    }
  }
  // Nodes added after the filter was made are not allowed by it.
  index.add(kBase + 1, data_.queries.row(0));
  for (const auto& hit : index.search(q, kK, {.ef = 50, .filter = &small})) EXPECT_NE(hit.label, kBase + 1);
}

TEST_F(HnswTest, FilteredSearchWorksWithSq8) {
  HnswParams p = params_for(Metric::L2);
  p.quantization = Quantization::SQ8;
  HnswIndex index(p);
  index.train(data_.base.data.data(), kBase);
  for (std::size_t i = 0; i < kBase; ++i) index.add(i, data_.base.row(i));
  const auto allowed = labels_where([](std::uint64_t l) { return l % 4 == 1; });
  const SearchFilter filter = index.make_filter(allowed.data(), allowed.size());
  for (const auto strategy : {FilterStrategy::Exact, FilterStrategy::Inline, FilterStrategy::TwoHop}) {
    EXPECT_GE(filtered_recall(index, data_, filter, allowed, strategy, 100), 0.93) << to_string(strategy);
  }
}

TEST(Hnsw, FilterStrategyNames) {
  EXPECT_EQ(to_string(FilterStrategy::Auto), "auto");
  EXPECT_EQ(to_string(FilterStrategy::Exact), "exact");
  EXPECT_EQ(to_string(FilterStrategy::Inline), "inline");
  EXPECT_EQ(to_string(FilterStrategy::TwoHop), "two-hop");
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
