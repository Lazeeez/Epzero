#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "oryx/brute_force.h"
#include "oryx/dataset.h"
#include "oryx/hnsw.h"
#include "oryx/recall.h"

namespace oryx {
namespace {

constexpr std::size_t kDim = 32;
constexpr std::size_t kBase = 5000;
constexpr std::size_t kQueries = 300;
constexpr std::size_t kK = 10;

class RecallTest : public ::testing::Test {
 protected:
  void SetUp() override {
    data_ = make_clustered(kBase, kQueries, kDim, 50, 3);
    index_ = std::make_unique<HnswIndex>(
        HnswParams{.dim = kDim, .metric = Metric::L2, .max_elements = kBase, .M = 8, .ef_construction = 64});
    for (std::size_t i = 0; i < kBase; ++i) index_->add(i, data_.base.row(i));
    truth_ = compute_ground_truth(data_.base.data.data(), kBase, kDim, data_.queries.data.data(), kQueries, kK,
                                  Metric::L2);
  }

  // True recall@k of the index at `ef`, over every query.
  double true_recall(std::size_t ef) const {
    std::vector<std::vector<SearchResult>> results;
    for (std::size_t q = 0; q < kQueries; ++q) results.push_back(index_->search(data_.queries.row(q), kK, {.ef = ef}));
    return recall_at_k(results, truth_, kK);
  }

  SyntheticData data_;
  std::unique_ptr<HnswIndex> index_;
  std::vector<std::vector<std::uint32_t>> truth_;
};

TEST_F(RecallTest, ExactSearchIsTheGroundTruth) {
  for (std::size_t q = 0; q < 20; ++q) {
    const auto exact = index_->exact_search(data_.queries.row(q), kK);
    ASSERT_EQ(exact.size(), kK);
    for (std::size_t i = 0; i < kK; ++i) EXPECT_EQ(exact[i].label, truth_[q][i]);
  }
  // With a filter: the exact answer among the allowed labels only.
  const SearchFilter odd = index_->make_filter([](std::uint64_t l) { return l % 2 == 1; });
  for (const auto& hit : index_->exact_search(data_.queries.row(0), kK, &odd)) EXPECT_EQ(hit.label % 2, 1u);
  // Deleted vectors are never part of the truth.
  const auto first = index_->exact_search(data_.queries.row(0), 1)[0].label;
  index_->remove(first);
  EXPECT_NE(index_->exact_search(data_.queries.row(0), 1)[0].label, first);
}

TEST_F(RecallTest, ExplainsEveryKindOfMiss) {
  using Cause = MissExplanation::Cause;
  const float* q = data_.queries.row(0);
  const auto top = index_->search(q, kK, {.ef = 64});
  EXPECT_EQ(index_->explain(q, kK, top[0].label, {.ef = 64}).cause, Cause::Returned);
  EXPECT_EQ(index_->explain(q, kK, 999'999, {.ef = 64}).cause, Cause::NotInIndex);

  const SearchFilter none_of_top = index_->make_filter([&](std::uint64_t l) { return l != top[0].label; });
  EXPECT_EQ(index_->explain(q, kK, top[0].label, {.ef = 64, .filter = &none_of_top}).cause, Cause::FilteredOut);

  // A far-away vector is simply not near enough: the search was right not to return it.
  const auto far = index_->exact_search(q, kBase).back().label;
  const auto far_why = index_->explain(q, kK, far, {.ef = 64});
  EXPECT_EQ(far_why.cause, Cause::NotNearEnough);
  EXPECT_EQ(far_why.exact_rank, kBase);
  EXPECT_GT(far_why.distance, far_why.kth_distance);

  // A true neighbour the narrowest search misses: found again with a wider beam.
  bool found_miss = false;
  for (std::size_t i = 0; i < kQueries && !found_miss; ++i) {
    const float* query = data_.queries.row(i);
    const auto got = index_->search(query, kK, {.ef = kK});
    for (const auto label : truth_[i]) {
      if (std::any_of(got.begin(), got.end(), [&](const SearchResult& r) { return r.label == label; })) continue;
      const auto why = index_->explain(query, kK, label, {.ef = kK});
      EXPECT_EQ(why.cause, Cause::SearchMissed);
      EXPECT_LE(why.exact_rank, kK);
      EXPECT_GT(why.ef_needed, kK);
      const auto wide = index_->search(query, kK, {.ef = why.ef_needed});
      EXPECT_TRUE(std::any_of(wide.begin(), wide.end(), [&](const SearchResult& r) { return r.label == label; }));
      found_miss = true;
      break;
    }
  }
  EXPECT_TRUE(found_miss) << "no query missed a neighbour at ef = k; the test needs a harder setting";
  EXPECT_EQ(to_string(Cause::SearchMissed), "search-missed");
}

TEST_F(RecallTest, MonitorEstimatesTheTrueRecall) {
  RecallMonitor monitor(*index_, {.sample_rate = 1.0, .window = 10'000, .queue = kQueries});
  constexpr std::size_t kEf = 12;  // low enough that recall is visibly below 1
  for (std::size_t q = 0; q < kQueries; ++q) {
    const float* query = data_.queries.row(q);
    monitor.observe(query, kK, index_->search(query, kK, {.ef = kEf}));
  }
  monitor.drain();
  const RecallEstimate e = monitor.estimate();
  const double truth = true_recall(kEf);
  EXPECT_EQ(e.samples, kQueries);
  EXPECT_NEAR(e.recall, truth, 1e-9);  // every query was checked: the estimate is exact
  EXPECT_LE(e.lower, truth);
  EXPECT_GE(e.upper, truth);
  EXPECT_LT(e.upper - e.lower, 0.1);
  EXPECT_EQ(monitor.checked(), kQueries);
}

TEST_F(RecallTest, MonitorSamplesAndNeverBlocks) {
  RecallMonitor monitor(*index_, {.sample_rate = 0.1, .window = 1000, .queue = 1});
  for (std::size_t i = 0; i < 2000; ++i) {
    const float* query = data_.queries.row(i % kQueries);
    monitor.observe(query, kK, index_->search(query, kK, {.ef = 32}));
  }
  monitor.drain();
  // 200 sampled; with room for one pending check, some were dropped rather than waited for.
  EXPECT_EQ(monitor.checked() + monitor.dropped(), 200u);
  EXPECT_GT(monitor.checked(), 0u);
}

TEST_F(RecallTest, GuardRaisesTheBeamToMeetItsTarget) {
  constexpr double kTarget = 0.97;
  RecallGuard guard(*index_, {.target = kTarget, .initial_ef = kK, .min_ef = kK, .max_ef = 512, .min_samples = 100,
                              .margin = 0.01, .monitor = {.sample_rate = 1.0, .window = 400, .queue = 1000}});
  for (int pass = 0; pass < 40; ++pass) {
    for (std::size_t q = 0; q < kQueries; ++q) (void)guard.search(data_.queries.row(q), kK);
    guard.monitor().drain();
  }
  const auto report = guard.report();
  ASSERT_EQ(report.size(), 1u);
  EXPECT_EQ(report[0].name, "unfiltered");
  EXPECT_GT(report[0].changes, 0u);
  EXPECT_GT(report[0].ef, kK);
  // Where it settled actually meets the target (within sampling error).
  EXPECT_GE(true_recall(report[0].ef), kTarget - 0.01);
}

TEST(RecallGuardSegments, BandsBySelectivity) {
  EXPECT_EQ(RecallGuard::segment_of(nullptr), "unfiltered");
}

}  // namespace
}  // namespace oryx
