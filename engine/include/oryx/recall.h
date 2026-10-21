#pragma once

// Live recall: how close an approximate index's answers are to the exact ones, measured on the
// queries it actually serves, and kept at a target.
//
// HNSW trades recall for speed through the beam width (ef), and the right ef depends on the data,
// the queries and the filters, all of which change in production. RecallMonitor re-answers a small
// sample of real queries exactly, in the background, and estimates recall@k with a confidence
// interval per traffic segment. RecallGuard closes the loop: it searches each segment at the
// smallest beam width whose measured recall stays above a target.

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "oryx/hnsw.h"

namespace oryx {

struct RecallEstimate {
  std::size_t samples = 0;  // queries re-answered exactly in the current window
  double recall = 1.0;      // mean recall@k over them
  double lower = 0.0;       // confidence interval (Wilson score, one observation per query)
  double upper = 1.0;
};

class RecallMonitor {
 public:
  struct Options {
    double sample_rate = 0.01;  // share of observed queries re-answered exactly
    std::size_t window = 1000;  // the estimate covers this many most recent samples per segment
    std::size_t queue = 64;     // pending exact checks; beyond this, samples are dropped
    double z = 1.96;            // confidence level of the interval (1.96: 95%)
    // The reference each sample is compared with: 0 = an exact scan (the truth, at the cost of
    // scoring every vector), or a beam width for a far wider search than production uses. On SIFT1M,
    // ef 1024 is 31x cheaper than a scan with recall 0.9995: the estimate is off by at most the
    // reference's own misses, in either direction.
    std::size_t reference_ef = 0;
  };

  explicit RecallMonitor(const HnswIndex& index) : RecallMonitor(index, Options{}) {}
  RecallMonitor(const HnswIndex& index, Options options);
  ~RecallMonitor();
  RecallMonitor(const RecallMonitor&) = delete;
  RecallMonitor& operator=(const RecallMonitor&) = delete;

  // Records a served query and what it returned. A sample of them is re-answered exactly on the
  // monitor's own thread; the caller never waits (a full queue drops the sample instead).
  void observe(const float* query, std::size_t k, const std::vector<SearchResult>& results,
               const SearchFilter* filter = nullptr, const std::string& segment = "all");

  RecallEstimate estimate(const std::string& segment = "all") const;
  std::vector<std::pair<std::string, RecallEstimate>> segments() const;
  // Forgets a segment's samples, e.g. after its search settings changed.
  void reset(const std::string& segment);
  // Waits for every queued check to finish (tests and benchmarks).
  void drain();

  std::size_t checked() const noexcept { return checked_.load(std::memory_order_relaxed); }
  std::size_t dropped() const noexcept { return dropped_.load(std::memory_order_relaxed); }

  // Called on the monitor's thread after each exact check, with the segment it updated.
  void on_sample(std::function<void(const std::string& segment)> callback);

 private:
  struct Pending {
    std::vector<float> query;
    std::size_t k;
    std::vector<std::uint64_t> returned;
    std::optional<SearchFilter> filter;
    std::string segment;
  };
  void run();
  RecallEstimate estimate_locked(const std::deque<double>& window) const;

  const HnswIndex& index_;
  Options options_;
  std::uint64_t period_;
  std::atomic<std::uint64_t> seen_{0};
  std::atomic<std::size_t> checked_{0};
  std::atomic<std::size_t> dropped_{0};

  mutable std::mutex mutex_;
  std::condition_variable wake_;
  std::condition_variable idle_;
  std::deque<Pending> queue_;
  bool busy_ = false;
  bool stop_ = false;
  std::map<std::string, std::deque<double>> windows_;
  std::function<void(const std::string&)> callback_;
  std::thread worker_;
};

// Searches at the smallest beam width that keeps measured recall above a target, per segment:
// unfiltered queries, and filtered ones by how selective their filter is. The beam widens as soon
// as measured recall falls below the target, and narrows slowly while the confidence interval's
// lower end stays above it by a margin, so the index keeps up with drift (deletes, new data, new
// kinds of queries) without being over-provisioned forever.
class RecallGuard {
 public:
  struct Options {
    double target = 0.95;           // recall@k to hold
    std::size_t initial_ef = 64;
    std::size_t min_ef = 16;
    std::size_t max_ef = 2048;
    std::size_t min_samples = 200;  // samples since the last change before adjusting again
    double margin = 0.01;           // narrow only while the lower bound clears target + margin
    // Per-query effort for unfiltered searches: each query stops once its best k have settled
    // (SearchOptions::patience = ef / 3), and the beam width the guard steers becomes a cap. Easy
    // queries finish early and hard ones search further, so the same recall costs less and fewer
    // queries come back badly wrong (measured on GloVe-100: see bench/bench_adaptive.cpp).
    bool per_query = false;
    RecallMonitor::Options monitor{};
  };

  explicit RecallGuard(const HnswIndex& index) : RecallGuard(index, Options{}) {}
  RecallGuard(const HnswIndex& index, Options options);

  std::vector<SearchResult> search(const float* query, std::size_t k, const SearchFilter* filter = nullptr,
                                   SearchStats* stats = nullptr);

  struct Segment {
    std::string name;
    std::size_t ef;
    std::size_t changes;  // how many times its beam width moved
    RecallEstimate estimate;
  };
  std::vector<Segment> report() const;
  RecallMonitor& monitor() noexcept { return monitor_; }
  double target() const noexcept { return options_.target; }

  // The segment a query falls in: "unfiltered", or its filter's selectivity band.
  static std::string segment_of(const SearchFilter* filter);

 private:
  struct State {
    std::atomic<std::size_t> ef;
    std::atomic<std::size_t> changes{0};
    explicit State(std::size_t initial) : ef(initial) {}
  };
  State& state(const std::string& segment);
  void adjust(const std::string& segment);

  const HnswIndex& index_;
  Options options_;
  mutable std::mutex states_lock_;
  std::map<std::string, std::unique_ptr<State>> states_;
  // Last, so it is destroyed first: its thread calls adjust(), which uses the members above.
  RecallMonitor monitor_;
};

}  // namespace oryx
