#include "oryx/recall.h"

#include <algorithm>
#include <cmath>

namespace oryx {

RecallMonitor::RecallMonitor(const HnswIndex& index, Options options)
    : index_(index),
      options_(options),
      // Systematic sampling (every n-th query) is lock-free and as unbiased as random sampling
      // for traffic that doesn't repeat with that period.
      period_(options.sample_rate <= 0.0 ? 0 : std::max<std::uint64_t>(1, std::llround(1.0 / options.sample_rate))),
      worker_([this] { run(); }) {}

RecallMonitor::~RecallMonitor() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
  }
  wake_.notify_all();
  worker_.join();
}

void RecallMonitor::observe(const float* query, std::size_t k, const std::vector<SearchResult>& results,
                            const SearchFilter* filter, const std::string& segment) {
  if (period_ == 0 || k == 0 || seen_.fetch_add(1, std::memory_order_relaxed) % period_ != 0) return;
  Pending pending{std::vector<float>(query, query + index_.params().dim), k, {}, std::nullopt, segment};
  pending.returned.reserve(results.size());
  for (const auto& r : results) pending.returned.push_back(r.label);
  if (filter != nullptr) pending.filter = *filter;  // the caller's filter may not outlive the check
  {
    std::lock_guard lock(mutex_);
    if (queue_.size() >= options_.queue) {
      dropped_.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    queue_.push_back(std::move(pending));
  }
  wake_.notify_one();
}

void RecallMonitor::run() {
  for (;;) {
    Pending job;
    {
      std::unique_lock lock(mutex_);
      wake_.wait(lock, [&] { return stop_ || !queue_.empty(); });
      if (stop_) return;
      job = std::move(queue_.front());
      queue_.pop_front();
      busy_ = true;
    }
    const SearchFilter* filter = job.filter ? &*job.filter : nullptr;
    const auto exact = options_.reference_ef == 0
                           ? index_.exact_search(job.query.data(), job.k, filter)
                           : index_.search(job.query.data(), job.k, {.ef = options_.reference_ef, .filter = filter});
    std::size_t hits = 0;
    for (const auto& truth : exact) {
      if (std::find(job.returned.begin(), job.returned.end(), truth.label) != job.returned.end()) ++hits;
    }
    // Fewer than k exact answers (a small filter): recall is over the ones that exist.
    const double recall = exact.empty() ? 1.0 : static_cast<double>(hits) / static_cast<double>(exact.size());
    std::function<void(const std::string&)> callback;
    {
      std::lock_guard lock(mutex_);
      auto& window = windows_[job.segment];
      window.push_back(recall);
      if (window.size() > options_.window) window.pop_front();
      busy_ = false;
      callback = callback_;
    }
    checked_.fetch_add(1, std::memory_order_relaxed);
    if (callback) callback(job.segment);
    idle_.notify_all();
  }
}

RecallEstimate RecallMonitor::estimate_locked(const std::deque<double>& window) const {
  RecallEstimate out;
  out.samples = window.size();
  if (window.empty()) return out;
  double sum = 0.0;
  for (const double r : window) sum += r;
  const double n = static_cast<double>(window.size());
  const double p = sum / n;
  // Wilson score interval: sound near 0 and 1, where recall estimates live (a normal interval
  // would collapse to zero width when every sample is perfect).
  const double z2 = options_.z * options_.z;
  const double center = (p + z2 / (2 * n)) / (1 + z2 / n);
  const double half = options_.z * std::sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n);
  out.recall = p;
  out.lower = std::max(0.0, center - half);
  out.upper = std::min(1.0, center + half);
  return out;
}

RecallEstimate RecallMonitor::estimate(const std::string& segment) const {
  std::lock_guard lock(mutex_);
  const auto it = windows_.find(segment);
  return it == windows_.end() ? RecallEstimate{} : estimate_locked(it->second);
}

std::vector<std::pair<std::string, RecallEstimate>> RecallMonitor::segments() const {
  std::lock_guard lock(mutex_);
  std::vector<std::pair<std::string, RecallEstimate>> out;
  for (const auto& [name, window] : windows_) out.emplace_back(name, estimate_locked(window));
  return out;
}

void RecallMonitor::reset(const std::string& segment) {
  std::lock_guard lock(mutex_);
  windows_[segment].clear();
}

void RecallMonitor::drain() {
  std::unique_lock lock(mutex_);
  idle_.wait(lock, [&] { return queue_.empty() && !busy_; });
}

void RecallMonitor::on_sample(std::function<void(const std::string& segment)> callback) {
  std::lock_guard lock(mutex_);
  callback_ = std::move(callback);
}

// ---------------------------------------------------------------------------------------------

RecallGuard::RecallGuard(const HnswIndex& index, Options options)
    : index_(index), options_(options), monitor_(index, options.monitor) {
  monitor_.on_sample([this](const std::string& segment) { adjust(segment); });
}

std::string RecallGuard::segment_of(const SearchFilter* filter) {
  if (filter == nullptr) return "unfiltered";
  // Bands by decade of selectivity: within a band, recall at a given ef behaves alike.
  const double s = filter->selectivity();
  if (s >= 0.5) return "filtered: over 50%";
  if (s >= 0.1) return "filtered: 10-50%";
  if (s >= 0.01) return "filtered: 1-10%";
  return "filtered: under 1%";
}

RecallGuard::State& RecallGuard::state(const std::string& segment) {
  std::lock_guard lock(states_lock_);
  auto& slot = states_[segment];
  if (!slot) slot = std::make_unique<State>(options_.initial_ef);
  return *slot;
}

std::vector<SearchResult> RecallGuard::search(const float* query, std::size_t k, const SearchFilter* filter,
                                              SearchStats* stats) {
  const std::string segment = segment_of(filter);
  const std::size_t ef = state(segment).ef.load(std::memory_order_relaxed);
  auto results = index_.search(query, k, {.ef = ef, .stats = stats, .filter = filter});
  monitor_.observe(query, k, results, filter, segment);
  return results;
}

void RecallGuard::adjust(const std::string& segment) {
  const RecallEstimate e = monitor_.estimate(segment);
  if (e.samples < options_.min_samples) return;
  State& s = state(segment);
  const std::size_t ef = s.ef.load(std::memory_order_relaxed);
  std::size_t next = ef;
  // Widen as soon as measured recall falls short; narrow only when even the interval's lower end
  // clears the target by the margin; otherwise hold. Widening on the lower bound alone would react
  // to uncertainty rather than to recall (a fresh window's interval is wide) and over-provision.
  if (e.recall < options_.target) {
    next = std::min(options_.max_ef, std::max(ef + 1, ef * 3 / 2));  // short of the target: widen fast
  } else if (e.lower >= options_.target + options_.margin) {
    next = std::max(options_.min_ef, ef * 7 / 8);  // comfortably above: narrow gently
  }
  if (next == ef) return;
  s.ef.store(next, std::memory_order_relaxed);
  s.changes.fetch_add(1, std::memory_order_relaxed);
  monitor_.reset(segment);  // the old samples describe the old beam width
}

std::vector<RecallGuard::Segment> RecallGuard::report() const {
  std::vector<Segment> out;
  std::lock_guard lock(states_lock_);
  for (const auto& [name, s] : states_) {
    out.push_back({name, s->ef.load(std::memory_order_relaxed), s->changes.load(std::memory_order_relaxed),
                   monitor_.estimate(name)});
  }
  return out;
}

}  // namespace oryx
