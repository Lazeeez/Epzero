#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace epzero::detail {

// Visited set for one search: one bit per node, so it stays small enough to live in the L2 cache
// (1M nodes: 128 KiB). The search then pays no cache miss to ask "seen it?", where the byte-per-
// node version (1 MiB for 1M nodes, 10 MiB for 10M) missed L2 on a large share of its checks
// (measured: ~8% of GloVe-100 search time). Clearing touches only the words of the nodes this
// search marked, a few thousand, unless it marked so many that clearing everything is cheaper.
class VisitedList {
 public:
  explicit VisitedList(std::size_t capacity) : words_((capacity + 63) / 64, 0) {}

  void reset() {
    if (marked_.size() > words_.size() / 4) {
      std::fill(words_.begin(), words_.end(), 0);
    } else {
      for (const std::uint32_t id : marked_) words_[id >> 6] = 0;
    }
    marked_.clear();
  }

  // Marks `id` visited; returns true if it had already been visited in this search.
  bool test_and_set(std::uint32_t id) {
    std::uint64_t& word = words_[id >> 6];
    const std::uint64_t bit = std::uint64_t{1} << (id & 63);
    if ((word & bit) != 0) return true;
    word |= bit;
    marked_.push_back(id);
    return false;
  }

 private:
  std::vector<std::uint64_t> words_;
  std::vector<std::uint32_t> marked_;  // this search's marks, to clear them
};

// Recycles VisitedLists across queries so steady-state search does no O(n) allocation.
class VisitedPool {
 public:
  explicit VisitedPool(std::size_t capacity) : capacity_(capacity) {}

  class Handle {
   public:
    Handle(VisitedPool& pool, std::unique_ptr<VisitedList> list) : pool_(pool), list_(std::move(list)) {}
    ~Handle() { pool_.release(std::move(list_)); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    VisitedList* operator->() const noexcept { return list_.get(); }

   private:
    VisitedPool& pool_;
    std::unique_ptr<VisitedList> list_;
  };

  Handle acquire() {
    std::unique_ptr<VisitedList> list;
    {
      std::lock_guard lock(mutex_);
      if (!free_.empty()) {
        list = std::move(free_.back());
        free_.pop_back();
      }
    }
    if (!list) list = std::make_unique<VisitedList>(capacity_);
    list->reset();
    return Handle(*this, std::move(list));
  }

 private:
  void release(std::unique_ptr<VisitedList> list) {
    std::lock_guard lock(mutex_);
    free_.push_back(std::move(list));
  }

  std::size_t capacity_;
  std::mutex mutex_;
  std::vector<std::unique_ptr<VisitedList>> free_;
};

}  // namespace epzero::detail
