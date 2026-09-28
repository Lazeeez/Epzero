#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace oryx::detail {

// Visited set for one search. Instead of clearing n entries per query, each query bumps an
// epoch and a node counts as visited when its tag equals the current epoch. The array is
// only cleared when the epoch wraps.
//
// Tags are 8-bit on purpose: for 1M nodes the array is 1 MiB and stays in L2, whereas 16-bit
// tags (2 MiB) spill to L3 on every check. The extra full clear every 255 queries costs about
// 0.1 us per query amortized.
class VisitedList {
 public:
  using Tag = std::uint8_t;

  explicit VisitedList(std::size_t capacity) : tags_(capacity, 0) {}

  void reset() {
    if (++epoch_ == 0) {
      std::fill(tags_.begin(), tags_.end(), 0);
      epoch_ = 1;
    }
  }

  // Marks `id` visited; returns true if it had already been visited in this epoch.
  bool test_and_set(std::uint32_t id) noexcept {
    if (tags_[id] == epoch_) return true;
    tags_[id] = epoch_;
    return false;
  }

 private:
  std::vector<Tag> tags_;
  Tag epoch_ = 0;
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

}  // namespace oryx::detail
