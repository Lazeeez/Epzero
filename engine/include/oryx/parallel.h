#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace oryx {

// Runs fn(i) for every i in [begin, end) on up to `threads` threads (0 = hardware concurrency),
// handing out indices dynamically so uneven work (e.g. HNSW inserts) stays balanced. The first
// exception thrown by any worker stops the remaining work and is rethrown to the caller.
template <typename Fn>
void parallel_for(std::size_t begin, std::size_t end, std::size_t threads, Fn fn) {
  if (begin >= end) return;
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  threads = std::min(threads, end - begin);
  if (threads == 1) {
    for (std::size_t i = begin; i < end; ++i) fn(i);
    return;
  }
  std::atomic<std::size_t> next{begin};
  std::exception_ptr error;
  std::mutex error_lock;
  auto worker = [&] {
    for (std::size_t i = next++; i < end; i = next++) {
      try {
        fn(i);
      } catch (...) {
        std::lock_guard lock(error_lock);
        if (!error) error = std::current_exception();
        next = end;
      }
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  for (std::size_t t = 1; t < threads; ++t) pool.emplace_back(worker);
  worker();  // the calling thread works too
  for (auto& t : pool) t.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace oryx
