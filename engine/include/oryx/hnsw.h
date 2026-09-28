#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "oryx/distance.h"

namespace oryx {

namespace detail {
class VisitedPool;
}

struct HnswParams {
  std::size_t dim = 0;
  Metric metric = Metric::L2;
  std::size_t max_elements = 0;
  std::size_t M = 16;                  // max links per node on layers >= 1 (layer 0 allows 2*M)
  std::size_t ef_construction = 200;   // beam width while building
  std::uint64_t seed = 42;             // makes level assignment reproducible
};

struct SearchResult {
  std::uint64_t label;
  float distance;
};

struct SearchStats {
  std::size_t distance_computations = 0;
  std::size_t nodes_expanded = 0;
};

// One step of a search: while expanding `from` on `layer`, the neighbor `to` was scored.
// `accepted` means it entered the candidate beam. Labels are external ids.
struct TraceEvent {
  std::uint32_t layer;
  std::uint64_t from;
  std::uint64_t to;
  float distance;
  bool accepted;
};

// Full record of a query's path through the graph, used by the visualizer.
struct SearchTrace {
  std::uint64_t entry_label = 0;
  std::uint32_t entry_layer = 0;
  std::vector<TraceEvent> events;
};

struct SearchOptions {
  std::size_t ef = 64;              // beam width on layer 0; clamped to at least k
  SearchStats* stats = nullptr;     // optional
  SearchTrace* trace = nullptr;     // optional; costs nothing when null
};

// Hierarchical Navigable Small World index (Malkov & Yashunin, 2016).
//
// Thread safety: add() and search() may be called concurrently from any number of threads.
// save() must not run concurrently with add().
//
// Concurrency: search is lock-free. Neighbor-list slots are atomics written with release
// stores and read with acquire loads, so a reader that sees a node id also sees that node's
// data. Writers are serialized per node by a mutex that guards only that node's lists, and a
// thread never holds more than one node lock at a time, which rules out lock-order deadlocks.
// The entry point (node id + top level) is packed into one atomic word so readers always see
// a consistent pair.
class HnswIndex {
 public:
  explicit HnswIndex(const HnswParams& params);
  ~HnswIndex();

  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;

  // Inserts a vector of params().dim floats. Throws std::length_error when the index is full
  // and std::invalid_argument when `label` already exists.
  void add(std::uint64_t label, const float* vector);

  // Returns up to k nearest neighbors, closest first.
  std::vector<SearchResult> search(const float* query, std::size_t k,
                                   const SearchOptions& options = {}) const;

  void save(const std::string& path) const;
  static std::unique_ptr<HnswIndex> load(const std::string& path);

  // Introspection, used by tests, benchmarks and the topology export for the visualizer.
  std::size_t size() const noexcept { return count_.load(std::memory_order_acquire); }
  const HnswParams& params() const noexcept { return params_; }
  SimdLevel simd_level() const noexcept { return simd_level_; }
  int max_level() const noexcept;
  int level_of(std::uint64_t label) const;
  std::vector<std::uint64_t> neighbors_of(std::uint64_t label, int layer) const;
  std::size_t memory_bytes() const noexcept;

 private:
  using id_t = std::uint32_t;
  struct Candidate {
    float distance;
    id_t id;
  };
  struct CloserFirst {
    bool operator()(const Candidate& a, const Candidate& b) const noexcept { return a.distance > b.distance; }
  };
  struct FartherFirst {
    bool operator()(const Candidate& a, const Candidate& b) const noexcept { return a.distance < b.distance; }
  };

  // Entry point packing: high 32 bits = top level + 1 (0 means empty), low 32 bits = node id.
  static constexpr std::uint64_t kEmpty = 0;
  static std::uint64_t pack_entry(id_t id, int level) noexcept {
    return (static_cast<std::uint64_t>(level + 1) << 32) | id;
  }

  const float* vector_of(id_t id) const noexcept {
    return reinterpret_cast<const float*>(level0_.get() + id * node_stride_);
  }
  float* vector_of(id_t id) noexcept { return reinterpret_cast<float*>(level0_.get() + id * node_stride_); }
  // Layout of a neighbor list: [count, id_0, id_1, ..., id_{capacity-1}].
  id_t* links_of(id_t id, int layer) noexcept;
  const id_t* links_of(id_t id, int layer) const noexcept;
  std::size_t max_links(int layer) const noexcept { return layer == 0 ? max_links0_ : params_.M; }

  float distance(const float* a, const float* b) const noexcept { return dist_(a, b, params_.dim); }
  int random_level();
  void copy_links(id_t id, int layer, std::vector<id_t>& out) const;
  // Caller must hold node_locks_[id].
  void write_links(id_t id, int layer, const std::vector<Candidate>& neighbors) noexcept;

  id_t greedy_descend(const float* query, id_t entry, int from_layer, int to_layer, SearchStats& stats,
                      SearchTrace* trace) const;
  std::vector<Candidate> search_layer(const float* query, id_t entry, std::size_t ef, int layer,
                                      SearchStats& stats, SearchTrace* trace) const;
  // The neighbor-selection heuristic (Algorithm 4 in the paper): keep a candidate only if it is
  // closer to the base point than to every neighbor already kept. This keeps links pointing in
  // diverse directions, which is what makes the graph navigable across clusters.
  std::vector<Candidate> select_neighbors(std::vector<Candidate> candidates, std::size_t max_count) const;
  void connect(id_t id, const std::vector<Candidate>& neighbors, int layer);

  HnswParams params_;
  SimdLevel simd_level_;
  DistanceFn dist_;  std::size_t max_links0_;
  double level_mult_;

  // Layer 0 is one contiguous, 64-byte-aligned block. Each node's record holds its vector
  // followed by its layer-0 neighbor list, so expanding a node touches adjacent cache lines.
  std::size_t links0_offset_;
  std::size_t node_stride_;  struct AlignedFree {
    void operator()(std::byte* p) const noexcept;
  };
  std::unique_ptr<std::byte[], AlignedFree> allocate_level0() const;
  std::unique_ptr<std::byte[], AlignedFree> level0_;

  // Upper layers are sparse (~1/M of nodes reach layer 1), so they are stored per node.
  std::vector<std::unique_ptr<id_t[]>> upper_links_;
  std::vector<std::int8_t> levels_;
  std::vector<std::uint64_t> labels_;

  std::unique_ptr<std::mutex[]> node_locks_;
  std::mutex entry_lock_;
  std::atomic<std::uint64_t> entry_{kEmpty};
  std::atomic<std::size_t> count_{0};

  mutable std::mutex label_lock_;
  std::unordered_map<std::uint64_t, id_t> label_to_id_;

  std::mutex rng_lock_;
  std::mt19937_64 rng_;

  std::unique_ptr<detail::VisitedPool> visited_pool_;
};

}  // namespace oryx
