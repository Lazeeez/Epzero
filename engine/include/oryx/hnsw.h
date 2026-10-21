#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "oryx/distance.h"

namespace oryx {

namespace detail {
class VisitedPool;
}

enum class Quantization {
  None,  // full-precision float vectors in the graph
  SQ8,   // 8-bit scalar quantization: uint8 codes with per-dimension affine scales (call train() first)
  RaBitQ,  // 1-bit RaBitQ codes with an unbiased distance estimator (call train() first; keeps the
           // full vectors to build the graph and re-rank, see src/rabitq.h)
};

struct HnswParams {
  std::size_t dim = 0;
  Metric metric = Metric::L2;
  std::size_t max_elements = 0;
  std::size_t M = 16;                  // max links per node on layers >= 1 (layer 0 allows 2*M)
  std::size_t ef_construction = 200;   // beam width while building
  std::uint64_t seed = 42;             // makes level assignment reproducible
  Quantization quantization = Quantization::None;
  // SQ8: also keep full-precision vectors, used to build a better graph and to re-rank the final
  // candidates exactly. Without them the index stores ~4x less vector data. RaBitQ requires them.
  bool keep_vectors = true;
};

struct SearchResult {
  std::uint64_t label;
  float distance;
};

// How a filtered search finds its results. Auto picks one per query from the filter's size.
enum class FilterStrategy : std::uint8_t {
  Auto,
  Exact,   // score only the allowed vectors, one by one: exact, and cheapest when few pass
  Inline,  // the normal graph walk, keeping only allowed nodes as results: best when most pass
  TwoHop,  // walk allowed nodes only; past a filtered-out neighbor, continue to its neighbors
           // (the idea behind ACORN), so the walk stays connected when many nodes are excluded
};
std::string_view to_string(FilterStrategy strategy) noexcept;

struct SearchStats {
  std::size_t distance_computations = 0;
  // Of those, how many read a full-precision vector (RaBitQ estimates the rest from its codes).
  std::size_t full_distance_computations = 0;
  std::size_t nodes_expanded = 0;
  FilterStrategy strategy = FilterStrategy::Auto;  // what a filtered search used
};

// Which labels a search may return. Compiled against one index by HnswIndex::make_filter(), so
// any number of queries can reuse it: a bitset over the index's node slots, plus the list of
// allowed slots for exact scans. Nodes added after it was made are not allowed.
class SearchFilter {
 public:
  std::size_t allowed() const noexcept { return ids_.size(); }
  // Allowed share of the live nodes when the filter was made (1.0 when nothing is filtered out).
  double selectivity() const noexcept {
    return universe_ == 0 ? 0.0 : static_cast<double>(ids_.size()) / static_cast<double>(universe_);
  }

 private:
  friend class HnswIndex;
  bool allows(std::uint32_t id) const noexcept {
    return (id >> 6) < bits_.size() && ((bits_[id >> 6] >> (id & 63)) & 1u) != 0;
  }
  std::vector<std::uint64_t> bits_;
  std::vector<std::uint32_t> ids_;
  std::size_t universe_ = 0;
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
  // Per-query effort: stop once the best k haven't changed for this many node expansions in a
  // row (0 = off). An easy query settles quickly and stops early; a hard one keeps improving and
  // searches on, up to `ef`, which then only caps the beam. Unfiltered searches only.
  std::size_t patience = 0;
  // SQ8 with kept vectors: how many of the best approximate candidates to re-score exactly
  // (0 = 3 x k: on SIFT1M, GloVe-100 and GIST1M, re-scoring all ef of them gave the same recall to
  // within 0.0002, at up to ef full-vector reads per query). Ignored otherwise.
  std::size_t rerank = 0;
  SearchStats* stats = nullptr;     // optional
  SearchTrace* trace = nullptr;     // optional; costs nothing when null
  // Optional: only labels the filter allows are returned. Must come from this index's make_filter().
  const SearchFilter* filter = nullptr;
  FilterStrategy strategy = FilterStrategy::Auto;  // with a filter: force a strategy (benchmarks)
};

// Why a label is, or isn't, in a search's results: the answer to "why wasn't this retrieved?".
struct MissExplanation {
  enum class Cause : std::uint8_t {
    Returned,       // it is in the results
    NotInIndex,     // never added, or deleted
    FilteredOut,    // the search's filter excludes it
    NotNearEnough,  // the search was right: by exact distance it ranks below the top k
    SearchMissed,   // it is among the exact top k, but the approximate search didn't return it
  };
  Cause cause = Cause::NotInIndex;
  std::size_t exact_rank = 0;  // 1-based rank by exact distance among the live (allowed) vectors
  float distance = 0.0f;       // its exact distance to the query
  float kth_distance = 0.0f;   // distance of the k-th result the search returned
  bool scored = false;         // SearchMissed: the walk scored it but kept closer ones (else never reached it)
  std::size_t ef_needed = 0;   // SearchMissed: the smallest beam width (doubling, up to 4096) that returns it
};
std::string_view to_string(MissExplanation::Cause cause) noexcept;

// Hierarchical Navigable Small World index (Malkov & Yashunin, 2016).
//
// Thread safety: add(), remove() and search() may be called concurrently from any number of
// threads. train(), save() and compacted() must not run concurrently with add() or remove().
//
// Concurrency: search is lock-free. Neighbor-list slots are atomics written with release
// stores and read with acquire loads, so a reader that sees a node id also sees that node's
// data. Writers are serialized per node by a mutex that guards only that node's lists, and a
// thread never holds more than one node lock at a time, which rules out lock-order deadlocks.
// The entry point (node id + top level) is packed into one atomic word so readers always see
// a consistent pair.
//
// Deletes are tombstones: a removed node keeps routing searches through the graph but is never
// returned or linked to again. compacted() rebuilds the index without them.
class HnswIndex {
 public:
  explicit HnswIndex(const HnswParams& params);
  ~HnswIndex();

  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;

  // SQ8 only: learns per-dimension ranges from a representative sample. Must be called before
  // the first add(); values outside the sampled range are clamped.
  void train(const float* data, std::size_t n);
  bool trained() const noexcept {
    return params_.quantization == Quantization::None || !sq8_scale_.empty() || !rbq_center_.empty();
  }

  // Inserts a vector of params().dim floats. Throws std::length_error when the index is full
  // and std::invalid_argument when `label` is already present.
  void add(std::uint64_t label, const float* vector);

  // Tombstones `label`; returns false if it is not present. The label may be added again.
  bool remove(std::uint64_t label);

  // Returns up to k nearest live neighbors, closest first.
  std::vector<SearchResult> search(const float* query, std::size_t k,
                                   const SearchOptions& options = {}) const;

  // The exact top-k: scores every live vector (only the allowed ones, with a filter). It is the
  // ground truth that recall is measured against, at the cost of a full scan. Uses the
  // full-precision vectors; an SQ8 index without them scores its codes instead.
  std::vector<SearchResult> exact_search(const float* query, std::size_t k, const SearchFilter* filter = nullptr) const;

  // Explains why `label` is or isn't among search(query, k, options)'s results: excluded, not
  // near enough, or missed by the approximate walk (and then whether it was reached, and what beam
  // width would have found it). Costs a full scan plus a few searches.
  MissExplanation explain(const float* query, std::size_t k, std::uint64_t label, const SearchOptions& options = {}) const;

  // Compiles a filter for search(): `allow` is asked once for each live node's label.
  SearchFilter make_filter(const std::function<bool(std::uint64_t label)>& allow) const;
  // The same, from a list of allowed labels; labels not in the index are ignored.
  SearchFilter make_filter(const std::uint64_t* labels, std::size_t count) const;

  // A fresh index holding only the live vectors (same parameters), for reclaiming the space of
  // deleted nodes. Requires full-precision vectors (always true unless SQ8 without keep_vectors,
  // where the decoded approximations are used instead).
  std::unique_ptr<HnswIndex> compacted(std::size_t threads = 0) const;

  void save(const std::string& path) const;
  // Loads a snapshot; corrupt or truncated files throw. With max_bytes, a snapshot whose index
  // would allocate more than that is refused before allocating (for untrusted files).
  static std::unique_ptr<HnswIndex> load(const std::string& path, std::size_t max_bytes = 0);
  // Memory an index with these parameters allocates up front (the layer-0 block, plus full
  // vectors for SQ8 with keep_vectors), before any upper-layer lists.
  static std::size_t estimated_bytes(const HnswParams& params) noexcept;

  // The graph without its vectors: every node's label, level and links. For shipping a prebuilt
  // index next to vectors that are stored elsewhere (e.g. the web demo's int8 file): a few bytes
  // per link instead of 4 * dim per node. Links are 16-bit when the index has < 65,536 nodes.
  // Only for full-precision indexes without deletions.
  std::vector<std::uint8_t> export_graph() const;
  // Rebuilds an index from export_graph() bytes and the vectors, where row `label` of `vectors`
  // (n x params.dim) is that label's vector. Validates every link, so corrupt bytes throw.
  static std::unique_ptr<HnswIndex> import_graph(const HnswParams& params, const float* vectors, std::size_t n,
                                                 const std::uint8_t* bytes, std::size_t size);

  // Introspection, used by tests, benchmarks and the topology export for the visualizer.
  std::size_t size() const noexcept { return count_.load(std::memory_order_acquire); }  // slots used
  std::size_t deleted_count() const noexcept { return deleted_count_.load(std::memory_order_acquire); }
  std::size_t live_count() const noexcept { return size() - deleted_count(); }
  bool contains(std::uint64_t label) const;
  const HnswParams& params() const noexcept { return params_; }
  SimdLevel simd_level() const noexcept { return simd_level_; }
  int max_level() const noexcept;
  int level_of(std::uint64_t label) const;
  std::vector<std::uint64_t> neighbors_of(std::uint64_t label, int layer) const;
  std::size_t memory_bytes() const noexcept;

  // Memory prefetching in the search loop: keep `depth` of a node's unvisited neighbors prefetched
  // ahead of the one being scored, `lines` cache lines of each. The default (every neighbor, one
  // line) was the best measured on SIFT1M; tune_prefetch() measures the best for this machine and
  // data. Not thread-safe against concurrent searches; set it before serving.
  void set_prefetch(std::size_t depth, std::size_t lines) noexcept;
  std::pair<std::size_t, std::size_t> prefetch_settings() const noexcept { return {prefetch_depth_, prefetch_lines_}; }
  // Times searches of `queries` (n rows; a few hundred, e.g. sampled from the data) at a grid of
  // prefetch settings and keeps the fastest, if it beats the default by at least 3% (smaller
  // differences are noise). Machine-specific, so not saved with the index: tune after loading.
  // Measured: +17% on GloVe-100 at full precision, nothing on GIST1M with SQ8. Returns the setting.
  std::pair<std::size_t, std::size_t> tune_prefetch(const float* queries, std::size_t n, std::size_t k = 10,
                                                    std::size_t ef = 64);

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
  class Scorer;

  // Entry point packing: high 32 bits = top level + 1 (0 means empty), low 32 bits = node id.
  static constexpr std::uint64_t kEmpty = 0;
  static std::uint64_t pack_entry(id_t id, int level) noexcept {
    return (static_cast<std::uint64_t>(level + 1) << 32) | id;
  }

  bool sq8() const noexcept { return params_.quantization == Quantization::SQ8; }
  bool rabitq() const noexcept { return params_.quantization == Quantization::RaBitQ; }
  bool quantized() const noexcept { return params_.quantization != Quantization::None; }
  std::size_t rabitq_record_bytes() const noexcept;
  std::byte* record(id_t id) const noexcept { return level0_.get() + id * node_stride_; }
  // Full-precision vector: inside the layer-0 record (None) or in full_ (quantized + keep_vectors).
  const float* full_vector(id_t id) const noexcept;
  bool has_full_vectors() const noexcept { return !quantized() || params_.keep_vectors; }
  const std::uint8_t* codes_of(id_t id) const noexcept {
    return reinterpret_cast<const std::uint8_t*>(record(id) + sizeof(float));
  }
  float code_norm(id_t id) const noexcept { return *reinterpret_cast<const float*>(record(id)); }
  void encode(const float* v, std::uint8_t* codes, float* norm) const noexcept;
  void decode(id_t id, float* out) const noexcept;
  bool is_deleted(id_t id) const noexcept;

  // Layout of a neighbor list: [count, id_0, id_1, ..., id_{capacity-1}].
  id_t* links_of(id_t id, int layer) noexcept;
  const id_t* links_of(id_t id, int layer) const noexcept;
  std::size_t max_links(int layer) const noexcept { return layer == 0 ? max_links0_ : params_.M; }

  float pair_distance(id_t a, id_t b) const noexcept;
  int random_level();
  void copy_links(id_t id, int layer, std::vector<id_t>& out) const;
  // Caller must hold node_locks_[id].
  void write_links(id_t id, int layer, const std::vector<Candidate>& neighbors) noexcept;

  id_t greedy_descend(const Scorer& scorer, id_t entry, int from_layer, int to_layer, SearchStats& stats,
                      SearchTrace* trace) const;
  // Beam search on one layer. Tombstoned nodes are routed through but not returned, unless
  // `keep_deleted` (an insert that found no live neighbour links through them instead).
  // With `patience`, it also stops once the best `k` have gone that many expansions unchanged.
  std::vector<Candidate> search_layer(const Scorer& scorer, id_t entry, std::size_t ef, int layer,
                                      SearchStats& stats, SearchTrace* trace, bool keep_deleted = false,
                                      std::size_t k = 0, std::size_t patience = 0) const;
  // Layer-0 beam search that only returns nodes the filter allows (Inline or TwoHop).
  std::vector<Candidate> search_layer_filtered(const Scorer& scorer, id_t entry, std::size_t ef,
                                               const SearchFilter& filter, bool two_hop, SearchStats& stats,
                                               SearchTrace* trace) const;
  // Exact: scores every allowed node and keeps the best `keep`.
  std::vector<Candidate> scan_allowed(const Scorer& scorer, const SearchFilter& filter, std::size_t keep,
                                      SearchStats& stats) const;
  FilterStrategy choose_strategy(const SearchFilter& filter, std::size_t ef) const noexcept;
  // Loaders: every link names one of the first `count` nodes, on a layer that node is on.
  bool links_are_valid(std::size_t count) const noexcept;
  SearchFilter filter_from_ids(std::vector<id_t> ids) const;
  // The neighbor-selection heuristic (Algorithm 4 in the paper): keep a candidate only if it is
  // closer to the base point than to every neighbor already kept. This keeps links pointing in
  // diverse directions, which is what makes the graph navigable across clusters.
  std::vector<Candidate> select_neighbors(std::vector<Candidate> candidates, std::size_t max_count) const;
  void connect(id_t id, const std::vector<Candidate>& neighbors, int layer);

  HnswParams params_;
  SimdLevel simd_level_;
  DistanceFn dist_;
  Sq8DotFn sq8_dot_;
  // SQ8 in integers (int8 per-query weights), where the CPU has a kernel faster than the float one.
  Sq8IntDotFn sq8_int_dot_;
  RabitqIpFn rabitq_ip_;
  std::size_t max_links0_;
  double level_mult_;

  // Layer 0 is one contiguous block (2 MiB huge pages when large). Each node's record holds its
  // vector (floats, or SQ8 norm + codes) followed by its layer-0 neighbor list, so expanding a
  // node touches adjacent cache lines.
  std::size_t links0_offset_;
  std::size_t node_stride_;
  struct AlignedFree {
    void operator()(void* p) const noexcept;
  };
  std::unique_ptr<std::byte[], AlignedFree> allocate_level0() const;
  std::unique_ptr<std::byte[], AlignedFree> level0_;

  // SQ8: per-dimension x = min + code * scale, and the kept full-precision vectors.
  std::vector<float> sq8_min_;
  std::vector<float> sq8_scale_;
  std::unique_ptr<float[], AlignedFree> full_;

  // RaBitQ: the centroid, the rotation's random signs, and the padded (power of two) dimension.
  std::vector<float> rbq_center_;
  std::vector<std::int8_t> rbq_signs_;
  std::size_t rbq_padded_ = 0;

  // Upper layers are sparse (~1/M of nodes reach layer 1), so they are stored per node.
  std::vector<std::unique_ptr<id_t[]>> upper_links_;
  std::vector<std::int8_t> levels_;
  std::vector<std::uint64_t> labels_;
  std::unique_ptr<std::uint8_t[]> deleted_;  // tombstones, accessed atomically
  std::atomic<std::size_t> deleted_count_{0};

  std::unique_ptr<std::mutex[]> node_locks_;
  std::mutex entry_lock_;
  std::atomic<std::uint64_t> entry_{kEmpty};
  std::atomic<std::size_t> count_{0};

  mutable std::mutex label_lock_;
  std::unordered_map<std::uint64_t, id_t> label_to_id_;

  std::mutex rng_lock_;
  std::mt19937_64 rng_;

  std::unique_ptr<detail::VisitedPool> visited_pool_;

  std::size_t prefetch_depth_ = std::numeric_limits<std::size_t>::max();  // everything up front
  std::size_t prefetch_lines_ = 1;
};

}  // namespace oryx
