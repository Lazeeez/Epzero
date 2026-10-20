#include "oryx/hnsw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string_view>

#if defined(__linux__) && !defined(__EMSCRIPTEN__)
#include <sys/mman.h>
#endif

#include "oryx/parallel.h"
#include "visited_pool.h"

namespace oryx {
namespace {

constexpr char kMagic[8] = {'O', 'R', 'Y', 'X', 'H', 'N', 'S', 'W'};
constexpr std::uint32_t kFormatVersion = 2;  // v2: quantization + tombstones; v1 files still load
constexpr int kMaxLevel = 31;

std::size_t round_up(std::size_t value, std::size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

inline void prefetch(const void* p) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p, 0, 3);
#else
  (void)p;
#endif
}

// Neighbor lists are read without any lock while writers (serialized per node by node_locks_)
// update them, so every slot is accessed atomically. Writers publish with release stores and
// readers use acquire loads: a reader that observes a node id is guaranteed to also observe
// that node's vector and label, which were written before the id was first published. On
// x86-64 both compile to plain MOVs; on AArch64 to LDAR/STLR.
inline std::uint32_t load_link(const std::uint32_t& slot) noexcept {
  return std::atomic_ref<std::uint32_t>(const_cast<std::uint32_t&>(slot)).load(std::memory_order_acquire);
}

inline void store_link(std::uint32_t& slot, std::uint32_t value) noexcept {
  std::atomic_ref<std::uint32_t>(slot).store(value, std::memory_order_release);
}

// Large blocks are backed by 2 MiB transparent huge pages: search touches nodes at random, and
// with 4 KiB pages nearly every node visit would also be a TLB miss. Measured on SIFT1M:
// +10-15% QPS and -16% p99 latency. The block is zeroed (every neighbor count starts at 0).
// ORYX_NO_HUGEPAGES=1 turns the request off, so benchmarks can compare libraries on equal pages
// (on a kernel where huge pages are opt-in, only a library that asks for them gets them).
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
bool huge_pages_wanted() {
  static const bool wanted = [] {
    const char* off = std::getenv("ORYX_NO_HUGEPAGES");
    return off == nullptr || off[0] == '\0' || std::string_view(off) == "0";
  }();
  return wanted;
}
#endif

void* allocate_block(std::size_t bytes) {
  constexpr std::size_t kHugePage = std::size_t{2} << 20;
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
  const bool huge = bytes >= kHugePage && huge_pages_wanted();
#else
  const bool huge = false;  // no transparent huge pages to ask for; 2 MiB alignment would only waste memory
#endif
  bytes = round_up(std::max<std::size_t>(bytes, 64), huge ? kHugePage : 64);
  void* block = std::aligned_alloc(huge ? kHugePage : 64, bytes);
  if (block == nullptr) throw std::bad_alloc();
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
  if (huge) madvise(block, bytes, MADV_HUGEPAGE);  // advisory; ignored if THP is disabled
#endif
  std::memset(block, 0, bytes);
  return block;
}

// Snapshot I/O goes through 8 MiB chunks and memcpy. A 1M-node index is millions of small
// fields; per-call iostream overhead made loading several times slower than the disk.
constexpr std::size_t kIoChunk = std::size_t{8} << 20;

class ChunkedReader {
 public:
  explicit ChunkedReader(const std::string& path) : path_(path), file_(std::fopen(path.c_str(), "rb")) {
    if (file_ == nullptr) throw std::runtime_error("cannot open " + path);
    buffer_.resize(kIoChunk);
  }
  ~ChunkedReader() { std::fclose(file_); }
  ChunkedReader(const ChunkedReader&) = delete;
  ChunkedReader& operator=(const ChunkedReader&) = delete;

  void read(void* dst, std::size_t bytes) {
    auto* out = static_cast<char*>(dst);
    while (bytes > 0) {
      if (pos_ == end_) {
        end_ = std::fread(buffer_.data(), 1, buffer_.size(), file_);
        pos_ = 0;
        if (end_ == 0) throw std::runtime_error(path_ + ": unexpected end of file");
      }
      const std::size_t take = std::min(bytes, end_ - pos_);
      std::memcpy(out, buffer_.data() + pos_, take);
      pos_ += take;
      out += take;
      bytes -= take;
    }
  }
  template <typename T>
  T pod() {
    T value;
    read(&value, sizeof(T));
    return value;
  }
  template <typename T>
  void array(T* data, std::size_t count) {
    read(data, count * sizeof(T));
  }

 private:
  std::string path_;
  std::FILE* file_;
  std::vector<char> buffer_;
  std::size_t pos_ = 0;
  std::size_t end_ = 0;
};

class ChunkedWriter {
 public:
  explicit ChunkedWriter(const std::string& path) : path_(path), file_(std::fopen(path.c_str(), "wb")) {
    if (file_ == nullptr) throw std::runtime_error("cannot open " + path + " for writing");
    buffer_.reserve(kIoChunk);
  }
  ~ChunkedWriter() {
    if (file_ != nullptr) std::fclose(file_);
  }
  ChunkedWriter(const ChunkedWriter&) = delete;
  ChunkedWriter& operator=(const ChunkedWriter&) = delete;

  void write(const void* src, std::size_t bytes) {
    if (buffer_.size() + bytes > kIoChunk) flush();
    if (bytes >= kIoChunk) {
      put(src, bytes);
    } else {
      const auto* in = static_cast<const char*>(src);
      buffer_.insert(buffer_.end(), in, in + bytes);
    }
  }
  template <typename T>
  void pod(const T& value) {
    write(&value, sizeof(T));
  }
  template <typename T>
  void array(const T* data, std::size_t count) {
    write(data, count * sizeof(T));
  }
  // Flushes and closes, reporting any error (a failed close can mean lost data).
  void finish() {
    flush();
    const int result = std::fclose(file_);
    file_ = nullptr;
    if (result != 0) throw std::runtime_error("failed to write " + path_);
  }

 private:
  void flush() {
    put(buffer_.data(), buffer_.size());
    buffer_.clear();
  }
  void put(const void* data, std::size_t bytes) {
    if (bytes > 0 && std::fwrite(data, 1, bytes, file_) != bytes) throw std::runtime_error("failed to write " + path_);
  }

  std::string path_;
  std::FILE* file_;
  std::vector<char> buffer_;
};

// Largest M a snapshot may declare: far above any useful value, low enough that sizes can't overflow.
constexpr std::size_t kMaxM = 1024;

// Filtered search: Auto's choice of strategy (see choose_strategy). Exact wins while the allowed set
// is small enough to score outright; Inline while most nodes pass, since it walks the graph at
// almost full speed; TwoHop in between.
constexpr std::size_t kExactPerEf = 64;  // scan when allowed <= kExactPerEf * ef
constexpr double kInlineSelectivity = 0.5;

}  // namespace

std::string_view to_string(FilterStrategy strategy) noexcept {
  switch (strategy) {
    case FilterStrategy::Auto: return "auto";
    case FilterStrategy::Exact: return "exact";
    case FilterStrategy::Inline: return "inline";
    case FilterStrategy::TwoHop: return "two-hop";
  }
  return "unknown";
}

// Scores nodes against one query vector. With `adc` it reads only the node's SQ8 codes
// (asymmetric distance: the query stays in float, its dot product with the dequantized node is
// folded into per-dimension weights once); otherwise it uses full-precision vectors.
//   dot(q, x^) = sum_i q_i * (min_i + code_i * scale_i) = C + sum_i w_i * code_i
//   L2:  |q - x^|^2 = |q|^2 - 2 dot + |x^|^2      IP/cosine: 1 - dot
class HnswIndex::Scorer {
 public:
  Scorer(const HnswIndex& index, const float* query, bool adc) : index_(index), query_(query), adc_(adc) {
    if (!adc_) return;
    const std::size_t dim = index.params_.dim;
    weights_.resize(dim);
    float c = 0.0f;
    float qq = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
      weights_[i] = query[i] * index.sq8_scale_[i];
      c += query[i] * index.sq8_min_[i];
      qq += query[i] * query[i];
    }
    l2_ = index.params_.metric == Metric::L2;
    base_ = l2_ ? qq - 2.0f * c : 1.0f - c;
  }

  float operator()(id_t id) const noexcept {
    if (adc_) {
      const float dot = index_.sq8_dot_(weights_.data(), index_.codes_of(id), weights_.size());
      return l2_ ? base_ + index_.code_norm(id) - 2.0f * dot : base_ - dot;
    }
    return index_.dist_(query_, index_.full_vector(id), index_.params_.dim);
  }

  // What the distance computation will read first, for prefetching.
  const void* address(id_t id) const noexcept {
    return adc_ ? static_cast<const void*>(index_.record(id)) : static_cast<const void*>(index_.full_vector(id));
  }

 private:
  const HnswIndex& index_;
  const float* query_;
  bool adc_;
  bool l2_ = false;
  float base_ = 0.0f;
  std::vector<float> weights_;
};

void HnswIndex::AlignedFree::operator()(void* p) const noexcept { std::free(p); }

HnswIndex::HnswIndex(const HnswParams& params)
    : params_(params),
      simd_level_(active_simd_level()),
      dist_(get_distance_fn(params.metric, simd_level_)),
      sq8_dot_(get_sq8_dot_fn(simd_level_)),
      max_links0_(2 * params.M),
      level_mult_(params.M > 1 ? 1.0 / std::log(static_cast<double>(params.M)) : 1.0),
      rng_(params.seed) {
  if (params_.dim == 0) throw std::invalid_argument("dim must be > 0");
  if (params_.max_elements == 0) throw std::invalid_argument("max_elements must be > 0");
  if (params_.max_elements >= std::numeric_limits<id_t>::max()) {
    throw std::invalid_argument("max_elements must fit in 32-bit node ids");
  }
  if (params_.M < 2) throw std::invalid_argument("M must be >= 2");

  // Record layout. None: [float x dim][links]. SQ8: [float norm][uint8 x dim][links].
  links0_offset_ = sq8() ? round_up(sizeof(float) + params_.dim, sizeof(id_t)) : params_.dim * sizeof(float);
  node_stride_ = round_up(links0_offset_ + (1 + max_links0_) * sizeof(id_t), 64);
  level0_ = allocate_level0();
  if (sq8() && params_.keep_vectors) {
    full_.reset(static_cast<float*>(allocate_block(params_.max_elements * params_.dim * sizeof(float))));
  }

  upper_links_.resize(params_.max_elements);
  levels_.resize(params_.max_elements, 0);
  labels_.resize(params_.max_elements, 0);
  deleted_ = std::make_unique<std::uint8_t[]>(params_.max_elements);
  node_locks_ = std::make_unique<std::mutex[]>(params_.max_elements);
  visited_pool_ = std::make_unique<detail::VisitedPool>(params_.max_elements);
  label_to_id_.reserve(params_.max_elements);
}

HnswIndex::~HnswIndex() = default;

std::unique_ptr<std::byte[], HnswIndex::AlignedFree> HnswIndex::allocate_level0() const {
  return std::unique_ptr<std::byte[], AlignedFree>(
      static_cast<std::byte*>(allocate_block(node_stride_ * params_.max_elements)));
}

const float* HnswIndex::full_vector(id_t id) const noexcept {
  if (!sq8()) return reinterpret_cast<const float*>(record(id));
  return full_ ? full_.get() + static_cast<std::size_t>(id) * params_.dim : nullptr;
}

HnswIndex::id_t* HnswIndex::links_of(id_t id, int layer) noexcept {
  if (layer == 0) return reinterpret_cast<id_t*>(level0_.get() + id * node_stride_ + links0_offset_);
  return upper_links_[id].get() + static_cast<std::size_t>(layer - 1) * (1 + params_.M);
}

const HnswIndex::id_t* HnswIndex::links_of(id_t id, int layer) const noexcept {
  return const_cast<HnswIndex*>(this)->links_of(id, layer);
}

bool HnswIndex::is_deleted(id_t id) const noexcept {
  return std::atomic_ref<std::uint8_t>(deleted_[id]).load(std::memory_order_acquire) != 0;
}

// ---------------------------------------------------------------------------------------------
// Scalar quantization

void HnswIndex::train(const float* data, std::size_t n) {
  if (!sq8()) return;
  if (size() > 0) throw std::logic_error("train() must be called before the first add()");
  if (n == 0) throw std::invalid_argument("train() needs at least one vector");
  const std::size_t dim = params_.dim;
  std::vector<float> lo(dim, std::numeric_limits<float>::max());
  std::vector<float> hi(dim, std::numeric_limits<float>::lowest());
  std::vector<float> row(dim);
  for (std::size_t r = 0; r < n; ++r) {
    std::copy(data + r * dim, data + (r + 1) * dim, row.begin());
    if (params_.metric == Metric::Cosine) normalize(row.data(), dim);  // train on what gets stored
    for (std::size_t i = 0; i < dim; ++i) {
      lo[i] = std::min(lo[i], row[i]);
      hi[i] = std::max(hi[i], row[i]);
    }
  }
  sq8_min_ = lo;
  sq8_scale_.resize(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    const float range = hi[i] - lo[i];
    sq8_scale_[i] = range > 0.0f ? range / 255.0f : 1.0f;
  }
}

void HnswIndex::encode(const float* v, std::uint8_t* codes, float* norm) const noexcept {
  float sum = 0.0f;
  for (std::size_t i = 0; i < params_.dim; ++i) {
    const float q = std::nearbyint((v[i] - sq8_min_[i]) / sq8_scale_[i]);
    const auto code = static_cast<std::uint8_t>(std::clamp(q, 0.0f, 255.0f));
    codes[i] = code;
    const float decoded = sq8_min_[i] + code * sq8_scale_[i];
    sum += decoded * decoded;
  }
  *norm = sum;  // |x^|^2 of the dequantized vector, so ADC distances are self-consistent
}

void HnswIndex::decode(id_t id, float* out) const noexcept {
  const std::uint8_t* codes = codes_of(id);
  for (std::size_t i = 0; i < params_.dim; ++i) out[i] = sq8_min_[i] + codes[i] * sq8_scale_[i];
}

float HnswIndex::pair_distance(id_t a, id_t b) const noexcept {
  if (has_full_vectors()) return dist_(full_vector(a), full_vector(b), params_.dim);
  thread_local std::vector<float> va, vb;
  va.resize(params_.dim);
  vb.resize(params_.dim);
  decode(a, va.data());
  decode(b, vb.data());
  return dist_(va.data(), vb.data(), params_.dim);
}

// ---------------------------------------------------------------------------------------------
// Graph

int HnswIndex::random_level() {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  double u;
  {
    std::lock_guard lock(rng_lock_);
    u = uniform(rng_);
  }
  // l = floor(-ln(U) * mL) with U in (0, 1]; P(level >= l) = M^-l.
  const double level = -std::log(1.0 - u) * level_mult_;
  return std::min(static_cast<int>(level), kMaxLevel);
}

void HnswIndex::copy_links(id_t id, int layer, std::vector<id_t>& out) const {
  const id_t* links = links_of(id, layer);
  const id_t count = load_link(links[0]);
  out.resize(count);
  for (id_t i = 0; i < count; ++i) out[i] = load_link(links[1 + i]);
}

void HnswIndex::write_links(id_t id, int layer, const std::vector<Candidate>& neighbors) noexcept {
  id_t* links = links_of(id, layer);
  // Slots first, then the count: a concurrent reader sees either the old list or the new one
  // with some slots already replaced, and every slot always holds a valid node id.
  for (std::size_t j = 0; j < neighbors.size(); ++j) store_link(links[1 + j], neighbors[j].id);
  store_link(links[0], static_cast<id_t>(neighbors.size()));
}

int HnswIndex::max_level() const noexcept {
  const std::uint64_t entry = entry_.load(std::memory_order_acquire);
  return static_cast<int>(entry >> 32) - 1;
}

HnswIndex::id_t HnswIndex::greedy_descend(const Scorer& scorer, id_t entry, int from_layer, int to_layer,
                                          SearchStats& stats, SearchTrace* trace) const {
  id_t current = entry;
  float current_dist = scorer(current);
  ++stats.distance_computations;

  for (int layer = from_layer; layer > to_layer; --layer) {
    bool improved = true;
    while (improved) {
      improved = false;
      const id_t expanding = current;
      const id_t* links = links_of(expanding, layer);
      const id_t count = load_link(links[0]);
      ++stats.nodes_expanded;
      for (id_t i = 0; i < count; ++i) {
        const id_t neighbor = load_link(links[1 + i]);
        const float d = scorer(neighbor);
        ++stats.distance_computations;
        const bool closer = d < current_dist;
        if (trace != nullptr) {
          trace->events.push_back({static_cast<std::uint32_t>(layer), labels_[expanding], labels_[neighbor], d, closer});
        }
        if (closer) {  // deleted nodes still route: they are only excluded from results
          current = neighbor;
          current_dist = d;
          improved = true;
        }
      }
    }
  }
  return current;
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(const Scorer& scorer, id_t entry, std::size_t ef,
                                                          int layer, SearchStats& stats, SearchTrace* trace) const {
  auto visited = visited_pool_->acquire();
  std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst> candidates;  // closest on top
  std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;       // farthest on top
  // Tombstoned nodes are explored (they keep the graph connected) but never kept as results, so
  // with deletions the beam must fill up before it may stop.
  const bool has_deletions = deleted_count_.load(std::memory_order_relaxed) > 0;
  constexpr float kInfinity = std::numeric_limits<float>::infinity();

  const float entry_dist = scorer(entry);
  ++stats.distance_computations;
  visited->test_and_set(entry);
  candidates.push({entry_dist, entry});
  if (!has_deletions || !is_deleted(entry)) best.push({entry_dist, entry});
  float bound = best.empty() ? kInfinity : best.top().distance;

  // Scratch space for one node's unvisited neighbors; the largest list is layer 0's.
  std::vector<id_t> batch(max_links0_);

  while (!candidates.empty()) {
    const Candidate current = candidates.top();
    // Every remaining candidate is farther than the worst result we keep: the beam has converged.
    if (current.distance > bound && (best.size() >= ef || !has_deletions)) break;
    candidates.pop();

    const id_t* links = links_of(current.id, layer);
    const id_t count = load_link(links[0]);
    ++stats.nodes_expanded;

    // Filter out visited neighbors first and prefetch every survivor's vector, so their cache
    // misses overlap instead of being paid one after another inside the distance loop.
    // Measured on SIFT1M: +12-18% QPS. Prefetching only the first line is enough; the hardware
    // stream prefetcher follows once the kernel starts reading (more lines measured no better).
    std::size_t pending = 0;
    for (id_t i = 0; i < count; ++i) {
      const id_t neighbor = load_link(links[1 + i]);
      if (visited->test_and_set(neighbor)) continue;
      batch[pending++] = neighbor;
      prefetch(scorer.address(neighbor));
    }

    for (std::size_t j = 0; j < pending; ++j) {
      const id_t neighbor = batch[j];
      const float d = scorer(neighbor);
      ++stats.distance_computations;
      const bool accepted = best.size() < ef || d < bound;
      if (trace != nullptr) {
        trace->events.push_back({static_cast<std::uint32_t>(layer), labels_[current.id], labels_[neighbor], d, accepted});
      }
      if (accepted) {
        candidates.push({d, neighbor});
        if (!has_deletions || !is_deleted(neighbor)) {
          best.push({d, neighbor});
          if (best.size() > ef) best.pop();
        }
        bound = best.empty() ? kInfinity : best.top().distance;
      }
    }
  }

  std::vector<Candidate> result(best.size());
  for (std::size_t i = result.size(); i-- > 0;) {
    result[i] = best.top();
    best.pop();
  }
  return result;  // closest first
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer_filtered(const Scorer& scorer, id_t entry, std::size_t ef,
                                                                   const SearchFilter& filter, bool two_hop,
                                                                   SearchStats& stats, SearchTrace* trace) const {
  auto visited = visited_pool_->acquire();
  std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst> candidates;  // closest on top
  std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;       // farthest on top
  const bool has_deletions = deleted_count_.load(std::memory_order_relaxed) > 0;
  const auto keepable = [&](id_t id) { return filter.allows(id) && (!has_deletions || !is_deleted(id)); };
  constexpr float kInfinity = std::numeric_limits<float>::infinity();

  const float entry_dist = scorer(entry);
  ++stats.distance_computations;
  visited->test_and_set(entry);
  candidates.push({entry_dist, entry});
  if (keepable(entry)) best.push({entry_dist, entry});
  float bound = best.empty() ? kInfinity : best.top().distance;

  // TwoHop gathers up to two lists' worth of allowed nodes per expansion.
  const std::size_t cap = two_hop ? 2 * max_links0_ : max_links0_;
  std::vector<id_t> batch(cap);
  std::vector<id_t> skipped;  // TwoHop: filtered-out neighbors, whose own lists are looked through
  skipped.reserve(max_links0_);

  while (!candidates.empty()) {
    const Candidate current = candidates.top();
    // Unlike the unfiltered walk, the beam must hold ef allowed results before it may stop.
    if (current.distance > bound && best.size() >= ef) break;
    candidates.pop();

    const id_t* links = links_of(current.id, 0);
    const id_t count = load_link(links[0]);
    ++stats.nodes_expanded;

    std::size_t pending = 0;
    skipped.clear();
    for (id_t i = 0; i < count; ++i) {
      const id_t neighbor = load_link(links[1 + i]);
      if (two_hop && !filter.allows(neighbor)) {
        skipped.push_back(neighbor);
        prefetch(links_of(neighbor, 0));
        continue;
      }
      if (visited->test_and_set(neighbor)) continue;
      batch[pending++] = neighbor;
      prefetch(scorer.address(neighbor));
    }
    // A filtered-out neighbor is never scored, but its neighbors are reachable through it. Without
    // this, a walk restricted to a small allowed set breaks into islands.
    for (std::size_t s = 0; s < skipped.size() && pending < cap; ++s) {
      const id_t via = skipped[s];
      if (visited->test_and_set(via)) continue;  // its list was already looked through
      const id_t* via_links = links_of(via, 0);
      const id_t via_count = load_link(via_links[0]);
      for (id_t i = 0; i < via_count && pending < cap; ++i) {
        const id_t neighbor = load_link(via_links[1 + i]);
        if (!filter.allows(neighbor) || visited->test_and_set(neighbor)) continue;
        batch[pending++] = neighbor;
        prefetch(scorer.address(neighbor));
      }
    }

    for (std::size_t j = 0; j < pending; ++j) {
      const id_t neighbor = batch[j];
      const float d = scorer(neighbor);
      ++stats.distance_computations;
      const bool accepted = best.size() < ef || d < bound;
      if (trace != nullptr) {
        trace->events.push_back({0, labels_[current.id], labels_[neighbor], d, accepted});
      }
      if (accepted) {
        candidates.push({d, neighbor});
        if (keepable(neighbor)) {
          best.push({d, neighbor});
          if (best.size() > ef) best.pop();
        }
        bound = best.empty() ? kInfinity : best.top().distance;
      }
    }
  }

  std::vector<Candidate> result(best.size());
  for (std::size_t i = result.size(); i-- > 0;) {
    result[i] = best.top();
    best.pop();
  }
  return result;  // closest first
}

std::vector<HnswIndex::Candidate> HnswIndex::scan_allowed(const Scorer& scorer, const SearchFilter& filter,
                                                          std::size_t keep, SearchStats& stats) const {
  const bool has_deletions = deleted_count_.load(std::memory_order_relaxed) > 0;
  std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;  // farthest on top
  const std::vector<id_t>& ids = filter.ids_;
  // The allowed slots are sorted, so this mostly streams forward; prefetch a few rows ahead.
  constexpr std::size_t kAhead = 8;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    if (i + kAhead < ids.size()) prefetch(scorer.address(ids[i + kAhead]));
    const id_t id = ids[i];
    if (has_deletions && is_deleted(id)) continue;
    const float d = scorer(id);
    ++stats.distance_computations;
    if (best.size() < keep) {
      best.push({d, id});
    } else if (d < best.top().distance) {
      best.pop();
      best.push({d, id});
    }
  }
  std::vector<Candidate> result(best.size());
  for (std::size_t i = result.size(); i-- > 0;) {
    result[i] = best.top();
    best.pop();
  }
  return result;  // closest first
}

FilterStrategy HnswIndex::choose_strategy(const SearchFilter& filter, std::size_t ef) const noexcept {
  if (filter.allowed() <= kExactPerEf * ef) return FilterStrategy::Exact;
  if (filter.selectivity() >= kInlineSelectivity) return FilterStrategy::Inline;
  return FilterStrategy::TwoHop;
}

SearchFilter HnswIndex::filter_from_ids(std::vector<id_t> ids) const {
  SearchFilter filter;
  filter.bits_.assign((size() + 63) / 64, 0);
  std::sort(ids.begin(), ids.end());  // sequential order for exact scans
  filter.ids_.reserve(ids.size());
  for (const id_t id : ids) {
    std::uint64_t& word = filter.bits_[id >> 6];
    const std::uint64_t bit = std::uint64_t{1} << (id & 63);
    if ((word & bit) != 0) continue;  // listed twice
    word |= bit;
    filter.ids_.push_back(id);
  }
  filter.universe_ = live_count();
  return filter;
}

SearchFilter HnswIndex::make_filter(const std::function<bool(std::uint64_t label)>& allow) const {
  std::vector<id_t> ids;
  const id_t n = static_cast<id_t>(size());
  for (id_t id = 0; id < n; ++id) {
    if (!is_deleted(id) && allow(labels_[id])) ids.push_back(id);
  }
  return filter_from_ids(std::move(ids));
}

SearchFilter HnswIndex::make_filter(const std::uint64_t* labels, std::size_t count) const {
  std::vector<id_t> ids;
  ids.reserve(count);
  {
    std::lock_guard lock(label_lock_);
    for (std::size_t i = 0; i < count; ++i) {
      const auto it = label_to_id_.find(labels[i]);
      if (it != label_to_id_.end()) ids.push_back(it->second);  // removed labels are not in the map
    }
  }
  return filter_from_ids(std::move(ids));
}

std::vector<HnswIndex::Candidate> HnswIndex::select_neighbors(std::vector<Candidate> candidates,
                                                              std::size_t max_count) const {
  if (candidates.size() <= max_count) return candidates;
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });

  std::vector<Candidate> kept;
  kept.reserve(max_count);
  for (const Candidate& candidate : candidates) {
    bool diverse = true;
    for (const Candidate& k : kept) {
      if (pair_distance(candidate.id, k.id) < candidate.distance) {
        diverse = false;
        break;
      }
    }
    if (diverse) {
      kept.push_back(candidate);
      if (kept.size() == max_count) break;
    }
  }
  return kept;
}

void HnswIndex::connect(id_t id, const std::vector<Candidate>& neighbors, int layer) {
  const std::size_t capacity = max_links(layer);

  // Forward links. Another thread may already have linked itself to this node on this layer
  // (it can reach us from a layer above), so merge with what is there instead of overwriting.
  std::vector<id_t> existing;
  {
    std::lock_guard lock(node_locks_[id]);
    copy_links(id, layer, existing);
    std::vector<Candidate> merged = neighbors;
    for (const id_t e : existing) {
      const bool duplicate = std::any_of(merged.begin(), merged.end(), [&](const Candidate& c) { return c.id == e; });
      if (!duplicate) merged.push_back({pair_distance(id, e), e});
    }
    write_links(id, layer, select_neighbors(std::move(merged), capacity));
  }

  // Reverse links, one neighbor lock at a time.
  for (const Candidate& neighbor : neighbors) {
    std::lock_guard lock(node_locks_[neighbor.id]);
    copy_links(neighbor.id, layer, existing);
    if (existing.size() < capacity) {
      id_t* links = links_of(neighbor.id, layer);
      store_link(links[1 + existing.size()], id);
      store_link(links[0], static_cast<id_t>(existing.size() + 1));
      continue;
    }
    // The neighbor is full: re-run the heuristic over its current links plus the new node.
    std::vector<Candidate> pool;
    pool.reserve(existing.size() + 1);
    pool.push_back({neighbor.distance, id});
    for (const id_t e : existing) pool.push_back({pair_distance(neighbor.id, e), e});
    write_links(neighbor.id, layer, select_neighbors(std::move(pool), capacity));
  }
}

void HnswIndex::add(std::uint64_t label, const float* vector) {
  if (!trained()) throw std::logic_error("an SQ8 index must be trained before add()");
  id_t id;
  {
    std::lock_guard lock(label_lock_);
    if (label_to_id_.contains(label)) throw std::invalid_argument("duplicate label " + std::to_string(label));
    const std::size_t count = count_.load(std::memory_order_relaxed);
    if (count >= params_.max_elements) throw std::length_error("index is full");
    id = static_cast<id_t>(count);
    count_.store(count + 1, std::memory_order_release);
    label_to_id_.emplace(label, id);
  }

  // Everything below is written before the node becomes reachable. Other threads only reach it
  // through a neighbor list slot, published with a release store (see load_link / store_link).
  std::vector<float> normalized;
  if (params_.metric == Metric::Cosine) {
    normalized.assign(vector, vector + params_.dim);
    normalize(normalized.data(), params_.dim);
    vector = normalized.data();
  }
  if (sq8()) {
    auto* codes = reinterpret_cast<std::uint8_t*>(record(id) + sizeof(float));
    encode(vector, codes, reinterpret_cast<float*>(record(id)));
    if (full_) std::memcpy(full_.get() + static_cast<std::size_t>(id) * params_.dim, vector, params_.dim * sizeof(float));
  } else {
    std::memcpy(record(id), vector, params_.dim * sizeof(float));
  }
  labels_[id] = label;
  const int level = random_level();
  levels_[id] = static_cast<std::int8_t>(level);
  if (level > 0) upper_links_[id] = std::make_unique<id_t[]>(static_cast<std::size_t>(level) * (1 + params_.M));

  std::uint64_t entry = entry_.load(std::memory_order_acquire);
  if (entry == kEmpty) {
    std::lock_guard lock(entry_lock_);
    entry = entry_.load(std::memory_order_acquire);
    if (entry == kEmpty) {
      entry_.store(pack_entry(id, level), std::memory_order_release);
      return;
    }
  }

  // Build with exact distances whenever full-precision vectors exist: a better graph costs
  // nothing at query time.
  const Scorer scorer(*this, vector, /*adc=*/!has_full_vectors());
  id_t entry_point = static_cast<id_t>(entry & 0xffffffffu);
  const int top_level = static_cast<int>(entry >> 32) - 1;
  SearchStats unused;
  if (level < top_level) entry_point = greedy_descend(scorer, entry_point, top_level, level, unused, nullptr);

  for (int layer = std::min(level, top_level); layer >= 0; --layer) {
    std::vector<Candidate> candidates =
        search_layer(scorer, entry_point, params_.ef_construction, layer, unused, nullptr);
    // A concurrent insert may have linked to us already, making us reachable from our own search.
    std::erase_if(candidates, [id](const Candidate& c) { return c.id == id; });
    if (candidates.empty()) continue;
    entry_point = candidates.front().id;
    connect(id, select_neighbors(std::move(candidates), params_.M), layer);
  }

  if (level > top_level) {
    std::lock_guard lock(entry_lock_);
    if (level > max_level()) entry_.store(pack_entry(id, level), std::memory_order_release);
  }
}

bool HnswIndex::remove(std::uint64_t label) {
  id_t id;
  {
    std::lock_guard lock(label_lock_);
    const auto it = label_to_id_.find(label);
    if (it == label_to_id_.end()) return false;
    id = it->second;
    label_to_id_.erase(it);
  }
  std::atomic_ref<std::uint8_t>(deleted_[id]).store(1, std::memory_order_release);
  deleted_count_.fetch_add(1, std::memory_order_acq_rel);
  return true;
}

std::vector<SearchResult> HnswIndex::search(const float* query, std::size_t k, const SearchOptions& options) const {
  const std::uint64_t entry = entry_.load(std::memory_order_acquire);
  if (k == 0 || entry == kEmpty) return {};

  std::vector<float> normalized;
  if (params_.metric == Metric::Cosine) {
    normalized.assign(query, query + params_.dim);
    normalize(normalized.data(), params_.dim);
    query = normalized.data();
  }

  const id_t entry_point = static_cast<id_t>(entry & 0xffffffffu);
  const int top_level = static_cast<int>(entry >> 32) - 1;
  SearchTrace* trace = options.trace;
  if (trace != nullptr) {
    trace->entry_label = labels_[entry_point];
    trace->entry_layer = static_cast<std::uint32_t>(top_level);
    trace->events.clear();
  }

  SearchStats stats;
  const Scorer scorer(*this, query, /*adc=*/sq8());
  const std::size_t ef = std::max(options.ef, k);
  std::vector<Candidate> candidates;
  if (options.filter == nullptr) {
    const id_t start = greedy_descend(scorer, entry_point, top_level, 0, stats, trace);
    candidates = search_layer(scorer, start, ef, 0, stats, trace);
  } else {
    const SearchFilter& filter = *options.filter;
    FilterStrategy strategy =
        options.strategy == FilterStrategy::Auto ? choose_strategy(filter, ef) : options.strategy;
    if (strategy != FilterStrategy::Exact) {
      const id_t start = greedy_descend(scorer, entry_point, top_level, 0, stats, trace);
      candidates = search_layer_filtered(scorer, start, ef, filter, strategy == FilterStrategy::TwoHop, stats, trace);
      // A walk can strand in a region with too few allowed nodes; Auto then answers exactly.
      if (options.strategy == FilterStrategy::Auto && candidates.size() < std::min(k, filter.allowed())) {
        strategy = FilterStrategy::Exact;
      }
    }
    if (strategy == FilterStrategy::Exact) candidates = scan_allowed(scorer, filter, ef, stats);
    stats.strategy = strategy;
  }

  // SQ8: the beam ran on approximate distances; re-score the best candidates exactly.
  if (sq8() && full_ && !candidates.empty()) {
    std::size_t rerank = options.rerank == 0 ? candidates.size() : std::max(options.rerank, k);
    rerank = std::min(rerank, candidates.size());
    for (std::size_t i = 0; i < rerank; ++i) prefetch(full_vector(candidates[i].id));
    for (std::size_t i = 0; i < rerank; ++i) {
      candidates[i].distance = dist_(query, full_vector(candidates[i].id), params_.dim);
    }
    stats.distance_computations += rerank;
    std::sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(rerank),
              [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });
  }

  std::vector<SearchResult> results;
  results.reserve(std::min(k, candidates.size()));
  for (std::size_t i = 0; i < candidates.size() && i < k; ++i) {
    results.push_back({labels_[candidates[i].id], candidates[i].distance});
  }
  if (options.stats != nullptr) *options.stats = stats;
  return results;
}

std::unique_ptr<HnswIndex> HnswIndex::compacted(std::size_t threads) const {
  auto out = std::make_unique<HnswIndex>(params_);
  out->sq8_min_ = sq8_min_;
  out->sq8_scale_ = sq8_scale_;
  std::vector<id_t> live;
  live.reserve(live_count());
  for (id_t id = 0; id < size(); ++id) {
    if (!is_deleted(id)) live.push_back(id);
  }
  parallel_for(0, live.size(), threads, [&](std::size_t i) {
    const id_t id = live[i];
    if (has_full_vectors()) {
      out->add(labels_[id], full_vector(id));
    } else {
      std::vector<float> v(params_.dim);
      decode(id, v.data());
      out->add(labels_[id], v.data());
    }
  });
  return out;
}

bool HnswIndex::contains(std::uint64_t label) const {
  std::lock_guard lock(label_lock_);
  return label_to_id_.contains(label);
}

int HnswIndex::level_of(std::uint64_t label) const {
  std::lock_guard lock(label_lock_);
  const auto it = label_to_id_.find(label);
  if (it == label_to_id_.end()) throw std::out_of_range("unknown label " + std::to_string(label));
  return levels_[it->second];
}

std::vector<std::uint64_t> HnswIndex::neighbors_of(std::uint64_t label, int layer) const {
  id_t id;
  {
    std::lock_guard lock(label_lock_);
    const auto it = label_to_id_.find(label);
    if (it == label_to_id_.end()) throw std::out_of_range("unknown label " + std::to_string(label));
    id = it->second;
  }
  if (layer < 0 || layer > levels_[id]) return {};
  std::vector<id_t> links;
  copy_links(id, layer, links);
  std::vector<std::uint64_t> result;
  result.reserve(links.size());
  for (id_t n : links) result.push_back(labels_[n]);
  return result;
}

// What an index with these parameters allocates up front, mirroring the constructor's layout:
// the layer-0 block plus, for SQ8 with kept vectors, the full-precision copies.
std::size_t HnswIndex::estimated_bytes(const HnswParams& params) noexcept {
  const bool sq8 = params.quantization == Quantization::SQ8;
  const std::size_t links0 = sq8 ? round_up(sizeof(float) + params.dim, sizeof(id_t)) : params.dim * sizeof(float);
  const std::size_t stride = round_up(links0 + (1 + 2 * params.M) * sizeof(id_t), 64);
  std::size_t bytes = stride * params.max_elements;
  if (sq8 && params.keep_vectors) bytes += params.max_elements * params.dim * sizeof(float);
  return bytes;
}

std::size_t HnswIndex::memory_bytes() const noexcept {
  std::size_t bytes = node_stride_ * params_.max_elements;
  if (full_) bytes += params_.max_elements * params_.dim * sizeof(float);
  bytes += params_.max_elements * (sizeof(std::uint64_t) + sizeof(std::int8_t) + sizeof(std::uint8_t) +
                                   sizeof(std::mutex) + sizeof(std::unique_ptr<id_t[]>));
  const std::size_t count = size();
  for (std::size_t i = 0; i < count; ++i) {
    bytes += static_cast<std::size_t>(levels_[i]) * (1 + params_.M) * sizeof(id_t);
  }
  // Rough cost of an unordered_map node plus its bucket pointer.
  bytes += count * (sizeof(std::uint64_t) + sizeof(id_t) + 2 * sizeof(void*)) + params_.max_elements * sizeof(void*);
  return bytes;
}

// ---------------------------------------------------------------------------------------------
// Snapshots
//
// v2 (little-endian): magic, version, dim, metric, max_elements, M, ef_construction, seed,
// quantization, keep_vectors, count, entry, deleted_count, [SQ8: trained, min[], scale[]], then
// per node: vector payload (floats, or SQ8 norm + codes), layer-0 links, label, level, deleted
// flag, upper-layer links, [SQ8 + keep_vectors: full floats]. The format does not depend on
// in-memory padding. v1 (no quantization, no tombstones) is still readable.

void HnswIndex::save(const std::string& path) const {
  ChunkedWriter out(path);
  out.write(kMagic, sizeof(kMagic));
  out.pod(kFormatVersion);
  out.pod<std::uint64_t>(params_.dim);
  out.pod<std::uint32_t>(static_cast<std::uint32_t>(params_.metric));
  out.pod<std::uint64_t>(params_.max_elements);
  out.pod<std::uint64_t>(params_.M);
  out.pod<std::uint64_t>(params_.ef_construction);
  out.pod<std::uint64_t>(params_.seed);
  out.pod<std::uint32_t>(static_cast<std::uint32_t>(params_.quantization));
  out.pod<std::uint8_t>(params_.keep_vectors ? 1 : 0);
  const std::size_t count = size();
  out.pod<std::uint64_t>(count);
  out.pod<std::uint64_t>(entry_.load(std::memory_order_acquire));
  out.pod<std::uint64_t>(deleted_count());
  if (sq8()) {
    out.pod<std::uint8_t>(sq8_scale_.empty() ? 0 : 1);
    if (!sq8_scale_.empty()) {
      out.array(sq8_min_.data(), params_.dim);
      out.array(sq8_scale_.data(), params_.dim);
    }
  }

  for (id_t id = 0; id < count; ++id) {
    std::lock_guard lock(node_locks_[id]);
    if (sq8()) {
      out.pod(code_norm(id));
      out.array(codes_of(id), params_.dim);
    } else {
      out.array(full_vector(id), params_.dim);
    }
    out.array(links_of(id, 0), 1 + max_links0_);
    out.pod(labels_[id]);
    out.pod(levels_[id]);
    out.pod<std::uint8_t>(is_deleted(id) ? 1 : 0);
    if (levels_[id] > 0) out.array(upper_links_[id].get(), static_cast<std::size_t>(levels_[id]) * (1 + params_.M));
    if (full_) out.array(full_vector(id), params_.dim);
  }
  out.finish();
}

// Every link must name a loaded node that exists on the link's layer: a search follows links
// blindly, and an upper-layer list of a node that doesn't reach that layer doesn't exist (a corrupt
// file could otherwise make it read out of bounds or through a null list).
bool HnswIndex::links_are_valid(std::size_t count) const noexcept {
  for (id_t id = 0; id < count; ++id) {
    for (int layer = 0; layer <= levels_[id]; ++layer) {
      const id_t* links = links_of(id, layer);
      if (links[0] > max_links(layer)) return false;
      for (id_t i = 1; i <= links[0]; ++i) {
        if (links[i] >= count || levels_[links[i]] < layer) return false;
      }
    }
  }
  return true;
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path, std::size_t max_bytes) {
  ChunkedReader in(path);
  char magic[sizeof(kMagic)];
  in.read(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error(path + " is not an Oryx index");
  const auto version = in.pod<std::uint32_t>();
  if (version != 1 && version != 2) throw std::runtime_error(path + ": unsupported format version");

  HnswParams params;
  params.dim = in.pod<std::uint64_t>();
  params.metric = static_cast<Metric>(in.pod<std::uint32_t>());
  params.max_elements = in.pod<std::uint64_t>();
  params.M = in.pod<std::uint64_t>();
  params.ef_construction = in.pod<std::uint64_t>();
  params.seed = in.pod<std::uint64_t>();
  if (version >= 2) {
    const auto quantization = in.pod<std::uint32_t>();
    if (quantization > static_cast<std::uint32_t>(Quantization::SQ8)) throw std::runtime_error(path + ": bad quantization");
    params.quantization = static_cast<Quantization>(quantization);
    params.keep_vectors = in.pod<std::uint8_t>() != 0;
  }
  const std::size_t count = in.pod<std::uint64_t>();
  const std::uint64_t entry = in.pod<std::uint64_t>();
  if (version >= 2) (void)in.pod<std::uint64_t>();  // deleted count, recomputed from the flags
  if (params.dim == 0 || params.dim > (1u << 20) || params.M < 2 || params.M > kMaxM || count > params.max_elements ||
      params.max_elements >= std::numeric_limits<id_t>::max()) {
    throw std::runtime_error(path + ": corrupt header");
  }
  // Every node stores at least its vector (or codes), its layer-0 list, label and level: a header
  // claiming more nodes than the file can hold is corrupt, and is caught before allocating.
  const bool sq8_file = params.quantization == Quantization::SQ8;
  const std::size_t per_node = (sq8_file ? sizeof(float) + params.dim : params.dim * sizeof(float)) +
                               (1 + 2 * params.M) * sizeof(id_t) + sizeof(std::uint64_t) + 1;
  if (count > std::filesystem::file_size(path) / per_node) throw std::runtime_error(path + ": truncated");
  if (max_bytes != 0 && estimated_bytes(params) > max_bytes) {
    throw std::runtime_error(path + ": the index would need " + std::to_string(estimated_bytes(params) >> 20) +
                             " MiB, over the " + std::to_string(max_bytes >> 20) + " MiB limit");
  }

  auto index = std::make_unique<HnswIndex>(params);
  if (index->sq8() && in.pod<std::uint8_t>() != 0) {
    index->sq8_min_.resize(params.dim);
    index->sq8_scale_.resize(params.dim);
    in.array(index->sq8_min_.data(), params.dim);
    in.array(index->sq8_scale_.data(), params.dim);
  }
  std::size_t deleted = 0;
  for (id_t id = 0; id < count; ++id) {
    if (index->sq8()) {
      in.read(index->record(id), sizeof(float) + params.dim);
    } else {
      in.array(reinterpret_cast<float*>(index->record(id)), params.dim);
    }
    id_t* links0 = index->links_of(id, 0);
    in.array(links0, 1 + index->max_links0_);
    index->labels_[id] = in.pod<std::uint64_t>();
    const auto level = in.pod<std::int8_t>();
    const bool is_deleted = version >= 2 && in.pod<std::uint8_t>() != 0;
    if (level < 0 || level > kMaxLevel || links0[0] > index->max_links0_) {
      throw std::runtime_error(path + ": corrupt node record");
    }
    index->levels_[id] = level;
    if (level > 0) {
      const std::size_t n = static_cast<std::size_t>(level) * (1 + params.M);
      index->upper_links_[id] = std::make_unique<id_t[]>(n);
      in.array(index->upper_links_[id].get(), n);
    }
    if (index->full_) in.array(index->full_.get() + static_cast<std::size_t>(id) * params.dim, params.dim);
    if (is_deleted) {
      index->deleted_[id] = 1;
      ++deleted;
    } else if (!index->label_to_id_.emplace(index->labels_[id], id).second) {
      throw std::runtime_error(path + ": duplicate live label");
    }
  }

  if (!index->links_are_valid(count)) throw std::runtime_error(path + ": corrupt neighbor list");
  const id_t entry_id = static_cast<id_t>(entry & 0xffffffffu);
  const std::uint64_t entry_level_plus_one = entry >> 32;  // unsigned: a crafted value can't overflow
  // An empty index has no entry point; otherwise it must be a node on its own top layer.
  if (count == 0 ? entry != kEmpty : (entry_id >= count || entry_level_plus_one != static_cast<std::uint64_t>(index->levels_[entry_id]) + 1)) {
    throw std::runtime_error(path + ": corrupt entry point");
  }

  index->deleted_count_.store(deleted, std::memory_order_release);
  index->count_.store(count, std::memory_order_release);
  index->entry_.store(entry, std::memory_order_release);
  return index;
}


namespace {

constexpr char kGraphMagic[8] = {'O', 'R', 'Y', 'X', 'G', 'R', 'P', 'H'};
constexpr std::uint32_t kGraphVersion = 1;

class ByteWriter {
 public:
  template <typename T>
  void pod(T value) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), p, p + sizeof(T));
  }
  std::vector<std::uint8_t> bytes;
};

class ByteReader {
 public:
  ByteReader(const std::uint8_t* data, std::size_t size) : data_(data), size_(size) {}
  template <typename T>
  T pod() {
    if (size_ - pos_ < sizeof(T)) throw std::runtime_error("graph: truncated");
    T value;
    std::memcpy(&value, data_ + pos_, sizeof(T));
    pos_ += sizeof(T);
    return value;
  }
  bool done() const noexcept { return pos_ == size_; }

 private:
  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t pos_ = 0;
};

}  // namespace

std::vector<std::uint8_t> HnswIndex::export_graph() const {
  if (sq8()) throw std::logic_error("export_graph: only full-precision indexes");
  if (deleted_count() != 0) throw std::logic_error("export_graph: compact the index first");
  if (max_links0_ > 0xff) throw std::logic_error("export_graph: M too large for the format");
  const std::size_t count = size();
  const bool narrow = count <= 0xffff;
  ByteWriter out;
  for (char c : kGraphMagic) out.pod(c);
  out.pod(kGraphVersion);
  out.pod<std::uint64_t>(params_.dim);
  out.pod<std::uint32_t>(static_cast<std::uint32_t>(params_.metric));
  out.pod<std::uint64_t>(params_.M);
  out.pod<std::uint64_t>(params_.ef_construction);
  out.pod<std::uint64_t>(count);
  out.pod<std::uint64_t>(entry_.load(std::memory_order_acquire));
  out.pod<std::uint8_t>(narrow ? 2 : 4);
  auto links = [&](const id_t* list) {
    out.pod(static_cast<std::uint8_t>(list[0]));
    for (id_t i = 1; i <= list[0]; ++i) {
      if (narrow) out.pod(static_cast<std::uint16_t>(list[i]));
      else out.pod(list[i]);
    }
  };
  for (id_t id = 0; id < count; ++id) {
    std::lock_guard lock(node_locks_[id]);
    out.pod(labels_[id]);
    out.pod(levels_[id]);
    for (int layer = 0; layer <= levels_[id]; ++layer) links(links_of(id, layer));
  }
  return std::move(out.bytes);
}

std::unique_ptr<HnswIndex> HnswIndex::import_graph(const HnswParams& params, const float* vectors, std::size_t n,
                                                   const std::uint8_t* bytes, std::size_t size) {
  ByteReader in(bytes, size);
  for (char c : kGraphMagic) {
    if (in.pod<char>() != c) throw std::runtime_error("graph: not an Oryx graph");
  }
  if (in.pod<std::uint32_t>() != kGraphVersion) throw std::runtime_error("graph: unsupported version");
  const auto dim = in.pod<std::uint64_t>();
  const auto metric = static_cast<Metric>(in.pod<std::uint32_t>());
  const auto M = in.pod<std::uint64_t>();
  const auto ef_construction = in.pod<std::uint64_t>();
  const auto count = in.pod<std::uint64_t>();
  const auto entry = in.pod<std::uint64_t>();
  const auto width = in.pod<std::uint8_t>();
  if (dim != params.dim || metric != params.metric || M != params.M || params.quantization != Quantization::None) {
    throw std::runtime_error("graph: built with different parameters");
  }
  if (count > n || count > params.max_elements || (width != 2 && width != 4)) throw std::runtime_error("graph: corrupt header");

  HnswParams actual = params;
  actual.ef_construction = ef_construction;
  auto index = std::make_unique<HnswIndex>(actual);
  std::vector<float> normalized(dim);
  for (id_t id = 0; id < count; ++id) {
    const auto label = in.pod<std::uint64_t>();
    const auto level = in.pod<std::int8_t>();
    if (label >= n || level < 0 || level > kMaxLevel) throw std::runtime_error("graph: corrupt node");
    if (!index->label_to_id_.emplace(label, id).second) throw std::runtime_error("graph: duplicate label");
    index->labels_[id] = label;
    index->levels_[id] = level;
    const float* vector = vectors + label * dim;
    if (metric == Metric::Cosine) {
      std::copy(vector, vector + dim, normalized.begin());
      normalize(normalized.data(), dim);
      vector = normalized.data();
    }
    std::memcpy(index->record(id), vector, dim * sizeof(float));
    if (level > 0) index->upper_links_[id] = std::make_unique<id_t[]>(static_cast<std::size_t>(level) * (1 + M));
    for (int layer = 0; layer <= level; ++layer) {
      id_t* list = index->links_of(id, layer);
      const std::size_t links = in.pod<std::uint8_t>();
      if (links > index->max_links(layer)) throw std::runtime_error("graph: corrupt neighbor list");
      list[0] = static_cast<id_t>(links);
      for (std::size_t i = 1; i <= links; ++i) {
        const id_t neighbor = width == 2 ? in.pod<std::uint16_t>() : in.pod<std::uint32_t>();
        if (neighbor >= count) throw std::runtime_error("graph: link out of range");
        list[i] = neighbor;
      }
    }
  }
  if (!in.done()) throw std::runtime_error("graph: trailing bytes");
  if (!index->links_are_valid(count)) throw std::runtime_error("graph: link to a node not on that layer");
  const id_t entry_id = static_cast<id_t>(entry & 0xffffffffu);
  const std::uint64_t entry_level_plus_one = entry >> 32;  // unsigned: a crafted value can't overflow
  if (count == 0 ? entry != kEmpty : (entry_id >= count || entry_level_plus_one != static_cast<std::uint64_t>(index->levels_[entry_id]) + 1)) {
    throw std::runtime_error("graph: corrupt entry point");
  }
  index->count_.store(count, std::memory_order_release);
  index->entry_.store(entry, std::memory_order_release);
  return index;
}

}  // namespace oryx
