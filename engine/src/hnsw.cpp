#include "epzero/hnsw.h"

#include <algorithm>
#include <array>
#include <chrono>
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

#include "epzero/parallel.h"
#include "half.h"
#include "platform.h"
#include "pca.h"
#include "rabitq.h"
#include "visited_pool.h"

namespace epzero {
namespace {

constexpr char kMagic[8] = {'E', 'P', 'Z', 'R', 'H', 'N', 'S', 'W'};
// Written before the project was renamed from Oryx; still loaded.
constexpr char kLegacyMagic[8] = {'O', 'R', 'Y', 'X', 'H', 'N', 'S', 'W'};
// v2: quantization + tombstones. v3: nav_dim and its directions. v4: v3 plus the IndexProfile. Each
// newer version is written only by indexes that need it, so others stay readable by older builds.
constexpr std::uint32_t kFormatVersion = 2;
constexpr std::uint32_t kFormatVersionNav = 3;
constexpr std::uint32_t kFormatVersionProfile = 4;
constexpr std::size_t kMaxProfileString = 4096;
constexpr std::size_t kRerankBlock = 16;
constexpr int kMaxLevel = 31;

bool is_default(const IndexProfile& p) noexcept {
  const IndexProfile d;
  return p.k == d.k && p.ef == d.ef && p.rerank == d.rerank && p.rerank_z == d.rerank_z && p.recall_target == 0.0f &&
         p.calibrated_size == 0 && p.nav_variance == 0.0f &&
         !p.built_on_codes && !p.refined && !p.filled && p.prefetch_depth == d.prefetch_depth && p.prefetch_lines == d.prefetch_lines &&
         p.machine.empty() && p.notes.empty();
}

std::size_t round_up(std::size_t value, std::size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

inline void prefetch(const void* p) noexcept { platform::prefetch(p); }

// The first `lines` cache lines at `p`.
inline void prefetch_lines(const void* p, std::size_t lines) noexcept {
  const auto* bytes = static_cast<const std::byte*>(p);
  for (std::size_t i = 0; i < lines; ++i) prefetch(bytes + 64 * i);
}

// Neighbor lists are read without any lock while writers (serialized per node by node_locks_)
// update them, so every slot is accessed atomically. Writers publish with release stores and
// readers use acquire loads: a reader that observes a node id is guaranteed to also observe
// that node's vector and label, which were written before the id was first published. On
// x86-64 both compile to plain MOVs; on AArch64 to LDAR/STLR. (The GCC/Clang atomic builtins,
// rather than std::atomic_ref: Apple's standard library doesn't have atomic_ref yet.)
inline std::uint32_t load_link(const std::uint32_t& slot) noexcept { return platform::load_acquire(slot); }

inline void store_link(std::uint32_t& slot, std::uint32_t value) noexcept {
  platform::store_release(slot, value);
}

// Large blocks are backed by 2 MiB transparent huge pages: search touches nodes at random, and
// with 4 KiB pages nearly every node visit would also be a TLB miss. Measured on SIFT1M:
// +10-15% QPS and -16% p99 latency. The block is zeroed (every neighbor count starts at 0).
// EPZERO_NO_HUGEPAGES=1 turns the request off, so benchmarks can compare libraries on equal pages
// (on a kernel where huge pages are opt-in, only a library that asks for them gets them).
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
bool huge_pages_wanted() {
  static const bool wanted = [] {
    const char* off = std::getenv("EPZERO_NO_HUGEPAGES");
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
  void* block = platform::aligned_alloc(huge ? kHugePage : 64, bytes);
  if (block == nullptr) throw std::bad_alloc();
#if defined(__linux__) && !defined(__EMSCRIPTEN__)
  if (huge) madvise(block, bytes, MADV_HUGEPAGE);  // advisory; ignored if THP is disabled
#endif
  // Zeroed on every core, in huge-page-sized pieces. The first write places a page on the memory of
  // the core that makes it, so zeroing on one thread put a whole index on one NUMA node of a
  // many-socket server, where every build and search thread then shared that node's bandwidth.
  constexpr std::size_t kPiece = std::size_t{2} << 20;
  if (bytes >= 32 * kPiece) {
    auto* bytes_at = static_cast<std::byte*>(block);
    parallel_for(0, (bytes + kPiece - 1) / kPiece, 0, [&](std::size_t piece) {
      const std::size_t start = piece * kPiece;
      std::memset(bytes_at + start, 0, std::min(kPiece, bytes - start));
    });
  } else {
    std::memset(block, 0, bytes);
  }
  return block;
}

// The beam of a layer search: the best `capacity` nodes found so far, sorted by distance, each
// marked once expanded, with a cursor at the closest one not yet expanded. It replaces a pair of
// binary heaps (candidates to expand, and the best found): one sorted array, mostly in L1, where
// an insert is a binary search and a short memmove and taking the next node is a cursor step,
// instead of two heaps whose pushes and pops branch unpredictably at every level (measured: a
// heap's pop alone was 13% of GloVe-100 search time). It also tells for free where a node
// landed, which is what patience needs ("did the best k change?").
class Beam {
 public:
  static constexpr std::size_t kRejected = std::numeric_limits<std::size_t>::max();
  struct Item {
    float distance;
    std::uint32_t id;
    bool expanded;
  };

  explicit Beam(std::size_t capacity) : items_(std::max<std::size_t>(capacity, 1) + 1), capacity_(std::max<std::size_t>(capacity, 1)) {}

  bool full() const noexcept { return size_ == capacity_; }
  std::size_t size() const noexcept { return size_; }
  const Item& operator[](std::size_t i) const noexcept { return items_[i]; }
  // Distances at least this far can't enter (infinite while there is room).
  float bound() const noexcept { return full() ? items_[size_ - 1].distance : std::numeric_limits<float>::infinity(); }

  // Inserts `id` unless the beam is full and `distance` is no better than its worst; returns the
  // position it took, or kRejected. Equal distances keep the earlier-found node first.
  std::size_t insert(float distance, std::uint32_t id) noexcept {
    if (full() && distance >= items_[size_ - 1].distance) return kRejected;
    std::size_t lo = 0;
    for (std::size_t n = size_; n > 0;) {
      const std::size_t half = n / 2;
      if (items_[lo + half].distance <= distance) {
        lo += half + 1;
        n -= half + 1;
      } else {
        n = half;
      }
    }
    std::memmove(&items_[lo + 1], &items_[lo], (size_ - lo) * sizeof(Item));  // the last falls off when full
    items_[lo] = {distance, id, false};
    if (size_ < capacity_) ++size_;
    if (lo < cursor_) cursor_ = lo;
    return lo;
  }

  bool has_next() const noexcept { return cursor_ < size_; }
  // The closest node not yet expanded, now marked expanded.
  std::uint32_t next() noexcept {
    Item& item = items_[cursor_];
    item.expanded = true;
    while (cursor_ < size_ && items_[cursor_].expanded) ++cursor_;
    return item.id;
  }
  float next_distance() const noexcept { return items_[cursor_].distance; }

 private:
  std::vector<Item> items_;  // one spare slot: an insert into a full beam shifts the last into it
  std::size_t capacity_;
  std::size_t size_ = 0;
  std::size_t cursor_ = 0;
};

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

// Filtered search: Auto's choice of strategy (see choose_strategy), calibrated on SIFT1M on a
// dedicated server (bench/results/pace/gold6226/sift1m-filter.json):
//  - Inline from a quarter of the nodes up: it walks the graph at almost full speed, and beat
//    TwoHop at equal recall from 30% (24% more queries per second at recall 0.95).
//  - Below 1.5%, TwoHop's walk falls apart (recall 0.91 at best at 1%, even at ef 320), so the
//    choice is between scoring every allowed vector and Inline, by estimated cost. Inline scores
//    about 90 * (ef / selectivity)^0.7 vectors (fitted, within 44%), at random places in memory;
//    a scan streams through the allowed rows at about half the cost per vector.
//  - In between, scan while the allowed set is small next to the beam, and TwoHop otherwise.
constexpr double kInlineSelectivity = 0.25;
constexpr double kTwoHopMinSelectivity = 0.015;
constexpr std::size_t kExactPerEf = 64;  // scan when allowed <= kExactPerEf * ef
constexpr double kInlineScale = 90;
constexpr double kInlineExponent = 0.7;
constexpr double kScanCostRatio = 0.5;  // a scanned vector's cost relative to one Inline scores

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
    // Where the full vectors live, resolved once per query: in the records (no quantization), in the
    // float copy, or in the 16-bit one. A lookup per node instead cost full-precision search 3-6%.
    dim_ = index.params_.dim;
    if (index.half_) {
      half_full_ = true;
      full_base_ = reinterpret_cast<const std::byte*>(index.half_.get());
      full_stride_ = dim_ * sizeof(std::uint16_t);
    } else if (!index.quantized()) {
      full_base_ = index.level0_.get();
      full_stride_ = index.node_stride_;
    } else if (index.full_) {
      full_base_ = reinterpret_cast<const std::byte*>(index.full_.get());
      full_stride_ = dim_ * sizeof(float);
    }
    if (!adc_) return;
    std::size_t dim = index.params_.dim;
    l2_ = index.params_.metric == Metric::L2;
    if (index.rabitq()) {
      rabitq::prepare(query, dim, index.rbq_center_.data(), index.rbq_signs_.data(), index.rbq_padded_, rbq_);
      rbq_words_ = rabitq::words(index.rbq_padded_);
      for (const float c : index.rbq_center_) center_sq_ += c * c;
      return;
    }
    if (index.params_.nav_dim != 0) {
      dim = index.params_.nav_dim;
      projected_.resize(dim);
      index.project_query(query, projected_.data());
      query = projected_.data();
    }
    weights_.resize(dim);
    float c = 0.0f;
    float qq = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
      weights_[i] = query[i] * index.sq8_scale_[i];
      c += query[i] * index.sq8_min_[i];
      qq += query[i] * query[i];
    }
    base_ = l2_ ? qq - 2.0f * c : 1.0f - c;
    if (index.sq8_int_dot_ == nullptr) return;
    // Integer kernel: the weights rounded to int8 on one scale per query, against distances the
    // beam only needs to rank (the final candidates are re-scored exactly). The scale follows the
    // largest weight, so a few outlier dimensions would leave every other weight with one or two
    // levels: OpenAI embeddings have such dimensions, and a query's largest weight there is ~300x
    // its median (measured: 15% error, recall capped at 0.93). The largest weights (up to 8, one
    // dimension in twelve on small vectors) are therefore kept exact, in float, and only the rest
    // are rounded: error 0.3% on those embeddings. They cost a few float operations per node, so
    // they are split off only when they dominate: when the largest weight is over 4x the largest
    // of the rest. That is every query on OpenAI and CLIP embeddings (ratio ~60 and ~7), and
    // almost none on SIFT, GIST or GloVe (median 1.3-2), whose plain rounding is already accurate.
    outliers_ = std::min<std::size_t>({kMaxOutliers, dim / 12, dim});
    std::array<std::uint32_t, kMaxOutliers> top{};
    for (std::size_t i = 0, n = 0; outliers_ > 0 && i < dim; ++i) {
      // Keep the indices of the `outliers_` largest |w| (insertion into a tiny sorted list).
      const float a = std::abs(weights_[i]);
      if (n < outliers_) {
        top[n++] = static_cast<std::uint32_t>(i);
      } else if (a > std::abs(weights_[top[outliers_ - 1]])) {
        top[outliers_ - 1] = static_cast<std::uint32_t>(i);
      } else {
        continue;
      }
      for (std::size_t j = n - 1; j > 0 && std::abs(weights_[top[j]]) > std::abs(weights_[top[j - 1]]); --j) std::swap(top[j], top[j - 1]);
    }
    // The largest |w| outside the top `outliers_`: zero those slots for the pass, then restore them.
    std::array<float, kMaxOutliers> kept{};
    for (std::size_t j = 0; j < outliers_; ++j) std::swap(kept[j], weights_[top[j]]);
    float largest = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) largest = std::max(largest, std::abs(weights_[i]));
    if (outliers_ > 0 && std::abs(kept[0]) <= kOutlierRatio * largest) {
      for (std::size_t j = 0; j < outliers_; ++j) std::swap(kept[j], weights_[top[j]]);
      largest = std::abs(weights_[top[0]]);
      outliers_ = 0;
    }
    // Unused slots keep weight 0 (dimension 0), so scoring always runs all kMaxOutliers slots.
    for (std::size_t j = 0; j < outliers_; ++j) {
      outlier_dims_[j] = top[j];
      outlier_weights_[j] = kept[j];
    }
    w_scale_ = largest > 0.0f ? largest / static_cast<float>(kSq8IntWeightMax) : 1.0f;
    const float inv = 1.0f / w_scale_;
    iweights_.resize(dim);
    for (std::size_t i = 0; i < dim; ++i) iweights_[i] = static_cast<std::int8_t>(std::lrint(weights_[i] * inv));  // outliers: 0
    for (std::size_t j = 0; j < outliers_; ++j) weights_[top[j]] = kept[j];
  }

  float operator()(id_t id) const noexcept {
    if (rbq_words_ != 0) {
      // RaBitQ: estimate <o, q> for the centred unit vectors, then the metric's distance.
      const std::byte* rec = index_.record(id);
      const auto& f = *reinterpret_cast<const rabitq::Factors*>(rec);
      const auto* bits = reinterpret_cast<const std::uint64_t*>(rec + sizeof(rabitq::Factors));
      const float est = rabitq::estimate(rbq_, index_.rabitq_ip_(bits, rbq_.planes.data(), rbq_words_), f);
      if (l2_) return f.norm * f.norm + rbq_.norm * rbq_.norm - 2.0f * f.norm * rbq_.norm * est;
      return 1.0f - (f.norm * rbq_.norm * est + f.dot_c + rbq_.dot_c - center_sq_);
    }
    if (adc_) {
      const std::uint8_t* codes = index_.codes_of(id);
      float dot;
      if (iweights_.empty()) {
        dot = index_.sq8_dot_(weights_.data(), codes, weights_.size());
      } else {
        dot = w_scale_ * static_cast<float>(index_.sq8_int_dot_(iweights_.data(), codes, iweights_.size()));
        if (outliers_ != 0) {
          // A fixed count with independent sums: unrolled, without a chain of dependent adds.
          float part[4] = {};
          for (std::size_t j = 0; j < kMaxOutliers; ++j) part[j % 4] += outlier_weights_[j] * static_cast<float>(codes[outlier_dims_[j]]);
          dot += (part[0] + part[1]) + (part[2] + part[3]);
        }
      }
      return l2_ ? base_ + index_.code_norm(id) - 2.0f * dot : base_ - dot;
    }
    return full(id);
  }

  // The distance of `id` if it could matter, given that only distances below `bound` do. RaBitQ:
  // the estimate's error bound decides; a node that cannot beat `bound` keeps its (lower-bounded)
  // estimate and costs no full-vector read, any other is scored exactly, so the beam holds exact
  // distances only. Other modes score as usual. `exact` reports which happened.
  float bounded(id_t id, float bound, bool& exact) const noexcept {
    exact = rbq_words_ == 0;
    if (rbq_words_ == 0) return (*this)(id);
    const std::byte* rec = index_.record(id);
    const auto& f = *reinterpret_cast<const rabitq::Factors*>(rec);
    const auto* bits = reinterpret_cast<const std::uint64_t*>(rec + sizeof(rabitq::Factors));
    const float est = rabitq::estimate(rbq_, index_.rabitq_ip_(bits, rbq_.planes.data(), rbq_words_), f);
    const float scale = (l2_ ? 2.0f : 1.0f) * f.norm * rbq_.norm;
    const float d = l2_ ? f.norm * f.norm + rbq_.norm * rbq_.norm - 2.0f * f.norm * rbq_.norm * est
                        : 1.0f - (f.norm * rbq_.norm * est + f.dot_c + rbq_.dot_c - center_sq_);
    if (d - scale * rabitq::error_bound(f, index_.rbq_padded_) > bound) return d;
    exact = true;
    return full(id);
  }
  bool bounded_mode() const noexcept { return rbq_words_ != 0; }
  // What the distance computation will read first, for prefetching.
  const void* address(id_t id) const noexcept {
    return adc_ ? static_cast<const void*>(index_.record(id)) : static_cast<const void*>(full_base_ + id * full_stride_);
  }

 private:
  const HnswIndex& index_;
  // The full-precision distance to node `id`.
  float full(id_t id) const noexcept {
    const std::byte* v = full_base_ + static_cast<std::size_t>(id) * full_stride_;
    if (half_full_) [[unlikely]] {
      return index_.dist_half_(query_, reinterpret_cast<const std::uint16_t*>(v), dim_);
    }
    return index_.dist_(query_, reinterpret_cast<const float*>(v), dim_);
  }

  const float* query_;
  bool adc_;
  const std::byte* full_base_ = nullptr;
  std::size_t full_stride_ = 0;
  std::size_t dim_ = 0;
  bool half_full_ = false;
  bool l2_ = false;
  float base_ = 0.0f;
  std::vector<float> projected_;  // nav_dim: the query in the walk's directions
  std::vector<float> weights_;
  std::vector<std::int8_t> iweights_;  // integer SQ8 (empty: the float kernel)
  float w_scale_ = 1.0f;
  // Integer SQ8: the largest weights, kept exact (their int8 slots are 0).
  static constexpr std::size_t kMaxOutliers = 8;
  static constexpr float kOutlierRatio = 4.0f;
  std::size_t outliers_ = 0;
  std::array<std::uint32_t, kMaxOutliers> outlier_dims_{};
  std::array<float, kMaxOutliers> outlier_weights_{};
  rabitq::Query rbq_;
  std::size_t rbq_words_ = 0;  // nonzero in RaBitQ mode
  float center_sq_ = 0.0f;
};

void HnswIndex::AlignedFree::operator()(void* p) const noexcept { platform::aligned_free(p); }

namespace {

// Integer SQ8 where a SIMD integer kernel exists. EPZERO_SQ8_KERNEL=float keeps the float kernel
// (to compare the two on the same build).
Sq8IntDotFn choose_sq8_int_dot(SimdLevel level, std::size_t dim) noexcept {
  if (level != SimdLevel::AVX2 && level != SimdLevel::AVX512) return nullptr;
  if (const char* forced = std::getenv("EPZERO_SQ8_KERNEL"); forced != nullptr && std::string_view(forced) == "float") {
    return nullptr;
  }
  return get_sq8_int_dot_fn(level, dim);
}

}  // namespace

HnswIndex::HnswIndex(const HnswParams& params)
    : params_(params),
      simd_level_(active_simd_level()),
      dist_(get_distance_fn(params.metric, simd_level_)),
      sq8_dot_(get_sq8_dot_fn(simd_level_)),
      sq8_int_dot_(choose_sq8_int_dot(simd_level_, params.nav_dim != 0 ? params.nav_dim : params.dim)),
      rabitq_ip_(get_rabitq_ip_fn(simd_level_)),
      max_links0_(2 * params.M),
      level_mult_(params.M > 1 ? 1.0 / std::log(static_cast<double>(params.M)) : 1.0),
      rng_(params.seed) {
  if (params_.dim == 0) throw std::invalid_argument("dim must be > 0");
  if (params_.max_elements == 0) throw std::invalid_argument("max_elements must be > 0");
  if (params_.max_elements >= std::numeric_limits<id_t>::max()) {
    throw std::invalid_argument("max_elements must fit in 32-bit node ids");
  }
  if (params_.M < 2) throw std::invalid_argument("M must be >= 2");
  if (rabitq() && !params_.keep_vectors) throw std::invalid_argument("RaBitQ keeps the full vectors to re-rank");
  rbq_padded_ = rabitq::padded_dim(params_.dim);
  walk_full_ = std::getenv("EPZERO_WALK_FULL") != nullptr;  // research: walk at full precision
  if (params_.fast_build && !sq8()) throw std::invalid_argument("fast_build needs SQ8");
  if (params_.half_vectors && !(sq8() && params_.keep_vectors)) {
    throw std::invalid_argument("half_vectors needs SQ8 with keep_vectors");
  }
  if (params_.nav_dim != 0) {
    if (!sq8() || !params_.keep_vectors) throw std::invalid_argument("nav_dim needs SQ8 with keep_vectors");
    if (params_.nav_dim >= params_.dim) throw std::invalid_argument("nav_dim must be less than dim");
    dot_ = get_distance_fn(Metric::InnerProduct, simd_level_);
  }

  // Record layout. None: [float x dim][links]. SQ8: [float norm][uint8 x dim][links].
  // RaBitQ: [factors][bits, one per padded dimension][links].
  links0_offset_ = rabitq() ? rabitq_record_bytes()
                   : sq8()  ? round_up(sizeof(float) + code_dim(), sizeof(id_t))
                            : params_.dim * sizeof(float);
  node_stride_ = round_up(links0_offset_ + (1 + max_links0_) * sizeof(id_t), 64);
  level0_ = allocate_level0();
  if (params_.half_vectors) {
    half_.reset(static_cast<std::uint16_t*>(allocate_block(params_.max_elements * params_.dim * sizeof(std::uint16_t))));
    dist_half_ = get_half_distance_fn(params_.metric, simd_level_);
  } else if (quantized() && params_.keep_vectors) {
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

std::size_t HnswIndex::rabitq_record_bytes() const noexcept {
  return sizeof(rabitq::Factors) + rabitq::words(rbq_padded_) * sizeof(std::uint64_t);
}

const float* HnswIndex::full_floats(id_t id, std::vector<float>& scratch) const {
  if (!half_) return full_vector(id);
  scratch.resize(params_.dim);
  const std::uint16_t* h = half_vector(id);
  for (std::size_t d = 0; d < params_.dim; ++d) scratch[d] = half::to_float(h[d]);
  return scratch.data();
}

const float* HnswIndex::full_vector(id_t id) const noexcept {
  if (!quantized()) return reinterpret_cast<const float*>(record(id));
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
  return platform::load_acquire(deleted_[id]) != 0;
}

// ---------------------------------------------------------------------------------------------
// Scalar quantization

void HnswIndex::train(const float* data, std::size_t n) {
  if (!quantized()) return;
  if (size() > 0) throw std::logic_error("train() must be called before the first add()");
  if (n == 0) throw std::invalid_argument("train() needs at least one vector");
  const std::size_t dim = params_.dim;
  if (rabitq()) {
    // The centroid of what gets stored, and the rotation's signs (fixed by the seed).
    std::vector<double> sum(dim, 0.0);
    std::vector<float> row(dim);
    for (std::size_t r = 0; r < n; ++r) {
      std::copy(data + r * dim, data + (r + 1) * dim, row.begin());
      if (params_.metric == Metric::Cosine) normalize(row.data(), dim);
      for (std::size_t i = 0; i < dim; ++i) sum[i] += row[i];
    }
    rbq_center_.resize(dim);
    for (std::size_t i = 0; i < dim; ++i) rbq_center_[i] = static_cast<float>(sum[i] / static_cast<double>(n));
    rbq_signs_ = rabitq::random_signs(rbq_padded_, params_.seed);
    return;
  }
  const auto stored = [&](std::size_t r, float* out) {  // train on what gets stored
    std::copy(data + r * dim, data + (r + 1) * dim, out);
    if (params_.metric == Metric::Cosine) normalize(out, dim);
  };
  if (params_.nav_dim != 0) {
    nav_basis_ = pca::top_directions(stored, n, dim, params_.nav_dim, params_.seed);
    prepare_query_projection();
  }
  // Each code's range over every row, in blocks of rows on all cores: with principal directions each
  // row is projected first, nav_dim x dim multiply-adds, which on one core took minutes at 1M rows.
  const std::size_t codes = code_dim();
  std::vector<float> lo(codes, std::numeric_limits<float>::max());
  std::vector<float> hi(codes, std::numeric_limits<float>::lowest());
  constexpr std::size_t kBlock = 4096;
  std::mutex merge;
  parallel_for(0, (n + kBlock - 1) / kBlock, 0, [&](std::size_t block) {
    std::vector<float> block_lo(codes, std::numeric_limits<float>::max());
    std::vector<float> block_hi(codes, std::numeric_limits<float>::lowest());
    std::vector<float> row(dim);
    std::vector<float> walked(codes);
    for (std::size_t r = block * kBlock; r < std::min(n, (block + 1) * kBlock); ++r) {
      stored(r, row.data());
      if (params_.nav_dim != 0) project(row.data(), walked.data());
      const float* v = params_.nav_dim != 0 ? walked.data() : row.data();
      for (std::size_t i = 0; i < codes; ++i) {
        block_lo[i] = std::min(block_lo[i], v[i]);
        block_hi[i] = std::max(block_hi[i], v[i]);
      }
    }
    std::lock_guard lock(merge);
    for (std::size_t i = 0; i < codes; ++i) {
      lo[i] = std::min(lo[i], block_lo[i]);
      hi[i] = std::max(hi[i], block_hi[i]);
    }
  });
  sq8_min_ = lo;
  sq8_scale_.resize(codes);
  for (std::size_t i = 0; i < codes; ++i) {
    const float range = hi[i] - lo[i];
    sq8_scale_[i] = range > 0.0f ? range / 255.0f : 1.0f;
  }
}

void HnswIndex::project(const float* v, float* out) const noexcept {
  const std::size_t dim = params_.dim;
  for (std::size_t r = 0; r < params_.nav_dim; ++r) out[r] = 1.0f - dot_(nav_basis_.data() + r * dim, v, dim);
}

// A query's projection is only a starting point for 8-bit weights, so it doesn't need float precision.
// Rounding the directions and the query to 8 bits runs it on the integer SQ8 kernel, which takes a
// quarter of the memory and far fewer instructions: at 256 directions of 1,536 dimensions the float
// version is about 28 us per query, a large share of a short search.
void HnswIndex::prepare_query_projection() {
  if (params_.nav_dim == 0 || nav_basis_.empty() || sq8_int_dot_ == nullptr) return;
  const std::size_t dim = params_.dim;
  nav_ibasis_.resize(params_.nav_dim * dim);
  nav_iscale_.resize(params_.nav_dim);
  nav_isum_.resize(params_.nav_dim);
  for (std::size_t r = 0; r < params_.nav_dim; ++r) {
    const float* row = nav_basis_.data() + r * dim;
    float largest = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) largest = std::max(largest, std::abs(row[d]));
    const float scale = largest > 0.0f ? static_cast<float>(kSq8IntWeightMax) / largest : 1.0f;
    std::int64_t sum = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      const auto w = static_cast<std::int8_t>(std::lrint(row[d] * scale));
      nav_ibasis_[r * dim + d] = w;
      sum += w;
    }
    nav_iscale_[r] = 1.0f / scale;
    nav_isum_[r] = sum;
  }
}

void HnswIndex::project_query(const float* query, float* out) const {
  if (nav_ibasis_.empty()) {
    project(query, out);
    return;
  }
  const std::size_t dim = params_.dim;
  float largest = 0.0f;
  for (std::size_t d = 0; d < dim; ++d) largest = std::max(largest, std::abs(query[d]));
  const float to_code = largest > 0.0f ? 127.0f / largest : 1.0f;
  thread_local std::vector<std::uint8_t> codes;
  codes.resize(dim);
  for (std::size_t d = 0; d < dim; ++d) codes[d] = static_cast<std::uint8_t>(std::lrint(query[d] * to_code) + 128);
  const float from_code = 1.0f / to_code;
  for (std::size_t r = 0; r < params_.nav_dim; ++r) {
    const std::int64_t dot = sq8_int_dot_(nav_ibasis_.data() + r * dim, codes.data(), dim) - 128 * nav_isum_[r];
    out[r] = static_cast<float>(dot) * nav_iscale_[r] * from_code;
  }
}

void HnswIndex::encode(const float* v, std::uint8_t* codes, float* norm) const noexcept {
  thread_local std::vector<float> walked;
  if (params_.nav_dim != 0) {
    walked.resize(params_.nav_dim);
    project(v, walked.data());
    v = walked.data();
  }
  float sum = 0.0f;
  for (std::size_t i = 0; i < code_dim(); ++i) {
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
  for (std::size_t i = 0; i < code_dim(); ++i) out[i] = sq8_min_[i] + codes[i] * sq8_scale_[i];
}

float HnswIndex::pair_distance(id_t a, id_t b) const noexcept {
  // Building on nav_dim codes prunes on them too, as a scan of dim floats is the cost it avoids. With a
  // code per dimension it doesn't: decoding two nodes costs more than reading their vectors (2-3× the
  // build time on GloVe-100, SIFT1M and DBpedia, at the same recall).
  if (has_full_vectors() && (!params_.fast_build || build_exact_ || params_.nav_dim == 0)) {
    if (!half_) return dist_(full_vector(a), full_vector(b), params_.dim);
    thread_local std::vector<float> scratch;
    return full_distance(full_floats(a, scratch), b);
  }
  thread_local std::vector<float> va, vb;
  va.resize(code_dim());
  vb.resize(code_dim());
  decode(a, va.data());
  decode(b, vb.data());
  return dist_(va.data(), vb.data(), code_dim());
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
      // Upper-layer nodes are scattered through memory: request them all before scoring any, so
      // their cache misses overlap instead of being paid one after another.
      for (id_t i = 0; i < count; ++i) prefetch_lines(scorer.address(load_link(links[1 + i])), prefetch_lines_);
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

void HnswIndex::set_prefetch(std::size_t depth, std::size_t lines) noexcept {
  prefetch_depth_ = depth;
  prefetch_lines_ = lines;
}

std::pair<std::size_t, std::size_t> HnswIndex::tune_prefetch(const float* queries, std::size_t n, std::size_t k,
                                                             std::size_t ef) {
  constexpr std::size_t kAll = std::numeric_limits<std::size_t>::max();
  if (n == 0 || size() == 0) return prefetch_settings();
  // What one scored node reads: its codes (quantized) or its full vector.
  const std::size_t bytes = sq8() ? code_dim() + sizeof(float) : rabitq() ? rabitq_record_bytes() : params_.dim * sizeof(float);
  const std::size_t max_lines = std::max<std::size_t>(1, (bytes + 63) / 64);
  std::vector<std::pair<std::size_t, std::size_t>> grid = {{kAll, 1}};  // the default first
  for (const std::size_t depth : {kAll, std::size_t{16}, std::size_t{8}, std::size_t{4}, std::size_t{2}}) {
    // Always every line of a node: a record of 3, 5 or 7 lines must be able to prefetch all of them.
    for (const std::size_t lines : {std::size_t{1}, std::size_t{2}, std::size_t{4}, max_lines}) {
      if (lines > max_lines || (depth == kAll && lines == 1)) continue;
      const std::pair setting{depth, lines};
      if (std::find(grid.begin(), grid.end(), setting) == grid.end()) grid.push_back(setting);
    }
  }
  const auto time_pass = [&] {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t q = 0; q < n; ++q) search(queries + q * params_.dim, k, {.ef = ef});
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  };
  time_pass();  // warm the caches
  std::vector<double> best(grid.size(), std::numeric_limits<double>::infinity());
  for (int round = 0; round < 3; ++round) {  // interleaved, so drift affects every setting alike
    for (std::size_t g = 0; g < grid.size(); ++g) {
      set_prefetch(grid[g].first, grid[g].second);
      best[g] = std::min(best[g], time_pass());
    }
  }
  std::size_t pick = 0;
  for (std::size_t g = 1; g < grid.size(); ++g) {
    if (best[g] < best[pick]) pick = g;
  }
  if (best[pick] > 0.97 * best[0]) pick = 0;  // within 3% of the default: keep the default
  set_prefetch(grid[pick].first, grid[pick].second);
  return grid[pick];
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(const Scorer& scorer, id_t entry, std::size_t ef,
                                                          int layer, SearchStats& stats, SearchTrace* trace,
                                                          bool keep_deleted, std::size_t k, std::size_t patience) const {
  // Tombstoned nodes are explored (they keep the graph connected) but never kept as results, so
  // with deletions the beam must fill up with live nodes before it may stop: that bookkeeping
  // lives in search_layer_tombstones.
  if (!keep_deleted && deleted_count_.load(std::memory_order_relaxed) > 0) {
    return search_layer_tombstones(scorer, entry, ef, layer, stats, trace, k, patience);
  }
  constexpr float kInfinity = std::numeric_limits<float>::infinity();
  auto visited = visited_pool_->acquire();
  Beam beam(ef);

  bool exact = true;
  const float entry_dist = scorer.bounded(entry, kInfinity, exact);
  ++stats.distance_computations;
  stats.full_distance_computations += exact;
  visited->test_and_set(entry);
  beam.insert(entry_dist, entry);

  // Patience: how many expansions in a row have left the best k unchanged (an insert into the
  // first k places changes them).
  std::size_t unchanged = 0;
  std::vector<id_t> batch(max_links0_);  // one node's unvisited neighbors (layer 0's lists are the longest)

  // Expand the closest unexpanded node until every node in the beam has been expanded, which is
  // when the closest remaining candidate is farther than the beam's worst.
  while (beam.has_next()) {
    if (patience > 0 && unchanged >= patience) break;
    const id_t current = beam.next();
    bool improved = false;

    const id_t* links = links_of(current, layer);
    const id_t count = load_link(links[0]);
    ++stats.nodes_expanded;

    // Visited neighbors are dropped first and the rest prefetched, as in search_layer_tombstones.
    std::size_t pending = 0;
    for (id_t i = 0; i < count; ++i) {
      const id_t neighbor = load_link(links[1 + i]);
      if (visited->test_and_set(neighbor)) continue;
      batch[pending++] = neighbor;
      if (pending <= prefetch_depth_) prefetch_lines(scorer.address(neighbor), prefetch_lines_);
    }

    for (std::size_t j = 0; j < pending; ++j) {
      if (pending - j > prefetch_depth_) prefetch_lines(scorer.address(batch[j + prefetch_depth_]), prefetch_lines_);
      const id_t neighbor = batch[j];
      const float d = scorer.bounded(neighbor, beam.bound(), exact);
      ++stats.distance_computations;
      stats.full_distance_computations += exact;
      const std::size_t at = beam.insert(d, neighbor);
      if (trace != nullptr) {
        trace->events.push_back({static_cast<std::uint32_t>(layer), labels_[current], labels_[neighbor], d, at != Beam::kRejected});
      }
      improved |= at < k;
    }
    unchanged = improved ? 0 : unchanged + 1;
  }

  std::vector<Candidate> result(beam.size());
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = {beam[i].distance, beam[i].id};
  return result;  // closest first
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer_tombstones(const Scorer& scorer, id_t entry, std::size_t ef,
                                                                     int layer, SearchStats& stats, SearchTrace* trace,
                                                                     std::size_t k, std::size_t patience) const {
  auto visited = visited_pool_->acquire();
  std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst> candidates;  // closest on top
  std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;       // farthest on top
  const bool has_deletions = true;
  constexpr float kInfinity = std::numeric_limits<float>::infinity();

  bool exact = true;
  const float entry_dist = scorer.bounded(entry, kInfinity, exact);
  ++stats.distance_computations;
  stats.full_distance_computations += exact;
  visited->test_and_set(entry);
  candidates.push({entry_dist, entry});
  if (!has_deletions || !is_deleted(entry)) best.push({entry_dist, entry});
  float bound = best.empty() ? kInfinity : best.top().distance;

  // Patience: the best k so far (a small heap beside the beam), and how many expansions in a row
  // have left them unchanged.
  std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> top;
  float kth = kInfinity;
  std::size_t unchanged = 0;
  const auto offer_top = [&](float d, id_t id) {
    if (top.size() == k && d >= kth) return false;
    top.push({d, id});
    if (top.size() > k) top.pop();
    kth = top.size() == k ? top.top().distance : kInfinity;
    return true;
  };
  if (patience > 0 && !best.empty()) offer_top(entry_dist, entry);

  // Scratch space for one node's unvisited neighbors; the largest list is layer 0's.
  std::vector<id_t> batch(max_links0_);

  while (!candidates.empty()) {
    const Candidate current = candidates.top();
    // Every remaining candidate is farther than the worst result we keep: the beam has converged.
    if (current.distance > bound && (best.size() >= ef || !has_deletions)) break;
    // Or the answer has settled: the best k survived `patience` expansions in a row.
    if (patience > 0 && unchanged >= patience) break;
    candidates.pop();
    bool improved = false;

    const id_t* links = links_of(current.id, layer);
    const id_t count = load_link(links[0]);
    ++stats.nodes_expanded;

    // Filter out visited neighbors first and prefetch the survivors' vectors, so their cache
    // misses overlap instead of being paid one after another inside the distance loop.
    // Measured on SIFT1M: +12-18% QPS. By default every survivor's first line is prefetched up
    // front (the hardware stream prefetcher follows once the kernel reads); set_prefetch() can
    // instead keep a window of `depth` survivors ahead of the one being scored, `lines` lines each.
    std::size_t pending = 0;
    for (id_t i = 0; i < count; ++i) {
      const id_t neighbor = load_link(links[1 + i]);
      if (visited->test_and_set(neighbor)) continue;
      batch[pending++] = neighbor;
      if (pending <= prefetch_depth_) prefetch_lines(scorer.address(neighbor), prefetch_lines_);
    }

    for (std::size_t j = 0; j < pending; ++j) {
      if (pending - j > prefetch_depth_) prefetch_lines(scorer.address(batch[j + prefetch_depth_]), prefetch_lines_);
      const id_t neighbor = batch[j];
      const float d = scorer.bounded(neighbor, best.size() < ef ? kInfinity : bound, exact);
      ++stats.distance_computations;
      stats.full_distance_computations += exact;
      const bool accepted = best.size() < ef || d < bound;
      if (trace != nullptr) {
        trace->events.push_back({static_cast<std::uint32_t>(layer), labels_[current.id], labels_[neighbor], d, accepted});
      }
      if (accepted) {
        candidates.push({d, neighbor});
        if (!has_deletions || !is_deleted(neighbor)) {
          best.push({d, neighbor});
          if (best.size() > ef) best.pop();
          if (patience > 0) improved |= offer_top(d, neighbor);
        }
        bound = best.empty() ? kInfinity : best.top().distance;
      }
    }
    // Expanding a tombstone only routes through it (it can never improve the answer), so it doesn't
    // count against patience: otherwise a walk through a churned region would stop too soon.
    if (improved) {
      unchanged = 0;
    } else if (!has_deletions || !is_deleted(current.id)) {
      ++unchanged;
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

std::vector<SearchResult> HnswIndex::exact_search(const float* query, std::size_t k, const SearchFilter* filter) const {
  if (k == 0) return {};
  std::vector<float> normalized;
  if (params_.metric == Metric::Cosine) {
    normalized.assign(query, query + params_.dim);
    normalize(normalized.data(), params_.dim);
    query = normalized.data();
  }
  const Scorer scorer(*this, query, /*adc=*/!has_full_vectors());
  const bool has_deletions = deleted_count_.load(std::memory_order_relaxed) > 0;
  std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;  // farthest on top
  const auto consider = [&](id_t id) {
    if (has_deletions && is_deleted(id)) return;
    const float d = scorer(id);
    if (best.size() < k) {
      best.push({d, id});
    } else if (d < best.top().distance) {
      best.pop();
      best.push({d, id});
    }
  };
  if (filter != nullptr) {
    for (const id_t id : filter->ids_) consider(id);
  } else {
    const id_t n = static_cast<id_t>(size());
    for (id_t id = 0; id < n; ++id) consider(id);
  }
  std::vector<SearchResult> results(best.size());
  for (std::size_t i = results.size(); i-- > 0;) {
    results[i] = {labels_[best.top().id], best.top().distance};
    best.pop();
  }
  return results;
}

std::string_view to_string(MissExplanation::Cause cause) noexcept {
  switch (cause) {
    case MissExplanation::Cause::Returned: return "returned";
    case MissExplanation::Cause::NotInIndex: return "not-in-index";
    case MissExplanation::Cause::FilteredOut: return "filtered-out";
    case MissExplanation::Cause::NotNearEnough: return "not-near-enough";
    case MissExplanation::Cause::SearchMissed: return "search-missed";
  }
  return "unknown";
}

MissExplanation HnswIndex::explain(const float* query, std::size_t k, std::uint64_t label,
                                   const SearchOptions& options) const {
  using Cause = MissExplanation::Cause;
  MissExplanation out;
  id_t target;
  {
    std::lock_guard lock(label_lock_);
    const auto it = label_to_id_.find(label);
    if (it == label_to_id_.end()) return out;  // NotInIndex
    target = it->second;
  }
  const SearchFilter* filter = options.filter;
  if (filter != nullptr && !filter->allows(target)) {
    out.cause = Cause::FilteredOut;
    return out;
  }

  // Its exact distance, and how many live (allowed) vectors are strictly closer.
  std::vector<float> normalized;
  const float* q = query;
  if (params_.metric == Metric::Cosine) {
    normalized.assign(query, query + params_.dim);
    normalize(normalized.data(), params_.dim);
    q = normalized.data();
  }
  const Scorer exact(*this, q, /*adc=*/!has_full_vectors());
  out.distance = exact(target);
  const bool has_deletions = deleted_count_.load(std::memory_order_relaxed) > 0;
  std::size_t closer = 0;
  const auto count = [&](id_t id) {
    if (id != target && !(has_deletions && is_deleted(id)) && exact(id) < out.distance) ++closer;
  };
  if (filter != nullptr) {
    for (const id_t id : filter->ids_) count(id);
  } else {
    for (id_t id = 0; id < static_cast<id_t>(size()); ++id) count(id);
  }
  out.exact_rank = closer + 1;

  // The search as it ran, traced to see whether the walk ever scored it.
  SearchTrace trace;
  SearchOptions traced = options;
  traced.trace = &trace;
  traced.stats = nullptr;
  const auto results = search(query, k, traced);
  if (!results.empty()) out.kth_distance = results.back().distance;
  const auto found = [&](const std::vector<SearchResult>& r) {
    return std::any_of(r.begin(), r.end(), [&](const SearchResult& hit) { return hit.label == label; });
  };
  if (found(results)) {
    out.cause = Cause::Returned;
    return out;
  }
  if (out.exact_rank > k) {
    out.cause = Cause::NotNearEnough;
    return out;
  }
  out.cause = Cause::SearchMissed;
  out.scored = std::any_of(trace.events.begin(), trace.events.end(), [&](const TraceEvent& e) { return e.to == label; });
  SearchOptions wider = options;
  wider.trace = nullptr;
  wider.stats = nullptr;
  for (std::size_t ef = std::max<std::size_t>(options.ef, k) * 2; ef <= 4096; ef *= 2) {
    wider.ef = ef;
    if (found(search(query, k, wider))) {
      out.ef_needed = ef;
      break;
    }
  }
  return out;
}

FilterStrategy HnswIndex::choose_strategy(const SearchFilter& filter, std::size_t ef) const noexcept {
  const double selectivity = filter.selectivity();
  if (filter.allowed() <= kExactPerEf * ef) return FilterStrategy::Exact;
  if (selectivity >= kInlineSelectivity) return FilterStrategy::Inline;
  if (selectivity >= kTwoHopMinSelectivity) return FilterStrategy::TwoHop;
  const double inline_cost = kInlineScale * std::pow(static_cast<double>(ef) / selectivity, kInlineExponent);
  return kScanCostRatio * static_cast<double>(filter.allowed()) <= inline_cost ? FilterStrategy::Exact
                                                                               : FilterStrategy::Inline;
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
  std::vector<Candidate> pruned;
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
    } else if (params_.fill_links) {
      pruned.push_back(candidate);
    }
  }
  for (std::size_t i = 0; i < pruned.size() && kept.size() < max_count; ++i) kept.push_back(pruned[i]);
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
    if (std::find(existing.begin(), existing.end(), id) != existing.end()) continue;  // refine_links revisits
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
  if (!trained()) throw std::logic_error("a quantized index must be trained before add()");
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
  if (rabitq()) {
    std::vector<float> scratch(rbq_padded_);
    auto* factors = reinterpret_cast<rabitq::Factors*>(record(id));
    auto* bits = reinterpret_cast<std::uint64_t*>(record(id) + sizeof(rabitq::Factors));
    rabitq::encode(vector, params_.dim, rbq_center_.data(), rbq_signs_.data(), rbq_padded_, bits, *factors, scratch.data());
    std::memcpy(full_.get() + static_cast<std::size_t>(id) * params_.dim, vector, params_.dim * sizeof(float));
  } else if (sq8()) {
    auto* codes = reinterpret_cast<std::uint8_t*>(record(id) + sizeof(float));
    encode(vector, codes, reinterpret_cast<float*>(record(id)));
    if (half_) {
      std::uint16_t* h = half_.get() + static_cast<std::size_t>(id) * params_.dim;
      for (std::size_t d = 0; d < params_.dim; ++d) h[d] = half::from_float(vector[d]);
    } else if (full_) {
      std::memcpy(full_.get() + static_cast<std::size_t>(id) * params_.dim, vector, params_.dim * sizeof(float));
    }
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
  const Scorer scorer(*this, vector, /*adc=*/!has_full_vectors() || params_.fast_build);
  id_t entry_point = static_cast<id_t>(entry & 0xffffffffu);
  const int top_level = static_cast<int>(entry >> 32) - 1;
  SearchStats unused;
  if (level < top_level) entry_point = greedy_descend(scorer, entry_point, top_level, level, unused, nullptr);

  for (int layer = std::min(level, top_level); layer >= 0; --layer) {
    std::vector<Candidate> candidates =
        search_layer(scorer, entry_point, params_.ef_construction, layer, unused, nullptr);
    // A concurrent insert may have linked to us already, making us reachable from our own search.
    std::erase_if(candidates, [id](const Candidate& c) { return c.id == id; });
    if (candidates.empty() && deleted_count_.load(std::memory_order_relaxed) > 0) {
      // Every node the search reached is deleted (e.g. all of them were): new nodes normally don't
      // link to tombstones, but with no live neighbour this one would be unreachable. Link through
      // the tombstones instead; they keep routing searches until the index is compacted.
      candidates = search_layer(scorer, entry_point, params_.ef_construction, layer, unused, nullptr, true);
      std::erase_if(candidates, [id](const Candidate& c) { return c.id == id; });
    }
    if (candidates.empty()) continue;
    entry_point = candidates.front().id;
    connect(id, select_neighbors(std::move(candidates), params_.M), layer);
  }

  if (level > top_level) {
    std::lock_guard lock(entry_lock_);
    if (level > max_level()) entry_.store(pack_entry(id, level), std::memory_order_release);
  }
}

void HnswIndex::reserve(std::size_t max_elements) {
  if (max_elements <= params_.max_elements) return;
  if (max_elements >= std::numeric_limits<id_t>::max()) throw std::invalid_argument("max_elements must fit in 32-bit node ids");
  const std::size_t used = size();
  // Each block is allocated zeroed (an unused node has no links) and the used part copied over.
  std::unique_ptr<std::byte[], AlignedFree> level0(static_cast<std::byte*>(allocate_block(node_stride_ * max_elements)));
  std::memcpy(level0.get(), level0_.get(), node_stride_ * used);
  level0_ = std::move(level0);
  if (full_) {
    std::unique_ptr<float[], AlignedFree> full(static_cast<float*>(allocate_block(max_elements * params_.dim * sizeof(float))));
    std::memcpy(full.get(), full_.get(), used * params_.dim * sizeof(float));
    full_ = std::move(full);
  }
  if (half_) {
    std::unique_ptr<std::uint16_t[], AlignedFree> half(
        static_cast<std::uint16_t*>(allocate_block(max_elements * params_.dim * sizeof(std::uint16_t))));
    std::memcpy(half.get(), half_.get(), used * params_.dim * sizeof(std::uint16_t));
    half_ = std::move(half);
  }
  upper_links_.resize(max_elements);
  levels_.resize(max_elements, 0);
  labels_.resize(max_elements, 0);
  auto deleted = std::make_unique<std::uint8_t[]>(max_elements);
  std::memcpy(deleted.get(), deleted_.get(), used);
  deleted_ = std::move(deleted);
  node_locks_ = std::make_unique<std::mutex[]>(max_elements);
  visited_pool_ = std::make_unique<detail::VisitedPool>(max_elements);
  label_to_id_.reserve(max_elements);
  params_.max_elements = max_elements;
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
  platform::store_release(deleted_[id], std::uint8_t{1});
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
  const Scorer scorer(*this, query, /*adc=*/quantized() && !walk_full_);
  const std::size_t ef = std::max(options.ef, k);
  std::vector<Candidate> candidates;
  if (options.filter == nullptr) {
    const id_t start = greedy_descend(scorer, entry_point, top_level, 0, stats, trace);
    candidates = search_layer(scorer, start, ef, 0, stats, trace, false, k, options.patience);
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
    // An exact scan needs only the k best, unless quantized distances are re-ranked below.
    if (strategy == FilterStrategy::Exact) candidates = scan_allowed(scorer, filter, quantized() ? ef : k, stats);
    stats.strategy = strategy;
  }

  // Quantized: the beam ran on approximate distances; re-score the best candidates exactly.
  // (RaBitQ's unfiltered beam already holds exact distances, see Scorer::bounded.)
  const bool beam_exact = scorer.bounded_mode() && options.filter == nullptr;
  if (quantized() && (full_ || half_) && !candidates.empty() && !beam_exact) {
    std::size_t rerank = options.rerank == 0 ? 2 * k : std::max(options.rerank, k);
    rerank = std::min(rerank, candidates.size());
    if (options.rerank_z > 0.0f && rerank > k + kRerankBlock) {
      // In blocks, stopping when the rest can't reach the top k: the gap between estimated and exact
      // distances, learned from this query's own candidates, says where the next one would land.
      double gap = 0.0;
      double gap_sq = 0.0;
      std::vector<float> best;
      best.reserve(k + 1);
      std::size_t scored = 0;
      while (scored < rerank) {
        const std::size_t end = std::min(rerank, scored + kRerankBlock);
        for (std::size_t i = scored; i < end; ++i) prefetch(full_address(candidates[i].id));
        for (std::size_t i = scored; i < end; ++i) {
          const float estimate = candidates[i].distance;
          const float d = full_distance(query, candidates[i].id);
          candidates[i].distance = d;
          const double g = static_cast<double>(d) - estimate;
          gap += g;
          gap_sq += g * g;
          best.insert(std::upper_bound(best.begin(), best.end(), d), d);
          if (best.size() > k) best.pop_back();
        }
        scored = end;
        if (scored >= rerank || best.size() < k) continue;
        const double n = static_cast<double>(scored);
        const double mean = gap / n;
        const double spread = std::sqrt(std::max(0.0, gap_sq / n - mean * mean));
        if (candidates[scored].distance + mean - options.rerank_z * spread > best.back()) break;
      }
      rerank = scored;
    } else {
      for (std::size_t i = 0; i < rerank; ++i) prefetch(full_address(candidates[i].id));
      for (std::size_t i = 0; i < rerank; ++i) candidates[i].distance = full_distance(query, candidates[i].id);
    }
    stats.distance_computations += rerank;
    stats.full_distance_computations += rerank;
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
  out->nav_basis_ = nav_basis_;
  out->prepare_query_projection();
  out->rbq_center_ = rbq_center_;
  out->rbq_signs_ = rbq_signs_;
  std::vector<id_t> live;
  live.reserve(live_count());
  for (id_t id = 0; id < size(); ++id) {
    if (!is_deleted(id)) live.push_back(id);
  }
  parallel_for(0, live.size(), threads, [&](std::size_t i) {
    const id_t id = live[i];
    if (has_full_vectors()) {
      thread_local std::vector<float> scratch;
      out->add(labels_[id], full_floats(id, scratch));
    } else {
      std::vector<float> v(params_.dim);
      decode(id, v.data());
      out->add(labels_[id], v.data());
    }
  });
  // The same configuration: the build passes the profile records, its settings and prefetching.
  if (profile_.refined) out->refine_links(threads);
  if (profile_.filled) out->fill_free_links(threads);
  out->profile_ = profile_;
  out->set_prefetch(prefetch_depth_, prefetch_lines_);
  return out;
}

void HnswIndex::refine_links(std::size_t threads, std::size_t ef, bool exact) {
  build_exact_ = exact;
  if (ef == 0) ef = params_.ef_construction;
  const std::uint64_t entry = entry_.load(std::memory_order_acquire);
  if (entry == kEmpty) return;
  const int top_level = static_cast<int>(entry >> 32) - 1;
  parallel_for(0, size(), threads, [&](std::size_t i) {
    const auto id = static_cast<id_t>(i);
    if (is_deleted(id)) return;
    thread_local std::vector<float> vector;
    const float* v = nullptr;
    if (has_full_vectors()) {
      v = full_floats(id, vector);
    } else {
      vector.resize(params_.dim);
      decode(id, vector.data());
      v = vector.data();
    }
    const Scorer scorer(*this, v, /*adc=*/!has_full_vectors() || (params_.fast_build && !exact));
    SearchStats unused;
    id_t from = static_cast<id_t>(entry & 0xffffffffu);
    if (top_level > 0) from = greedy_descend(scorer, from, top_level, 0, unused, nullptr);
    std::vector<Candidate> candidates = search_layer(scorer, from, ef, 0, unused, nullptr);
    std::erase_if(candidates, [id](const Candidate& c) { return c.id == id; });
    if (!candidates.empty()) connect(id, select_neighbors(std::move(candidates), max_links0_), 0);
  });
  build_exact_ = false;
}

std::size_t HnswIndex::fill_free_links(std::size_t threads) {
  std::atomic<std::size_t> added{0};
  parallel_for(0, size(), threads, [&](std::size_t i) {
    const auto id = static_cast<id_t>(i);
    if (is_deleted(id)) return;
    std::vector<id_t> links;
    std::vector<id_t> hop;
    copy_links(id, 0, links);
    if (links.size() >= max_links0_) return;
    std::vector<id_t> pool;
    for (const id_t neighbor : links) {
      copy_links(neighbor, 0, hop);
      pool.insert(pool.end(), hop.begin(), hop.end());
    }
    std::sort(pool.begin(), pool.end());
    pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
    std::vector<Candidate> scored;
    scored.reserve(pool.size());
    for (const id_t other : pool) {
      if (other == id || is_deleted(other) || std::find(links.begin(), links.end(), other) != links.end()) continue;
      scored.push_back({pair_distance(id, other), other});
    }
    const std::size_t free_slots = std::min(max_links0_ - links.size(), scored.size());
    std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(free_slots), scored.end(),
                      [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });
    // Appended behind the existing links, so readers always see a valid list.
    id_t* slots = links_of(id, 0);
    for (std::size_t j = 0; j < free_slots; ++j) store_link(slots[1 + links.size() + j], scored[j].id);
    store_link(slots[0], static_cast<id_t>(links.size() + free_slots));
    added += free_slots;
  });
  return added;
}

bool HnswIndex::get_vector(std::uint64_t label, float* out) const {
  id_t id;
  {
    std::lock_guard lock(label_lock_);
    const auto it = label_to_id_.find(label);
    if (it == label_to_id_.end()) return false;
    id = it->second;
  }
  if (has_full_vectors()) {
    std::vector<float> scratch;
    std::memcpy(out, full_floats(id, scratch), params_.dim * sizeof(float));
  } else {
    decode(id, out);  // SQ8 without kept vectors: one code per dimension (nav_dim requires the vectors)
  }
  return true;
}

std::vector<std::uint64_t> HnswIndex::live_labels() const {
  std::lock_guard lock(label_lock_);
  std::vector<std::uint64_t> out;
  out.reserve(label_to_id_.size());
  for (const auto& [label, id] : label_to_id_) out.push_back(label);
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
  const bool rbq = params.quantization == Quantization::RaBitQ;
  const std::size_t links0 = rbq ? sizeof(rabitq::Factors) + rabitq::padded_dim(params.dim) / 8
                             : sq8 ? round_up(sizeof(float) + (params.nav_dim != 0 ? params.nav_dim : params.dim), sizeof(id_t))
                                   : params.dim * sizeof(float);
  const std::size_t stride = round_up(links0 + (1 + 2 * params.M) * sizeof(id_t), 64);
  std::size_t bytes = stride * params.max_elements;
  if ((sq8 || rbq) && params.keep_vectors) {
    bytes += params.max_elements * params.dim * (params.half_vectors ? sizeof(std::uint16_t) : sizeof(float));
  }
  return bytes;
}

std::size_t HnswIndex::memory_bytes() const noexcept {
  std::size_t bytes = node_stride_ * params_.max_elements;
  if (full_) bytes += params_.max_elements * params_.dim * sizeof(float);
  if (half_) bytes += params_.max_elements * params_.dim * sizeof(std::uint16_t);
  bytes += nav_basis_.size() * sizeof(float);
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
  const bool with_profile = !is_default(profile_) || params_.half_vectors;
  out.pod(with_profile ? kFormatVersionProfile : params_.nav_dim != 0 ? kFormatVersionNav : kFormatVersion);
  out.pod<std::uint64_t>(params_.dim);
  out.pod<std::uint32_t>(static_cast<std::uint32_t>(params_.metric));
  out.pod<std::uint64_t>(params_.max_elements);
  out.pod<std::uint64_t>(params_.M);
  out.pod<std::uint64_t>(params_.ef_construction);
  out.pod<std::uint64_t>(params_.seed);
  out.pod<std::uint32_t>(static_cast<std::uint32_t>(params_.quantization));
  out.pod<std::uint8_t>(params_.keep_vectors ? 1 : 0);
  if (params_.nav_dim != 0 || with_profile) out.pod<std::uint64_t>(params_.nav_dim);
  if (with_profile) {
    const auto string = [&](const std::string& text) {
      const std::size_t length = std::min(text.size(), kMaxProfileString);
      out.pod<std::uint32_t>(static_cast<std::uint32_t>(length));
      out.write(text.data(), length);
    };
    const IndexProfile& p = profile_;
    out.pod<std::uint8_t>(params_.half_vectors ? 1 : 0);  // storage flags
    out.pod<std::uint64_t>(p.k);
    out.pod<std::uint64_t>(p.ef);
    out.pod<std::uint64_t>(p.rerank);
    out.pod<float>(p.rerank_z);
    out.pod<float>(p.recall_target);
    out.pod<float>(p.recall_measured);
    out.pod<std::uint64_t>(p.calibrated_size);
    out.pod<float>(p.nav_variance);
    out.pod<std::uint8_t>(static_cast<std::uint8_t>((p.refined ? 1 : 0) | (p.built_on_codes ? 2 : 0)));
    out.pod<std::uint8_t>(p.filled ? 1 : 0);
    out.pod<std::uint64_t>(p.prefetch_depth);
    out.pod<std::uint64_t>(p.prefetch_lines);
    string(p.machine);
    const std::size_t notes = std::min<std::size_t>(p.notes.size(), 256);
    out.pod<std::uint32_t>(static_cast<std::uint32_t>(notes));
    for (std::size_t i = 0; i < notes; ++i) {
      string(p.notes[i].first);
      string(p.notes[i].second);
    }
  }
  const std::size_t count = size();
  out.pod<std::uint64_t>(count);
  out.pod<std::uint64_t>(entry_.load(std::memory_order_acquire));
  out.pod<std::uint64_t>(deleted_count());
  if (sq8()) {
    out.pod<std::uint8_t>(sq8_scale_.empty() ? 0 : 1);
    if (!sq8_scale_.empty()) {
      out.array(sq8_min_.data(), code_dim());
      out.array(sq8_scale_.data(), code_dim());
      if (params_.nav_dim != 0) out.array(nav_basis_.data(), nav_basis_.size());
    }
  }
  if (rabitq()) {
    out.pod<std::uint8_t>(rbq_center_.empty() ? 0 : 1);
    if (!rbq_center_.empty()) {
      out.array(rbq_center_.data(), params_.dim);
      out.array(rbq_signs_.data(), rbq_signs_.size());
    }
  }

  for (id_t id = 0; id < count; ++id) {
    std::lock_guard lock(node_locks_[id]);
    if (rabitq()) {
      out.array(reinterpret_cast<const std::uint8_t*>(record(id)), rabitq_record_bytes());
    } else if (sq8()) {
      out.pod(code_norm(id));
      out.array(codes_of(id), code_dim());
    } else {
      out.array(full_vector(id), params_.dim);
    }
    out.array(links_of(id, 0), 1 + max_links0_);
    out.pod(labels_[id]);
    out.pod(levels_[id]);
    out.pod<std::uint8_t>(is_deleted(id) ? 1 : 0);
    if (levels_[id] > 0) out.array(upper_links_[id].get(), static_cast<std::size_t>(levels_[id]) * (1 + params_.M));
    if (full_) out.array(full_vector(id), params_.dim);
    if (half_) out.array(half_vector(id), params_.dim);
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
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 && std::memcmp(magic, kLegacyMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error(path + " is not an Epzero index");
  }
  const auto version = in.pod<std::uint32_t>();
  if (version < 1 || version > kFormatVersionProfile) throw std::runtime_error(path + ": unsupported format version");

  HnswParams params;
  params.dim = in.pod<std::uint64_t>();
  params.metric = static_cast<Metric>(in.pod<std::uint32_t>());
  params.max_elements = in.pod<std::uint64_t>();
  params.M = in.pod<std::uint64_t>();
  params.ef_construction = in.pod<std::uint64_t>();
  params.seed = in.pod<std::uint64_t>();
  if (version >= 2) {
    const auto quantization = in.pod<std::uint32_t>();
    if (quantization > static_cast<std::uint32_t>(Quantization::RaBitQ)) throw std::runtime_error(path + ": bad quantization");
    params.quantization = static_cast<Quantization>(quantization);
    params.keep_vectors = in.pod<std::uint8_t>() != 0;
  }
  if (version >= kFormatVersionNav) params.nav_dim = in.pod<std::uint64_t>();
  IndexProfile profile;
  if (version >= kFormatVersionProfile) {
    const auto string = [&] {
      const auto length = in.pod<std::uint32_t>();
      if (length > kMaxProfileString) throw std::runtime_error(path + ": corrupt profile");
      std::string text(length, '\0');
      in.read(text.data(), length);
      return text;
    };
    const auto flags = in.pod<std::uint8_t>();
    if ((flags & ~1u) != 0) throw std::runtime_error(path + ": unsupported storage flags");
    params.half_vectors = (flags & 1u) != 0;
    profile.k = in.pod<std::uint64_t>();
    profile.ef = in.pod<std::uint64_t>();
    profile.rerank = in.pod<std::uint64_t>();
    profile.rerank_z = in.pod<float>();
    profile.recall_target = in.pod<float>();
    profile.recall_measured = in.pod<float>();
    profile.calibrated_size = in.pod<std::uint64_t>();
    profile.nav_variance = in.pod<float>();
    const auto passes = in.pod<std::uint8_t>();
    profile.refined = (passes & 1) != 0;
    profile.built_on_codes = (passes & 2) != 0;
    profile.filled = in.pod<std::uint8_t>() != 0;
    profile.prefetch_depth = in.pod<std::uint64_t>();
    profile.prefetch_lines = in.pod<std::uint64_t>();
    profile.machine = string();
    const auto notes = in.pod<std::uint32_t>();
    if (notes > 256) throw std::runtime_error(path + ": corrupt profile");
    for (std::uint32_t i = 0; i < notes; ++i) {
      auto setting = string();
      profile.notes.emplace_back(std::move(setting), string());
    }
    if (profile.k == 0 || profile.k > (1u << 20) || profile.ef == 0 || profile.ef > (1u << 24) ||
        profile.rerank > (1u << 24) || !(profile.rerank_z >= 0.0f && profile.rerank_z < 1e6f) ||
        profile.prefetch_lines == 0 || profile.prefetch_lines > 64) {
      throw std::runtime_error(path + ": corrupt profile");
    }
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
  const bool rbq_file = params.quantization == Quantization::RaBitQ;
  if (rbq_file && !params.keep_vectors) throw std::runtime_error(path + ": corrupt header");
  const std::size_t per_node = (rbq_file  ? sizeof(rabitq::Factors) + rabitq::padded_dim(params.dim) / 8
                                : sq8_file ? sizeof(float) + (params.nav_dim != 0 ? params.nav_dim : params.dim)
                                           : params.dim * sizeof(float)) +
                               (1 + 2 * params.M) * sizeof(id_t) + sizeof(std::uint64_t) + 1;
  if (count > std::filesystem::file_size(path) / per_node) throw std::runtime_error(path + ": truncated");
  if (max_bytes != 0 && estimated_bytes(params) > max_bytes) {
    throw std::runtime_error(path + ": the index would need " + std::to_string(estimated_bytes(params) >> 20) +
                             " MiB, over the " + std::to_string(max_bytes >> 20) + " MiB limit");
  }

  auto index = std::make_unique<HnswIndex>(params);
  // A prefetch setting is a measurement of one kind of machine: keep it only on the same kind.
  if (!profile.machine.empty() && profile.machine.front() != '/' && profile.machine == machine_name()) {
    index->set_prefetch(profile.prefetch_depth, profile.prefetch_lines);
  }
  index->profile_ = std::move(profile);
  if (index->sq8() && in.pod<std::uint8_t>() != 0) {
    index->sq8_min_.resize(index->code_dim());
    index->sq8_scale_.resize(index->code_dim());
    in.array(index->sq8_min_.data(), index->code_dim());
    in.array(index->sq8_scale_.data(), index->code_dim());
    if (params.nav_dim != 0) {
      index->nav_basis_.resize(params.nav_dim * params.dim);
      in.array(index->nav_basis_.data(), index->nav_basis_.size());
      index->prepare_query_projection();
    }
  }
  if (index->rabitq() && in.pod<std::uint8_t>() != 0) {
    index->rbq_center_.resize(params.dim);
    index->rbq_signs_.resize(3 * index->rbq_padded_);
    in.array(index->rbq_center_.data(), params.dim);
    in.array(index->rbq_signs_.data(), index->rbq_signs_.size());
    for (const std::int8_t s : index->rbq_signs_) {
      if (s != 1 && s != -1) throw std::runtime_error(path + ": corrupt RaBitQ rotation");
    }
  }
  std::size_t deleted = 0;
  for (id_t id = 0; id < count; ++id) {
    if (index->rabitq()) {
      in.read(index->record(id), index->rabitq_record_bytes());
    } else if (index->sq8()) {
      in.read(index->record(id), sizeof(float) + index->code_dim());
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
    if (index->half_) in.array(index->half_.get() + static_cast<std::size_t>(id) * params.dim, params.dim);
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

constexpr char kGraphMagic[8] = {'E', 'P', 'Z', 'R', 'G', 'R', 'P', 'H'};
constexpr char kLegacyGraphMagic[8] = {'O', 'R', 'Y', 'X', 'G', 'R', 'P', 'H'};  // from before the rename
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
  if (quantized()) throw std::logic_error("export_graph: only full-precision indexes");
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
  char magic[sizeof(kGraphMagic)];
  for (char& c : magic) c = in.pod<char>();
  if (std::memcmp(magic, kGraphMagic, sizeof(magic)) != 0 && std::memcmp(magic, kLegacyGraphMagic, sizeof(magic)) != 0) {
    throw std::runtime_error("graph: not an Epzero graph");
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

}  // namespace epzero
