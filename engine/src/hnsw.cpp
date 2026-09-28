#include "oryx/hnsw.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <stdexcept>

#include "visited_pool.h"

namespace oryx {
namespace {

constexpr char kMagic[8] = {'O', 'R', 'Y', 'X', 'H', 'N', 'S', 'W'};
constexpr std::uint32_t kFormatVersion = 1;
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

template <typename T>
void write_pod(std::ostream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void write_array(std::ostream& out, const T* data, std::size_t count) {
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
}

template <typename T>
T read_pod(std::istream& in) {
  T value;
  in.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

template <typename T>
void read_array(std::istream& in, T* data, std::size_t count) {
  in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(T)));
}

}  // namespace

void HnswIndex::AlignedFree::operator()(std::byte* p) const noexcept { std::free(p); }

HnswIndex::HnswIndex(const HnswParams& params)
    : params_(params),
      simd_level_(active_simd_level()),
      dist_(get_distance_fn(params.metric, simd_level_)),
      max_links0_(2 * params.M),
      level_mult_(params.M > 1 ? 1.0 / std::log(static_cast<double>(params.M)) : 1.0),
      rng_(params.seed) {
  if (params_.dim == 0) throw std::invalid_argument("dim must be > 0");
  if (params_.max_elements == 0) throw std::invalid_argument("max_elements must be > 0");
  if (params_.max_elements >= std::numeric_limits<id_t>::max()) {
    throw std::invalid_argument("max_elements must fit in 32-bit node ids");
  }
  if (params_.M < 2) throw std::invalid_argument("M must be >= 2");

  links0_offset_ = params_.dim * sizeof(float);
  node_stride_ = round_up(links0_offset_ + (1 + max_links0_) * sizeof(id_t), 64);

  const std::size_t bytes = node_stride_ * params_.max_elements;
  auto* block = static_cast<std::byte*>(std::aligned_alloc(64, bytes));
  if (block == nullptr) throw std::bad_alloc();
  std::memset(block, 0, bytes);  // every neighbor count starts at zero
  level0_.reset(block);

  upper_links_.resize(params_.max_elements);
  levels_.resize(params_.max_elements, 0);
  labels_.resize(params_.max_elements, 0);
  node_locks_ = std::make_unique<std::mutex[]>(params_.max_elements);
  visited_pool_ = std::make_unique<detail::VisitedPool>(params_.max_elements);
  label_to_id_.reserve(params_.max_elements);
}

HnswIndex::~HnswIndex() = default;

HnswIndex::id_t* HnswIndex::links_of(id_t id, int layer) noexcept {
  if (layer == 0) return reinterpret_cast<id_t*>(level0_.get() + id * node_stride_ + links0_offset_);
  return upper_links_[id].get() + static_cast<std::size_t>(layer - 1) * (1 + params_.M);
}

const HnswIndex::id_t* HnswIndex::links_of(id_t id, int layer) const noexcept {
  return const_cast<HnswIndex*>(this)->links_of(id, layer);
}

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
  std::lock_guard lock(node_locks_[id]);
  const id_t* links = links_of(id, layer);
  out.assign(links + 1, links + 1 + links[0]);
}

int HnswIndex::max_level() const noexcept {
  const std::uint64_t entry = entry_.load(std::memory_order_acquire);
  return static_cast<int>(entry >> 32) - 1;
}

HnswIndex::id_t HnswIndex::greedy_descend(const float* query, id_t entry, int from_layer, int to_layer,
                                          SearchStats& stats, SearchTrace* trace) const {
  id_t current = entry;
  float current_dist = distance(query, vector_of(current));
  ++stats.distance_computations;
  std::vector<id_t> links;
  links.reserve(params_.M);

  for (int layer = from_layer; layer > to_layer; --layer) {
    bool improved = true;
    while (improved) {
      improved = false;
      const id_t expanding = current;
      copy_links(expanding, layer, links);
      ++stats.nodes_expanded;
      for (id_t neighbor : links) {
        const float d = distance(query, vector_of(neighbor));
        ++stats.distance_computations;
        const bool closer = d < current_dist;
        if (trace != nullptr) {
          trace->events.push_back({static_cast<std::uint32_t>(layer), labels_[expanding], labels_[neighbor], d, closer});
        }
        if (closer) {
          current = neighbor;
          current_dist = d;
          improved = true;
        }
      }
    }
  }
  return current;
}

std::vector<HnswIndex::Candidate> HnswIndex::search_layer(const float* query, id_t entry, std::size_t ef, int layer,
                                                          SearchStats& stats, SearchTrace* trace) const {
  auto visited = visited_pool_->acquire();
  std::priority_queue<Candidate, std::vector<Candidate>, CloserFirst> candidates;  // closest on top
  std::priority_queue<Candidate, std::vector<Candidate>, FartherFirst> best;       // farthest on top

  const float entry_dist = distance(query, vector_of(entry));
  ++stats.distance_computations;
  visited->test_and_set(entry);
  candidates.push({entry_dist, entry});
  best.push({entry_dist, entry});

  std::vector<id_t> links;
  links.reserve(max_links0_);

  while (!candidates.empty()) {
    const Candidate current = candidates.top();
    // Every remaining candidate is farther than the worst result we keep: the beam has converged.
    if (current.distance > best.top().distance) break;
    candidates.pop();

    copy_links(current.id, layer, links);
    ++stats.nodes_expanded;
    for (std::size_t i = 0; i < links.size(); ++i) {
      // Pull the next vector toward L1 while this one is being scored.
      if (i + 1 < links.size()) prefetch(vector_of(links[i + 1]));
      const id_t neighbor = links[i];
      if (visited->test_and_set(neighbor)) continue;

      const float d = distance(query, vector_of(neighbor));
      ++stats.distance_computations;
      const bool accepted = best.size() < ef || d < best.top().distance;
      if (trace != nullptr) {
        trace->events.push_back({static_cast<std::uint32_t>(layer), labels_[current.id], labels_[neighbor], d, accepted});
      }
      if (accepted) {
        candidates.push({d, neighbor});
        best.push({d, neighbor});
        if (best.size() > ef) best.pop();
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
      if (distance(vector_of(candidate.id), vector_of(k.id)) < candidate.distance) {
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
  {
    std::lock_guard lock(node_locks_[id]);
    id_t* links = links_of(id, layer);
    std::vector<Candidate> merged = neighbors;
    for (std::size_t j = 0; j < links[0]; ++j) {
      const id_t existing = links[1 + j];
      const bool duplicate = std::any_of(merged.begin(), merged.end(), [&](const Candidate& c) { return c.id == existing; });
      if (!duplicate) merged.push_back({distance(vector_of(id), vector_of(existing)), existing});
    }
    merged = select_neighbors(std::move(merged), capacity);
    links[0] = static_cast<id_t>(merged.size());
    for (std::size_t j = 0; j < merged.size(); ++j) links[1 + j] = merged[j].id;
  }

  // Reverse links, one neighbor lock at a time.
  for (const Candidate& neighbor : neighbors) {
    std::lock_guard lock(node_locks_[neighbor.id]);
    id_t* links = links_of(neighbor.id, layer);
    const std::size_t count = links[0];
    if (count < capacity) {
      links[1 + count] = id;
      links[0] = static_cast<id_t>(count + 1);
      continue;
    }
    // The neighbor is full: re-run the heuristic over its current links plus the new node.
    const float* base = vector_of(neighbor.id);
    std::vector<Candidate> pool;
    pool.reserve(count + 1);
    pool.push_back({neighbor.distance, id});
    for (std::size_t j = 0; j < count; ++j) {
      pool.push_back({distance(base, vector_of(links[1 + j])), links[1 + j]});
    }
    const std::vector<Candidate> kept = select_neighbors(std::move(pool), capacity);
    links[0] = static_cast<id_t>(kept.size());
    for (std::size_t j = 0; j < kept.size(); ++j) links[1 + j] = kept[j].id;
  }
}

void HnswIndex::add(std::uint64_t label, const float* vector) {
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
  // through a neighbor list, which they read under that neighbor's lock, so these writes are
  // visible to them (the lock release/acquire orders them).
  float* stored = vector_of(id);
  std::memcpy(stored, vector, params_.dim * sizeof(float));
  if (params_.metric == Metric::Cosine) normalize(stored, params_.dim);
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

  id_t entry_point = static_cast<id_t>(entry & 0xffffffffu);
  const int top_level = static_cast<int>(entry >> 32) - 1;
  SearchStats unused;
  if (level < top_level) entry_point = greedy_descend(stored, entry_point, top_level, level, unused, nullptr);

  for (int layer = std::min(level, top_level); layer >= 0; --layer) {
    std::vector<Candidate> candidates =
        search_layer(stored, entry_point, params_.ef_construction, layer, unused, nullptr);
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
  const id_t start = greedy_descend(query, entry_point, top_level, 0, stats, trace);
  const std::vector<Candidate> candidates = search_layer(query, start, std::max(options.ef, k), 0, stats, trace);

  std::vector<SearchResult> results;
  results.reserve(std::min(k, candidates.size()));
  for (std::size_t i = 0; i < candidates.size() && i < k; ++i) {
    results.push_back({labels_[candidates[i].id], candidates[i].distance});
  }
  if (options.stats != nullptr) *options.stats = stats;
  return results;
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

std::size_t HnswIndex::memory_bytes() const noexcept {
  std::size_t bytes = node_stride_ * params_.max_elements;
  bytes += params_.max_elements * (sizeof(std::uint64_t) + sizeof(std::int8_t) + sizeof(std::mutex) +
                                   sizeof(std::unique_ptr<id_t[]>));
  const std::size_t count = size();
  for (std::size_t i = 0; i < count; ++i) {
    bytes += static_cast<std::size_t>(levels_[i]) * (1 + params_.M) * sizeof(id_t);
  }
  // Rough cost of an unordered_map node plus its bucket pointer.
  bytes += count * (sizeof(std::uint64_t) + sizeof(id_t) + 2 * sizeof(void*)) + params_.max_elements * sizeof(void*);
  return bytes;
}

// File format (little-endian): magic, version, params, count, entry, then per node:
// vector, layer-0 links, label, level, upper-layer links. The format is independent of
// in-memory padding so it survives layout changes.
void HnswIndex::save(const std::string& path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot open " + path + " for writing");
  out.exceptions(std::ios::failbit | std::ios::badbit);

  out.write(kMagic, sizeof(kMagic));
  write_pod(out, kFormatVersion);
  write_pod<std::uint64_t>(out, params_.dim);
  write_pod<std::uint32_t>(out, static_cast<std::uint32_t>(params_.metric));
  write_pod<std::uint64_t>(out, params_.max_elements);
  write_pod<std::uint64_t>(out, params_.M);
  write_pod<std::uint64_t>(out, params_.ef_construction);
  write_pod<std::uint64_t>(out, params_.seed);
  const std::size_t count = size();
  write_pod<std::uint64_t>(out, count);
  write_pod<std::uint64_t>(out, entry_.load(std::memory_order_acquire));

  for (id_t id = 0; id < count; ++id) {
    std::lock_guard lock(node_locks_[id]);
    write_array(out, vector_of(id), params_.dim);
    write_array(out, links_of(id, 0), 1 + max_links0_);
    write_pod(out, labels_[id]);
    write_pod(out, levels_[id]);
    if (levels_[id] > 0) write_array(out, upper_links_[id].get(), static_cast<std::size_t>(levels_[id]) * (1 + params_.M));
  }
}

std::unique_ptr<HnswIndex> HnswIndex::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  in.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);

  char magic[sizeof(kMagic)];
  in.read(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw std::runtime_error(path + " is not an Oryx index");
  if (read_pod<std::uint32_t>(in) != kFormatVersion) throw std::runtime_error(path + ": unsupported format version");

  HnswParams params;
  params.dim = read_pod<std::uint64_t>(in);
  params.metric = static_cast<Metric>(read_pod<std::uint32_t>(in));
  params.max_elements = read_pod<std::uint64_t>(in);
  params.M = read_pod<std::uint64_t>(in);
  params.ef_construction = read_pod<std::uint64_t>(in);
  params.seed = read_pod<std::uint64_t>(in);
  const std::size_t count = read_pod<std::uint64_t>(in);
  const std::uint64_t entry = read_pod<std::uint64_t>(in);
  if (count > params.max_elements) throw std::runtime_error(path + ": corrupt node count");

  auto index = std::make_unique<HnswIndex>(params);
  for (id_t id = 0; id < count; ++id) {
    read_array(in, index->vector_of(id), params.dim);
    id_t* links0 = index->links_of(id, 0);
    read_array(in, links0, 1 + index->max_links0_);
    index->labels_[id] = read_pod<std::uint64_t>(in);
    const auto level = read_pod<std::int8_t>(in);
    if (level < 0 || level > kMaxLevel || links0[0] > index->max_links0_) {
      throw std::runtime_error(path + ": corrupt node record");
    }
    index->levels_[id] = level;
    if (level > 0) {
      const std::size_t n = static_cast<std::size_t>(level) * (1 + params.M);
      index->upper_links_[id] = std::make_unique<id_t[]>(n);
      read_array(in, index->upper_links_[id].get(), n);
    }
    index->label_to_id_.emplace(index->labels_[id], id);
  }

  // Reject any link that points outside the loaded nodes, so a corrupt file cannot cause
  // out-of-bounds reads during search.
  auto valid = [&](const id_t* links, std::size_t capacity) {
    if (links[0] > capacity) return false;
    return std::all_of(links + 1, links + 1 + links[0], [&](id_t n) { return n < count; });
  };
  for (id_t id = 0; id < count; ++id) {
    bool ok = valid(index->links_of(id, 0), index->max_links0_);
    for (int layer = 1; ok && layer <= index->levels_[id]; ++layer) ok = valid(index->links_of(id, layer), params.M);
    if (!ok) throw std::runtime_error(path + ": corrupt neighbor list");
  }
  const id_t entry_id = static_cast<id_t>(entry & 0xffffffffu);
  const int entry_level = static_cast<int>(entry >> 32) - 1;
  if (count > 0 && (entry_id >= count || entry_level != index->levels_[entry_id])) {
    throw std::runtime_error(path + ": corrupt entry point");
  }

  index->count_.store(count, std::memory_order_release);
  index->entry_.store(entry, std::memory_order_release);
  return index;
}

}  // namespace oryx
