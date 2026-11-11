// Python bindings. Batch calls release the GIL and fan out across C++ threads, so Python only
// pays call overhead once per batch.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <limits>
#include <shared_mutex>

#include "epzero/auto.h"
#include "epzero/database.h"
#include "epzero/hnsw.h"
#include "epzero/parallel.h"
#include "epzero/recall.h"
#include "epzero/metadata.h"
#include "epzero/text.h"

namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using LabelArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

namespace {

epzero::Metric parse_metric(const std::string& name) {
  if (name == "l2") return epzero::Metric::L2;
  if (name == "ip") return epzero::Metric::InnerProduct;
  if (name == "cosine") return epzero::Metric::Cosine;
  throw std::invalid_argument("metric must be 'l2', 'ip' or 'cosine', not '" + name + "'");
}

epzero::Quantization parse_quantization(const std::string& name) {
  if (name == "none") return epzero::Quantization::None;
  if (name == "sq8") return epzero::Quantization::SQ8;
  if (name == "rabitq") return epzero::Quantization::RaBitQ;
  throw std::invalid_argument("quantization must be 'none', 'sq8' or 'rabitq', not '" + name + "'");
}

epzero::FilterStrategy parse_strategy(const std::string& name) {
  for (const auto s : {epzero::FilterStrategy::Auto, epzero::FilterStrategy::Exact, epzero::FilterStrategy::Inline,
                       epzero::FilterStrategy::TwoHop}) {
    if (name == epzero::to_string(s)) return s;
  }
  throw std::invalid_argument("strategy must be 'auto', 'exact', 'inline' or 'two-hop'");
}

epzero::SyncMode parse_sync(const std::string& name) {
  if (name == "always") return epzero::SyncMode::Always;
  if (name == "batch") return epzero::SyncMode::Batch;
  if (name == "none") return epzero::SyncMode::None;
  throw std::invalid_argument("sync must be 'always', 'batch' or 'none'");
}

epzero::HnswParams make_params(std::size_t dim, const std::string& metric, std::size_t max_elements, std::size_t M,
                             std::size_t ef_construction, std::uint64_t seed, const std::string& quantization,
                             bool keep_vectors, std::size_t nav_dim = 0, bool fill_links = false, bool fast_build = false,
                             bool half_vectors = false) {
  return {.dim = dim, .metric = parse_metric(metric), .max_elements = max_elements, .M = M,
          .ef_construction = ef_construction, .seed = seed, .quantization = parse_quantization(quantization),
          .keep_vectors = keep_vectors, .nav_dim = nav_dim, .fill_links = fill_links,
          .fast_build = fast_build, .half_vectors = half_vectors};
}

const float* rows_of(const FloatArray& data, std::size_t dim, std::size_t& n) {
  if (data.ndim() == 1 && static_cast<std::size_t>(data.shape(0)) == dim) {
    n = 1;
  } else if (data.ndim() == 2 && static_cast<std::size_t>(data.shape(1)) == dim) {
    n = static_cast<std::size_t>(data.shape(0));
  } else {
    std::string got = "(";
    for (py::ssize_t i = 0; i < data.ndim(); ++i) got += (i ? ", " : "") + std::to_string(data.shape(i));
    throw std::invalid_argument("expected an array of shape (n, " + std::to_string(dim) + "), got " + got +
                                (data.ndim() == 1 ? ",)" : ")"));
  }
  return data.data();
}

// Labels from Python (None = consecutive from `first`), validated against the row count.
std::vector<std::uint64_t> labels_of(const py::object& labels, std::size_t n, std::size_t first) {
  std::vector<std::uint64_t> ids(n);
  if (labels.is_none()) {
    for (std::size_t i = 0; i < n; ++i) ids[i] = first + i;
    return ids;
  }
  const auto arr = labels.cast<LabelArray>();
  if (arr.ndim() != 1 || static_cast<std::size_t>(arr.shape(0)) != n) {
    throw std::invalid_argument("labels must be a 1-D array with one label per row");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (arr.data()[i] < 0) throw std::invalid_argument("labels must be non-negative");
    ids[i] = static_cast<std::uint64_t>(arr.data()[i]);
  }
  return ids;
}

// Runs `search(row, k)` for every query row on `threads` threads and packs (labels, distances)
// arrays of shape (n, k); missing results are -1 / inf.
template <typename Search>
py::tuple batch_search(const FloatArray& queries, std::size_t dim, std::size_t k, std::size_t threads, Search search) {
  std::size_t n;
  const float* rows = rows_of(queries, dim, n);
  LabelArray labels({n, k});
  FloatArray distances({n, k});
  std::int64_t* out_labels = labels.mutable_data();
  float* out_distances = distances.mutable_data();
  {
    py::gil_scoped_release release;
    epzero::parallel_for(0, n, threads, [&](std::size_t q) {
      const auto results = search(rows + q * dim, k);
      for (std::size_t i = 0; i < k; ++i) {
        const bool found = i < results.size();
        out_labels[q * k + i] = found ? static_cast<std::int64_t>(results[i].label) : -1;
        out_distances[q * k + i] = found ? results[i].distance : std::numeric_limits<float>::infinity();
      }
    });
  }
  return py::make_tuple(labels, distances);
}

// 1234567 -> "1,234,567", for the reprs.
std::string grouped(std::size_t n) {
  std::string digits = std::to_string(n);
  for (int i = static_cast<int>(digits.size()) - 3; i > 0; i -= 3) digits.insert(static_cast<std::size_t>(i), ",");
  return digits;
}

std::string quantization_name(epzero::Quantization q) {
  switch (q) {
    case epzero::Quantization::None: return "none";
    case epzero::Quantization::SQ8: return "sq8";
    case epzero::Quantization::RaBitQ: return "rabitq";
  }
  return "unknown";
}

py::object prefetch_object(std::size_t depth, std::size_t lines) {
  return py::make_tuple(depth == std::numeric_limits<std::size_t>::max() ? py::object(py::none()) : py::int_(depth), lines);
}

py::dict params_dict(const epzero::HnswParams& p) {
  py::dict d;
  d["dim"] = p.dim;
  d["metric"] = std::string(epzero::to_string(p.metric));
  d["max_elements"] = p.max_elements;
  d["M"] = p.M;
  d["ef_construction"] = p.ef_construction;
  d["seed"] = p.seed;
  d["quantization"] = quantization_name(p.quantization);
  d["keep_vectors"] = p.keep_vectors;
  d["nav_dim"] = p.nav_dim;
  d["fast_build"] = p.fast_build;
  d["half_vectors"] = p.half_vectors;
  return d;
}

py::dict profile_dict(const epzero::IndexProfile& p) {
  py::dict d;
  d["k"] = p.k;
  d["ef"] = p.ef;
  d["rerank"] = p.rerank;
  d["rerank_z"] = p.rerank_z;
  d["recall_target"] = p.recall_target;
  d["recall_measured"] = p.recall_measured;
  d["calibrated_size"] = p.calibrated_size;
  d["nav_variance"] = p.nav_variance;
  d["fast_build"] = p.built_on_codes;
  d["refine"] = p.refined;
  d["fill_after"] = p.filled;
  d["prefetch"] = prefetch_object(p.prefetch_depth, p.prefetch_lines);
  d["machine"] = p.machine;
  py::dict notes;
  for (const auto& [setting, why] : p.notes) notes[py::str(setting)] = why;
  d["notes"] = notes;
  return d;
}

// AutoOptions from build()'s arguments; `fixed` holds the settings the caller chose.
epzero::AutoOptions auto_options(const std::string& metric, std::size_t M, std::size_t ef_construction, std::size_t k,
                                 float recall, std::size_t threads, std::size_t capacity, std::uint64_t seed,
                                 const py::dict& fixed) {
  epzero::AutoOptions o{.metric = parse_metric(metric), .M = M, .ef_construction = ef_construction, .k = k,
                        .recall = recall, .threads = threads, .capacity = capacity, .seed = seed};
  for (const auto& [key, value] : fixed) {
    const auto name = key.cast<std::string>();
    if (value.is_none()) continue;
    if (name == "quantization") o.fixed.quantization = parse_quantization(value.cast<std::string>());
    else if (name == "nav_dim") o.fixed.nav_dim = value.cast<std::size_t>();
    else if (name == "fast_build") o.fixed.fast_build = value.cast<bool>();
    else if (name == "refine") o.fixed.refine = value.cast<bool>();
    else if (name == "fill_after") o.fixed.fill_after = value.cast<bool>();
    else if (name == "rerank") o.fixed.rerank = value.cast<std::size_t>();
    else if (name == "rerank_z") o.fixed.rerank_z = value.cast<float>();
    else if (name == "ef") o.fixed.ef = value.cast<std::size_t>();
    else if (name == "half_vectors") o.fixed.half_vectors = value.cast<bool>();
    else if (name == "prefetch") {
      const auto pair = value.cast<py::tuple>();
      if (pair.size() != 2) throw std::invalid_argument("prefetch must be (depth, lines)");
      const std::size_t depth = pair[0].is_none() ? std::numeric_limits<std::size_t>::max() : pair[0].cast<std::size_t>();
      o.fixed.prefetch = std::make_pair(depth, pair[1].cast<std::size_t>());
    } else {
      throw std::invalid_argument("unknown setting: " + name);
    }
  }
  return o;
}

class Index {
 public:
  Index(std::size_t dim, const std::string& metric, std::size_t max_elements, std::size_t M,
        std::size_t ef_construction, std::uint64_t seed, const std::string& quantization, bool keep_vectors,
        std::size_t nav_dim, bool fill_links, bool fast_build, bool half_vectors)
      : index_(std::make_unique<epzero::HnswIndex>(make_params(dim, metric, max_elements, M, ef_construction, seed,
                                                               quantization, keep_vectors, nav_dim, fill_links, fast_build,
                                                               half_vectors))) {}

  explicit Index(std::unique_ptr<epzero::HnswIndex> index) : index_(std::move(index)) {}

  void train(const FloatArray& data) {
    std::size_t n;
    const float* rows = rows_of(data, index_->params().dim, n);
    py::gil_scoped_release release;
    index_->train(rows, n);
  }

  // Grows the capacity when the rows don't fit (at least doubling, so a stream of small adds stays
  // cheap): the nodes are copied, the graph kept.
  void add(const FloatArray& data, const py::object& labels, std::size_t threads) {
    const std::size_t dim = index_->params().dim;
    std::size_t n;
    const float* rows = rows_of(data, dim, n);
    const auto ids = labels_of(labels, n, index_->size());
    py::gil_scoped_release release;
    {
      std::unique_lock growing(sync_->lock);
      const std::size_t needed = index_->size() + n;
      const std::size_t capacity = index_->params().max_elements;
      if (needed > capacity) {
        if (sync_->guards.load() > 0) {
          throw std::length_error("the index is full; with a RecallGuard attached, grow it first with reserve()");
        }
        index_->reserve(std::max(needed, 2 * capacity));
      }
    }
    std::shared_lock reading(sync_->lock);
    epzero::parallel_for(0, n, threads, [&](std::size_t i) { index_->add(ids[i], rows + i * dim); });
  }

  void reserve(std::size_t capacity) {
    py::gil_scoped_release release;
    std::unique_lock growing(sync_->lock);
    if (capacity > index_->params().max_elements && sync_->guards.load() > 0) {
      throw std::length_error("reserve() moves the index's memory: detach the RecallGuard first");
    }
    index_->reserve(capacity);
  }
  std::size_t capacity() const { return index_->params().max_elements; }

  // Searches hold this shared; growing the index takes it alone (its memory moves).
  std::shared_lock<std::shared_mutex> reading() const { return std::shared_lock(sync_->lock); }
  void attach_guard() const { ++sync_->guards; }
  void detach_guard() const { --sync_->guards; }

  bool remove(std::uint64_t label) { return index_->remove(label); }

  // `filter` is None, an epzero.Filter from filter(), or a sequence of allowed labels (compiled
  // for this call; build an epzero.Filter once to reuse it across calls).
  // A filter argument: None, an epzero.Filter, or a sequence of labels compiled into `storage`.
  const epzero::SearchFilter* resolve_filter(const py::object& filter, epzero::SearchFilter& storage) const {
    if (filter.is_none()) return nullptr;
    if (py::isinstance<epzero::SearchFilter>(filter)) return filter.cast<const epzero::SearchFilter*>();
    storage = make_filter(filter.cast<LabelArray>());
    return &storage;
  }

  // Settings left as None come from the index's profile (see IndexProfile): a hand-built index has
  // the old defaults (k 10, ef 64, re-rank 2 x k), an automatic one what build() calibrated.
  py::tuple search(const FloatArray& queries, const py::object& k_arg, const py::object& ef_arg,
                   const py::object& rerank_arg, std::size_t threads, const py::object& filter,
                   const std::string& strategy, std::size_t patience, const py::object& rerank_z_arg) const {
    const epzero::IndexProfile& p = index_->profile();
    const std::size_t k = k_arg.is_none() ? p.k : k_arg.cast<std::size_t>();
    // The profile's beam was calibrated for its k; a larger k needs at least k candidates.
    const std::size_t ef = ef_arg.is_none() ? std::max(p.ef, k) : ef_arg.cast<std::size_t>();
    const std::size_t rerank = rerank_arg.is_none() ? p.rerank : rerank_arg.cast<std::size_t>();
    const float rerank_z = rerank_z_arg.is_none() ? p.rerank_z : rerank_z_arg.cast<float>();
    const epzero::FilterStrategy chosen = parse_strategy(strategy);
    epzero::SearchFilter compiled;
    const epzero::SearchFilter* active = resolve_filter(filter, compiled);
    return batch_search(queries, index_->params().dim, k, threads, [&](const float* q, std::size_t kk) {
      const auto lock = reading();
      return index_->search(q, kk, {.ef = ef, .patience = patience, .rerank = rerank, .rerank_z = rerank_z, .filter = active, .strategy = chosen});
    });
  }

  // Prefetch settings: depth None means every neighbor up front (the default).
  py::object prefetch() const {
    const auto [depth, lines] = index_->prefetch_settings();
    return prefetch_object(depth, lines);
  }
  py::dict profile() const { return profile_dict(index_->profile()); }

  py::dict recalibrate(const FloatArray& queries, const py::object& recall, std::size_t threads) {
    std::size_t n;
    const float* rows = rows_of(queries, index_->params().dim, n);
    const float target = recall.is_none() ? 0.0f : recall.cast<float>();
    std::pair<std::size_t, float> result;
    {
      py::gil_scoped_release release;
      result = epzero::recalibrate(*index_, rows, n, target, threads);
    }
    py::dict out;
    out["ef"] = result.first;
    out["recall"] = result.second;
    return out;
  }

  py::dict health(std::size_t threads) const {
    epzero::IndexHealth h;
    {
      py::gil_scoped_release release;
      h = epzero::check_health(*index_, threads);
    }
    py::dict out;
    out["live"] = h.live;
    out["deleted"] = h.deleted;
    out["growth"] = h.growth;
    out["nav_variance_built"] = h.nav_variance_built;
    out["nav_variance_now"] = h.nav_variance_now;
    out["advice"] = h.advice;
    return out;
  }

  py::dict check_queries(const FloatArray& queries, std::size_t threads) const {
    std::size_t n;
    const float* rows = rows_of(queries, index_->params().dim, n);
    epzero::QueryCheck c;
    {
      py::gil_scoped_release release;
      const auto lock = reading();
      c = epzero::check_queries(*index_, rows, n, threads);
    }
    py::dict out;
    out["query_distance"] = c.query_distance;
    out["row_distance"] = c.row_distance;
    out["ratio"] = c.ratio;
    out["shifted"] = c.shifted;
    out["advice"] = c.advice;
    return out;
  }

  py::array_t<float> get_vector(std::uint64_t label) const {
    py::array_t<float> out(static_cast<py::ssize_t>(index_->params().dim));
    if (!index_->get_vector(label, out.mutable_data())) throw py::key_error("label " + std::to_string(label) + " is not in the index");
    return out;
  }
  py::dict params() const { return params_dict(index_->params()); }
  void set_prefetch(const py::object& depth, std::size_t lines) {
    index_->set_prefetch(depth.is_none() ? std::numeric_limits<std::size_t>::max() : depth.cast<std::size_t>(), lines);
  }
  void refine_links(std::size_t threads, std::size_t ef) {
    py::gil_scoped_release release;
    index_->refine_links(threads, ef);
  }

  std::size_t fill_free_links(std::size_t threads) {
    py::gil_scoped_release release;
    return index_->fill_free_links(threads);
  }

  py::tuple tune_prefetch(const FloatArray& queries, std::size_t k, std::size_t ef) {
    std::size_t n;
    const float* rows = rows_of(queries, index_->params().dim, n);
    {
      py::gil_scoped_release release;
      index_->tune_prefetch(rows, n, k, ef);
    }
    return prefetch();
  }

  // Exact answers by a full scan, the ground truth recall is measured against.
  py::tuple exact_search(const FloatArray& queries, std::size_t k, std::size_t threads, const py::object& filter) const {
    epzero::SearchFilter compiled;
    const epzero::SearchFilter* active = resolve_filter(filter, compiled);
    return batch_search(queries, index_->params().dim, k, threads, [&](const float* q, std::size_t kk) {
      const auto lock = reading();
      return index_->exact_search(q, kk, active);
    });
  }

  py::dict explain(const FloatArray& query, std::size_t k, std::uint64_t label, const py::object& ef_arg,
                   const py::object& filter) const {
    std::size_t n;
    const float* row = rows_of(query, index_->params().dim, n);
    if (n != 1) throw std::invalid_argument("explain takes exactly one query");
    epzero::SearchFilter compiled;
    const epzero::SearchFilter* active = resolve_filter(filter, compiled);
    const epzero::IndexProfile& p = index_->profile();
    const std::size_t ef = ef_arg.is_none() ? std::max(p.ef, k) : ef_arg.cast<std::size_t>();  // needs the GIL
    epzero::MissExplanation why;
    {
      py::gil_scoped_release release;
      why = index_->explain(row, k, label, {.ef = ef, .rerank = p.rerank, .rerank_z = p.rerank_z, .filter = active});
    }
    py::dict out;
    out["cause"] = std::string(epzero::to_string(why.cause));
    out["exact_rank"] = why.exact_rank;
    out["distance"] = why.distance;
    out["kth_distance"] = why.kth_distance;
    out["scored"] = why.scored;
    out["ef_needed"] = why.ef_needed;
    return out;
  }

  epzero::SearchFilter make_filter(const LabelArray& labels) const {
    if (labels.ndim() != 1) throw std::invalid_argument("filter labels must be a 1-D sequence");
    std::vector<std::uint64_t> allowed(static_cast<std::size_t>(labels.shape(0)));
    for (std::size_t i = 0; i < allowed.size(); ++i) {
      if (labels.data()[i] < 0) throw std::invalid_argument("labels must be non-negative");
      allowed[i] = static_cast<std::uint64_t>(labels.data()[i]);
    }
    py::gil_scoped_release release;
    return index_->make_filter(allowed.data(), allowed.size());
  }

  // One query with its full traversal, for the visualizer and debugging.
  py::dict trace(const FloatArray& query, std::size_t k, const py::object& ef_arg) const {
    std::size_t n;
    const float* row = rows_of(query, index_->params().dim, n);
    if (n != 1) throw std::invalid_argument("trace takes exactly one query");
    epzero::SearchTrace trace;
    epzero::SearchStats stats;
    const epzero::IndexProfile& p = index_->profile();
    const std::size_t ef = ef_arg.is_none() ? std::max(p.ef, k) : ef_arg.cast<std::size_t>();
    const auto results =
        index_->search(row, k, {.ef = ef, .rerank = p.rerank, .rerank_z = p.rerank_z, .stats = &stats, .trace = &trace});

    py::list events;
    for (const auto& e : trace.events) {
      events.append(py::make_tuple(e.layer, e.from, e.to, e.distance, e.accepted));
    }
    py::list result_list;
    for (const auto& r : results) result_list.append(py::make_tuple(r.label, r.distance));
    py::dict out;
    out["entry_label"] = trace.entry_label;
    out["entry_layer"] = trace.entry_layer;
    out["events"] = events;
    out["results"] = result_list;
    out["distance_computations"] = stats.distance_computations;
    out["nodes_expanded"] = stats.nodes_expanded;
    return out;
  }

  Index compact(std::size_t threads) const {
    py::gil_scoped_release release;
    return Index(index_->compacted(threads));
  }

  void save(const std::string& path) const {
    py::gil_scoped_release release;
    index_->save(path);
  }

  static Index load(const std::string& path) {
    py::gil_scoped_release release;
    return Index(epzero::HnswIndex::load(path));
  }

  const epzero::HnswIndex& raw() const { return *index_; }

 private:
  std::unique_ptr<epzero::HnswIndex> index_;
  // Behind a pointer, so the wrapper stays movable (it's returned by value).
  struct Sync {
    std::shared_mutex lock;
    std::atomic<int> guards{0};
  };
  std::unique_ptr<Sync> sync_ = std::make_unique<Sync>();
};

py::dict estimate_to_dict(const epzero::RecallEstimate& e) {
  py::dict out;
  out["samples"] = e.samples;
  out["recall"] = e.recall;
  out["lower"] = e.lower;
  out["upper"] = e.upper;
  return out;
}

// Searches through a RecallGuard: the beam width per segment follows measured recall.
class Guard {
 public:
  // target and initial_ef default to the index's profile: the recall it was calibrated for (else 0.95)
  // and the beam width that reached it.
  Guard(const Index& index, const py::object& target_arg, const py::object& initial_ef_arg, std::size_t min_ef,
        std::size_t max_ef, std::size_t min_samples, double margin, double sample_rate, std::size_t window,
        std::size_t reference_ef, bool per_query)
      : index_(index),
        guard_(index.raw(), {.target = target_arg.is_none() ? (index.raw().profile().recall_target > 0.0f
                                                                    ? static_cast<double>(index.raw().profile().recall_target)
                                                                    : 0.95)
                                                              : target_arg.cast<double>(),
                             .initial_ef = initial_ef_arg.is_none() ? index.raw().profile().ef : initial_ef_arg.cast<std::size_t>(),
                             .min_ef = min_ef, .max_ef = max_ef,
                             .min_samples = min_samples, .margin = margin, .per_query = per_query,
                             .monitor = {.sample_rate = sample_rate, .window = window, .reference_ef = reference_ef}}) {
    // Its background checks search the index on their own thread, which the index's lock can't cover:
    // while a guard is attached, the index refuses to grow.
    index_.attach_guard();
  }
  ~Guard() { index_.detach_guard(); }
  Guard(const Guard&) = delete;
  Guard& operator=(const Guard&) = delete;

  py::tuple search(const FloatArray& queries, std::size_t k, std::size_t threads, const py::object& filter) {
    epzero::SearchFilter compiled;
    const epzero::SearchFilter* active = nullptr;
    if (py::isinstance<epzero::SearchFilter>(filter)) {
      active = filter.cast<const epzero::SearchFilter*>();
    } else if (!filter.is_none()) {
      compiled = index_.make_filter(filter.cast<LabelArray>());
      active = &compiled;
    }
    return batch_search(queries, index_.raw().params().dim, k, threads,
                        [&](const float* q, std::size_t kk) {
                          const auto lock = index_.reading();
                          return guard_.search(q, kk, active);
                        });
  }

  py::list report() const {
    py::list out;
    for (const auto& s : guard_.report()) {
      py::dict row = estimate_to_dict(s.estimate);
      row["segment"] = s.name;
      row["ef"] = s.ef;
      row["changes"] = s.changes;
      out.append(row);
    }
    return out;
  }

  void drain() {
    py::gil_scoped_release release;
    guard_.monitor().drain();
  }

 private:
  const Index& index_;
  epzero::RecallGuard guard_;
};

// Ranked results as (labels, scores) arrays, scores larger-is-better (the negated distances).
py::tuple ranked(const std::vector<epzero::SearchResult>& results) {
  py::array_t<std::int64_t> labels(static_cast<py::ssize_t>(results.size()));
  py::array_t<float> scores(static_cast<py::ssize_t>(results.size()));
  for (std::size_t i = 0; i < results.size(); ++i) {
    labels.mutable_data()[i] = static_cast<std::int64_t>(results[i].label);
    scores.mutable_data()[i] = -results[i].distance;
  }
  return py::make_tuple(labels, scores);
}

// Python values <-> metadata values. bool is checked before int (in Python a bool is an int).
std::optional<epzero::MetaValue> meta_value(const py::handle& v) {
  if (py::isinstance<py::bool_>(v)) return epzero::MetaValue(v.cast<bool>());
  if (py::isinstance<py::int_>(v)) return epzero::MetaValue(v.cast<std::int64_t>());
  if (py::isinstance<py::float_>(v)) return epzero::MetaValue(v.cast<double>());
  if (py::isinstance<py::str>(v)) return epzero::MetaValue(v.cast<std::string>());
  return std::nullopt;
}

py::object meta_object(const epzero::MetaValue& v) {
  if (const auto* i = std::get_if<std::int64_t>(&v)) return py::int_(*i);
  if (const auto* d = std::get_if<double>(&v)) return py::float_(*d);
  if (const auto* t = std::get_if<std::string>(&v)) return py::str(*t);
  return py::bool_(std::get<bool>(v));
}

epzero::MetadataIndex::Fields meta_fields(const py::dict& fields) {
  epzero::MetadataIndex::Fields out;
  for (const auto& [key, value] : fields) {
    const auto v = meta_value(value);
    if (!v) {
      throw std::invalid_argument("metadata values must be str, int, float or bool (field '" + key.cast<std::string>() + "')");
    }
    out.emplace_back(key.cast<std::string>(), *v);
  }
  return out;
}

std::vector<epzero::MetaValue> meta_list(const py::handle& values, const std::string& field) {
  std::vector<epzero::MetaValue> out;
  for (const auto& item : values) {
    const auto v = meta_value(item);
    if (!v) throw std::invalid_argument("filter values must be str, int, float or bool (field '" + field + "')");
    out.push_back(*v);
  }
  return out;
}

// A `where` dict: {field: value} for equality, {field: [values]} for any of them, or {field: {op: x}}
// with $eq, $ne, $in, $nin, $gt, $gte, $lt, $lte and $exists. Every entry must hold.
std::vector<epzero::MetaCondition> meta_where(const py::dict& where) {
  std::vector<epzero::MetaCondition> out;
  for (const auto& [key, spec] : where) {
    const auto field = key.cast<std::string>();
    if (py::isinstance<py::dict>(spec)) {
      for (const auto& [op_key, arg] : spec.cast<py::dict>()) {
        const auto op = op_key.cast<std::string>();
        epzero::MetaCondition c{.field = field};
        const auto one = [&] {
          const auto v = meta_value(arg);
          if (!v) throw std::invalid_argument(op + " needs a str, int, float or bool (field '" + field + "')");
          return *v;
        };
        const auto number = [&] {
          if (py::isinstance<py::bool_>(arg) || !(py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg))) {
            throw std::invalid_argument(op + " needs a number (field '" + field + "')");
          }
          return arg.cast<double>();
        };
        if (op == "$eq") c.any_of = {one()};
        else if (op == "$ne") { c.any_of = {one()}; c.negate = true; }
        else if (op == "$in") c.any_of = meta_list(arg, field);
        else if (op == "$nin") { c.any_of = meta_list(arg, field); c.negate = true; }
        else if (op == "$gt") { c.min = number(); c.min_exclusive = true; }
        else if (op == "$gte") c.min = number();
        else if (op == "$lt") { c.max = number(); c.max_exclusive = true; }
        else if (op == "$lte") c.max = number();
        else if (op == "$exists") {
          // A condition with no values and no range means "has the field".
          if (!arg.cast<bool>()) throw std::invalid_argument("$exists: false is not supported (labels without the field aren't indexed)");
        } else {
          throw std::invalid_argument("unknown filter operator " + op);
        }
        out.push_back(std::move(c));
      }
    } else if (py::isinstance<py::list>(spec) || py::isinstance<py::tuple>(spec) || py::isinstance<py::set>(spec)) {
      out.push_back({.field = field, .any_of = meta_list(spec, field)});
    } else {
      const auto v = meta_value(spec);
      if (!v) throw std::invalid_argument("filter values must be str, int, float or bool (field '" + field + "')");
      out.push_back({.field = field, .any_of = {*v}});
    }
  }
  return out;
}

class Metadata {
 public:
  Metadata() = default;
  explicit Metadata(epzero::MetadataIndex index) : index_(std::move(index)) {}

  void set(std::uint64_t label, const py::dict& fields) { index_.set(label, meta_fields(fields)); }
  void set_many(const py::object& labels, const py::list& rows) {
    const auto ids = labels_of(labels, rows.size(), 0);
    std::vector<epzero::MetadataIndex::Fields> parsed;
    parsed.reserve(rows.size());
    for (const auto& row : rows) parsed.push_back(meta_fields(row.cast<py::dict>()));
    py::gil_scoped_release release;
    for (std::size_t i = 0; i < ids.size(); ++i) index_.set(ids[i], parsed[i]);
  }
  bool remove(std::uint64_t label) { return index_.remove(label); }
  py::array_t<std::int64_t> select(const py::dict& where) const {
    const auto conditions = meta_where(where);
    std::vector<std::uint64_t> labels;
    {
      py::gil_scoped_release release;
      labels = index_.select(conditions);
    }
    py::array_t<std::int64_t> out(static_cast<py::ssize_t>(labels.size()));
    for (std::size_t i = 0; i < labels.size(); ++i) out.mutable_data()[i] = static_cast<std::int64_t>(labels[i]);
    return out;
  }
  py::dict get(std::uint64_t label) const {
    py::dict out;
    for (const auto& [name, value] : index_.get(label)) out[py::str(name)] = meta_object(value);
    return out;
  }
  std::size_t size() const { return index_.size(); }
  bool contains(std::uint64_t label) const { return index_.contains(label); }
  void save(const std::string& path) const {
    py::gil_scoped_release release;
    index_.save(path);
  }
  static Metadata load(const std::string& path) {
    py::gil_scoped_release release;
    return Metadata(epzero::MetadataIndex::load(path));
  }

 private:
  epzero::MetadataIndex index_;
};

class TextIndex {
 public:
  TextIndex(float k1, float b) : index_({.k1 = k1, .b = b}) {}
  explicit TextIndex(epzero::TextIndex index) : index_(std::move(index)) {}

  void add(const py::object& labels, const std::vector<std::string>& texts) {
    const auto ids = labels_of(labels, texts.size(), 0);
    py::gil_scoped_release release;
    for (std::size_t i = 0; i < texts.size(); ++i) index_.add(ids[i], texts[i]);
  }
  bool remove(std::uint64_t label) { return index_.remove(label); }
  py::tuple search(const std::string& query, std::size_t k) const {
    std::vector<epzero::SearchResult> results;
    {
      py::gil_scoped_release release;
      results = index_.search(query, k);
    }
    return ranked(results);
  }
  void save(const std::string& path) const { index_.save(path); }
  static TextIndex load(const std::string& path) { return TextIndex(epzero::TextIndex::load(path)); }
  const epzero::TextIndex& raw() const noexcept { return index_; }

 private:
  epzero::TextIndex index_;
};

class Database {
 public:
  explicit Database(std::unique_ptr<epzero::Database> db) : db_(std::move(db)) {}

  static Database open(const std::string& path, std::size_t dim, const std::string& metric, std::size_t max_elements,
                       std::size_t M, std::size_t ef_construction, const std::string& quantization, bool keep_vectors,
                       const std::string& sync, double checkpoint_mb) {
    epzero::DatabaseOptions options;
    options.sync = parse_sync(sync);
    options.checkpoint_bytes = static_cast<std::size_t>(checkpoint_mb * 1024 * 1024);
    const auto params = make_params(dim, metric, max_elements, M, ef_construction, 42, quantization, keep_vectors);
    py::gil_scoped_release release;
    return Database(epzero::Database::open(path, params, options));
  }

  void train(const FloatArray& data) {
    std::size_t n;
    const float* rows = rows_of(data, db_->index().params().dim, n);
    py::gil_scoped_release release;
    db_->train(rows, n);
  }

  // Inserts or replaces rows; a batch is one log write and one fsync.
  void upsert(const FloatArray& data, const py::object& labels, std::size_t threads) {
    const std::size_t dim = db_->index().params().dim;
    std::size_t n;
    const float* rows = rows_of(data, dim, n);
    const auto ids = labels_of(labels, n, db_->index().size());
    py::gil_scoped_release release;
    if (n == 1) {
      db_->upsert(ids[0], rows);
    } else {
      db_->upsert_batch(ids.data(), rows, n, threads);
    }
  }

  bool remove(std::uint64_t label) {
    py::gil_scoped_release release;
    return db_->remove(label).has_value();
  }

  py::tuple search(const FloatArray& queries, std::size_t k, std::size_t ef, std::size_t rerank, std::size_t threads) const {
    return batch_search(queries, db_->index().params().dim, k, threads, [&](const float* q, std::size_t kk) {
      return db_->search(q, kk, {.ef = ef, .rerank = rerank});
    });
  }

  void flush() {
    py::gil_scoped_release release;
    db_->flush();
  }

  void checkpoint() {
    py::gil_scoped_release release;
    db_->checkpoint();
  }

  py::dict recovery() const {
    const auto& r = db_->recovery();
    py::dict out;
    out["loaded_snapshot"] = r.loaded_snapshot;
    out["snapshot_lsn"] = r.snapshot_lsn;
    out["replayed"] = r.replayed;
    out["truncated_bytes"] = r.truncated_bytes;
    return out;
  }

  const epzero::Database& raw() const { return *db_; }

 private:
  std::unique_ptr<epzero::Database> db_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Epzero: HNSW vector search engine";
  m.def("simd_level", [] { return std::string(epzero::to_string(epzero::active_simd_level())); },
        "SIMD instruction set used by the distance kernels on this CPU");
  m.def("machine", &epzero::machine_name,
        "This CPU's model and SIMD level: what machine-specific settings (prefetching) are measured on.");

  m.def(
      "_plan_auto",
      [](const FloatArray& data, const std::string& metric, std::size_t M, std::size_t ef_construction, std::size_t k,
         float recall, std::size_t threads, std::size_t capacity, std::uint64_t seed, bool with_queries,
         const py::dict& fixed) {
        if (data.ndim() != 2 || data.shape(0) == 0 || data.shape(1) == 0) {
          throw std::invalid_argument("data must be a non-empty 2-D array (rows of vectors)");
        }
        auto options = auto_options(metric, M, ef_construction, k, recall, threads, capacity, seed, fixed);
        static const float marker = 0.0f;  // only whether queries were given changes the plan
        if (with_queries) {
          options.queries = &marker;
          options.query_count = 1;
        }
        epzero::AutoPlan plan;
        {
          py::gil_scoped_release release;
          plan = epzero::plan_auto(data.data(), static_cast<std::size_t>(data.shape(0)),
                                   static_cast<std::size_t>(data.shape(1)), options);
        }
        py::dict out;
        out["params"] = params_dict(plan.params);
        out["profile"] = profile_dict(plan.profile);
        out["calibrate"] = plan.calibrate;
        out["tune"] = plan.tune;
        return out;
      },
      py::arg("data"), py::arg("metric"), py::arg("M"), py::arg("ef_construction"), py::arg("k"), py::arg("recall"),
      py::arg("threads"), py::arg("capacity"), py::arg("seed"), py::arg("with_queries"), py::arg("fixed"));
  m.def(
      "_build_auto",
      [](const FloatArray& data, const py::object& labels, const std::string& metric, std::size_t M,
         std::size_t ef_construction, std::size_t k, float recall, std::size_t threads, std::size_t capacity,
         std::uint64_t seed, const py::object& queries, const py::dict& fixed) {
        if (data.ndim() != 2 || data.shape(0) == 0 || data.shape(1) == 0) {
          throw std::invalid_argument("data must be a non-empty 2-D array (rows of vectors)");
        }
        const auto n = static_cast<std::size_t>(data.shape(0));
        const auto dim = static_cast<std::size_t>(data.shape(1));
        const auto ids = labels_of(labels, n, 0);
        auto options = auto_options(metric, M, ef_construction, k, recall, threads, capacity, seed, fixed);
        FloatArray query_rows;
        if (!queries.is_none()) {
          query_rows = queries.cast<FloatArray>();
          options.queries = rows_of(query_rows, dim, options.query_count);
        }
        std::unique_ptr<epzero::HnswIndex> built;
        {
          py::gil_scoped_release release;
          built = epzero::build_auto(data.data(), ids.data(), n, dim, options);
        }
        return Index(std::move(built));
      },
      py::arg("data"), py::arg("labels"), py::arg("metric"), py::arg("M"), py::arg("ef_construction"), py::arg("k"),
      py::arg("recall"), py::arg("threads"), py::arg("capacity"), py::arg("seed"), py::arg("queries"),
      py::arg("fixed"));

  py::class_<epzero::SearchFilter>(m, "Filter",
                                 "The labels a search may return, compiled by Index.filter() for reuse across queries.")
      .def_property_readonly("allowed", &epzero::SearchFilter::allowed, "How many live vectors pass.")
      .def_property_readonly("selectivity", &epzero::SearchFilter::selectivity, "The share of live vectors that pass.")
      .def("__repr__", [](const epzero::SearchFilter& f) {
        return "<epzero.Filter allowing " + std::to_string(f.allowed()) + " vectors (" +
               std::to_string(f.selectivity() * 100.0).substr(0, 5) + "%)>";
      });

  py::class_<Index>(m, "Index")
      .def(py::init<std::size_t, const std::string&, std::size_t, std::size_t, std::size_t, std::uint64_t,
                    const std::string&, bool, std::size_t, bool, bool, bool>(),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("max_elements") = 1'000'000, py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("seed") = 42, py::arg("quantization") = "none",
           py::arg("keep_vectors") = true, py::arg("nav_dim") = 0, py::arg("fill_links") = false,
           py::arg("fast_build") = false, py::arg("half_vectors") = false)
      .def("train", &Index::train, py::arg("data"), "SQ8: learn per-dimension ranges before the first add().")
      .def("add", &Index::add, py::arg("data"), py::arg("labels") = py::none(), py::arg("threads") = 0,
           "Insert rows of `data`; labels default to consecutive ids starting at the number of slots used.")
      .def("remove", &Index::remove, py::arg("label"), "Tombstone a label; returns False if it is absent.")
      .def("reserve", &Index::reserve, py::arg("capacity"),
           "Grows the capacity to `capacity` vectors, copying the nodes and keeping the graph. add() does this\n"
           "by itself when rows don't fit.")
      .def_property_readonly("capacity", &Index::capacity, "How many vectors fit before the index grows.")
      .def("search", &Index::search, py::arg("queries"), py::arg("k") = py::none(), py::arg("ef") = py::none(),
           py::arg("rerank") = py::none(), py::arg("threads") = 1, py::arg("filter") = py::none(),
           py::arg("strategy") = "auto", py::arg("patience") = 0, py::arg("rerank_z") = py::none(),
           "Returns (labels, distances), each of shape (n, k). `k`, `ef`, `rerank` and `rerank_z` default to\n"
           "the index's profile: k 10, ef 64 and a re-rank of 2 x k for an index built by hand, the\n"
           "calibrated settings for one from epzero.build(). With `filter` (an epzero.Filter or a list of\n"
           "allowed labels), only those labels are returned; `strategy` forces 'exact', 'inline' or\n"
           "'two-hop' instead of choosing by selectivity. With `patience`, each unfiltered query stops once\n"
           "its best k have gone that many steps unchanged (`ef` then caps the beam): easy queries finish\n"
           "early, hard ones search further.")
      .def_property_readonly("prefetch", &Index::prefetch, "(depth, lines): how far ahead the search prefetches.")
      .def_property_readonly("profile", &Index::profile,
                             "The search settings this index uses by default, how it was built, and why\n"
                             "(saved with it). Set by epzero.build().")
      .def_property_readonly("params", &Index::params, "The settings the index was created with.")
      .def("recalibrate", &Index::recalibrate, py::arg("queries"), py::arg("recall") = py::none(), py::arg("threads") = 0,
           "Re-measures the beam width that reaches `recall` (default: the profile's target) on `queries`\n"
           "and saves it in the profile, which search() then uses. Pass a few hundred real queries: rows\n"
           "already in the index make the search look easier than it is. Returns {'ef', 'recall'}.")
      .def("health", &Index::health, py::arg("threads") = 0,
           "What the index needs since it was built: its size and tombstones, growth since calibration,\n"
           "how much of the data its principal directions still capture, and 'advice' in plain words\n"
           "(empty when nothing needs doing).")
      .def("check_queries", &Index::check_queries, py::arg("queries"), py::arg("threads") = 0,
           "Whether `queries` look like the data: the median distance from a query to its nearest neighbour,\n"
           "against the same for rows of the data. Text queries against image vectors sit far farther, and then\n"
           "recall calibrated on the data overestimates theirs: 'advice' says to recalibrate(queries).")
      .def("get_vector", &Index::get_vector, py::arg("label"),
           "The stored vector of `label` (normalized for cosine). KeyError if it isn't in the index.")
      .def("set_prefetch", &Index::set_prefetch, py::arg("depth") = py::none(), py::arg("lines") = 1,
           "Prefetch `lines` cache lines of `depth` neighbors ahead (None: every neighbor up front).")
      .def("refine_links", &Index::refine_links, py::arg("threads") = 0, py::arg("ef") = 0,
           "Once built: every node searches the finished graph and reconnects (ef 0: ef_construction).")
      .def("fill_free_links", &Index::fill_free_links, py::arg("threads") = 0,
           "Once built: gives each node's free layer-0 link slots to its closest two-hop neighbours.\n"
           "Returns the number of links added.")
      .def("tune_prefetch", &Index::tune_prefetch, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64,
           "Times searches of `queries` (a few hundred rows, e.g. sampled from the data) at a grid of\n"
           "prefetch settings and keeps the fastest if it beats the default by 3%. Returns the setting.")
      .def("filter", &Index::make_filter, py::arg("labels"),
           "Compiles the allowed labels into an epzero.Filter, reusable across searches.")
      .def("exact_search", &Index::exact_search, py::arg("queries"), py::arg("k") = 10, py::arg("threads") = 1,
           py::arg("filter") = py::none(), "Exact (labels, distances) by scanning every live vector: the ground truth.")
      .def("explain", &Index::explain, py::arg("query"), py::arg("k"), py::arg("label"), py::arg("ef") = py::none(),
           py::arg("filter") = py::none(),
           "Why `label` is or isn't among search(query, k, ef)'s results. 'cause' is one of returned,\n"
           "not-in-index, filtered-out, not-near-enough (the search was right) or search-missed, with\n"
           "the label's exact rank, whether the walk scored it, and the beam width that finds it.")
      .def("trace", &Index::trace, py::arg("query"), py::arg("k") = 10, py::arg("ef") = py::none())
      .def("compact", &Index::compact, py::arg("threads") = 0, "A new index without deleted vectors.")
      .def("save", &Index::save, py::arg("path"))
      .def_static("load", &Index::load, py::arg("path"))
      .def("level", [](const Index& self, std::uint64_t label) { return self.raw().level_of(label); })
      .def("neighbors", [](const Index& self, std::uint64_t label, int layer) { return self.raw().neighbors_of(label, layer); },
           py::arg("label"), py::arg("layer") = 0)
      .def("__contains__", [](const Index& self, std::uint64_t label) { return self.raw().contains(label); })
      .def("__len__", [](const Index& self) { return self.raw().live_count(); })
      .def("__repr__", [](const Index& self) {
        const auto& index = self.raw();
        const auto& p = index.params();
        const auto& profile = index.profile();
        std::string out = "<epzero.Index: " + grouped(index.live_count()) + " vectors of " + std::to_string(p.dim) + " dimensions, " +
                          std::string(epzero::to_string(p.metric)) + ", " + quantization_name(p.quantization);
        if (p.nav_dim != 0) out += ", walk on " + std::to_string(p.nav_dim) + " directions";
        if (p.half_vectors) out += ", 16-bit vectors";
        out += ", ef " + std::to_string(profile.ef);
        if (profile.recall_target > 0.0f) out += " (calibrated for recall " + std::to_string(profile.recall_target).substr(0, 4) + ")";
        return out + ", capacity " + grouped(p.max_elements) + ">";
      })
      .def_property_readonly("dim", [](const Index& self) { return self.raw().params().dim; })
      .def_property_readonly("max_level", [](const Index& self) { return self.raw().max_level(); })
      .def_property_readonly("deleted_count", [](const Index& self) { return self.raw().deleted_count(); })
      .def_property_readonly("memory_bytes", [](const Index& self) { return self.raw().memory_bytes(); })
      .def_property_readonly("simd", [](const Index& self) { return std::string(epzero::to_string(self.raw().simd_level())); });

  py::class_<Guard>(m, "RecallGuard",
                    "Searches an Index at the smallest beam width that keeps live, measured recall above a\n"
                    "target. A sample of queries is re-answered exactly in the background.")
      .def(py::init<const Index&, const py::object&, const py::object&, std::size_t, std::size_t, std::size_t, double,
                    double, std::size_t, std::size_t, bool>(),
           py::arg("index"), py::arg("target") = py::none(), py::arg("initial_ef") = py::none(), py::arg("min_ef") = 16,
           py::arg("max_ef") = 2048, py::arg("min_samples") = 200, py::arg("margin") = 0.01,
           py::arg("sample_rate") = 0.01, py::arg("window") = 1000, py::arg("reference_ef") = 0,
           py::arg("per_query") = false,
           py::keep_alive<1, 2>())  // the guard holds the index
      .def("search", &Guard::search, py::arg("queries"), py::arg("k") = 10, py::arg("threads") = 1,
           py::arg("filter") = py::none(), "Returns (labels, distances), like Index.search.")
      .def("report", &Guard::report,
           "Per segment: its beam width, how often it moved, and the recall estimate with its interval.")
      .def("drain", &Guard::drain, "Waits for the queued exact checks to finish.");

  py::class_<Metadata>(m, "Metadata",
                       "Fields per label (str, int, float, bool) for filtered search: select(where) returns the\n"
                       "labels that pass, to hand to Index.search(filter=...) or Index.filter().")
      .def(py::init<>())
      .def("set", &Metadata::set, py::arg("label"), py::arg("fields"),
           "Sets `label`'s fields from a dict, replacing any it had.")
      .def("set_many", &Metadata::set_many, py::arg("labels"), py::arg("rows"),
           "Sets many labels' fields: rows is a list of dicts, one per label (labels None: 0..n-1).")
      .def("remove", &Metadata::remove, py::arg("label"), "Forgets `label`; returns False if it had no fields.")
      .def("select", &Metadata::select, py::arg("where"),
           "Labels whose fields meet every entry of `where`, sorted: {field: value} for equality,\n"
           "{field: [values]} for any of them, or {field: {op: x}} with $eq, $ne, $in, $nin, $gt, $gte,\n"
           "$lt, $lte or $exists. Integers and floats compare as numbers.")
      .def("get", &Metadata::get, py::arg("label"), "`label`'s fields as a dict ({} if none).")
      .def("save", &Metadata::save, py::arg("path"))
      .def_static("load", &Metadata::load, py::arg("path"))
      .def("__len__", &Metadata::size)
      .def("__repr__", [](const Metadata& self) { return "<epzero.Metadata: " + grouped(self.size()) + " labels>"; })
      .def("__contains__", &Metadata::contains);

  py::class_<TextIndex>(m, "TextIndex",
                        "Keyword search (Okapi BM25) over short texts, keyed by the same labels as an Index:\n"
                        "the keyword half of hybrid_search().")
      .def(py::init<float, float>(), py::arg("k1") = 1.2f, py::arg("b") = 0.75f)
      .def("add", &TextIndex::add, py::arg("labels"), py::arg("texts"),
           "Index `texts` under `labels` (None: 0..n-1), replacing any text a label had.")
      .def("remove", &TextIndex::remove, py::arg("label"), "Forget a label; returns False if it is absent.")
      .def("search", &TextIndex::search, py::arg("query"), py::arg("k") = 10,
           "Returns (labels, scores), best first, by BM25 score. Texts sharing no word with the query are omitted.")
      .def("save", &TextIndex::save, py::arg("path"))
      .def_static("load", &TextIndex::load, py::arg("path"))
      .def_static("tokenize", &epzero::TextIndex::tokenize, py::arg("text"), "The words a text is indexed by.")
      .def("__len__", [](const TextIndex& self) { return self.raw().size(); })
      .def("__repr__", [](const TextIndex& self) { return "<epzero.TextIndex: " + grouped(self.raw().size()) + " documents>"; })
      .def("__contains__", [](const TextIndex& self, std::uint64_t label) { return self.raw().contains(label); });

  m.def(
      "hybrid_search",
      [](const Index& index, const TextIndex& text, const FloatArray& vector, const std::string& query, std::size_t k,
         std::size_t ef, std::size_t candidates, double rrf_k, double vector_weight, double text_weight) {
        std::size_t n;
        const float* row = rows_of(vector, index.raw().params().dim, n);
        if (n != 1) throw std::invalid_argument("hybrid_search takes one query vector");
        std::vector<epzero::SearchResult> results;
        {
          py::gil_scoped_release release;
          results = epzero::hybrid_search(index.raw(), text.raw(), row, query, k,
                                        {.ef = ef, .candidates = candidates, .rrf_k = rrf_k,
                                         .vector_weight = vector_weight, .text_weight = text_weight});
        }
        return ranked(results);
      },
      py::arg("index"), py::arg("text_index"), py::arg("vector"), py::arg("text"), py::arg("k") = 10, py::arg("ef") = 64,
      py::arg("candidates") = 50, py::arg("rrf_k") = 60.0, py::arg("vector_weight") = 1.0, py::arg("text_weight") = 1.0,
      "The k best labels for a query given as a vector and as text: a vector search and a keyword search,\n"
      "each contributing `candidates` results, merged by reciprocal rank fusion. Returns (labels, scores).");
  m.def(
      "fuse_rankings",
      [](const std::vector<std::vector<std::int64_t>>& rankings, std::size_t k, double rrf_k,
         const std::vector<double>& weights) {
        std::vector<std::vector<epzero::SearchResult>> lists;
        for (const auto& ranking : rankings) {
          auto& list = lists.emplace_back();
          for (const auto label : ranking) {
            if (label < 0) throw std::invalid_argument("labels must be non-negative");
            list.push_back({static_cast<std::uint64_t>(label), 0.0f});
          }
        }
        return ranked(epzero::fuse_rankings(lists, k, rrf_k, weights));
      },
      py::arg("rankings"), py::arg("k") = 10, py::arg("rrf_k") = 60.0, py::arg("weights") = std::vector<double>{},
      "Reciprocal rank fusion of ranked label lists (each best first), e.g. from another search engine.\n"
      "Returns (labels, scores) of the k best.");

  py::class_<Database>(m, "Database", "A durable collection: HNSW index + write-ahead log with crash recovery.")
      .def_static("open", &Database::open, py::arg("path"), py::arg("dim"), py::arg("metric") = "l2",
                  py::arg("max_elements") = 1'000'000, py::arg("M") = 16, py::arg("ef_construction") = 200,
                  py::arg("quantization") = "none", py::arg("keep_vectors") = true, py::arg("sync") = "always",
                  py::arg("checkpoint_mb") = 256.0,
                  "Open (or create) a database directory; an existing snapshot's parameters take precedence.")
      .def("train", &Database::train, py::arg("data"))
      .def("upsert", &Database::upsert, py::arg("data"), py::arg("labels") = py::none(), py::arg("threads") = 0)
      .def("remove", &Database::remove, py::arg("label"))
      .def("search", &Database::search, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64,
           py::arg("rerank") = 0, py::arg("threads") = 1)
      .def("flush", &Database::flush, "Make every write so far durable (group commit).")
      .def("checkpoint", &Database::checkpoint, "Snapshot the index and truncate the log.")
      .def("recovery", &Database::recovery, "What the last open() recovered.")
      .def("__contains__", [](const Database& self, std::uint64_t label) { return self.raw().index().contains(label); })
      .def("__len__", [](const Database& self) { return self.raw().size(); })
      .def("__repr__", [](const Database& self) {
        return "<epzero.Database: " + grouped(self.raw().size()) + " vectors, last log record " + std::to_string(self.raw().last_lsn()) + ">";
      })
      .def_property_readonly("last_lsn", [](const Database& self) { return self.raw().last_lsn(); });
}
