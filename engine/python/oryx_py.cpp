// Python bindings. Batch calls release the GIL and fan out across C++ threads, so Python only
// pays call overhead once per batch.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>

#include "oryx/database.h"
#include "oryx/hnsw.h"
#include "oryx/parallel.h"
#include "oryx/recall.h"
#include "oryx/text.h"

namespace py = pybind11;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using LabelArray = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;

namespace {

oryx::Metric parse_metric(const std::string& name) {
  if (name == "l2") return oryx::Metric::L2;
  if (name == "ip") return oryx::Metric::InnerProduct;
  if (name == "cosine") return oryx::Metric::Cosine;
  throw std::invalid_argument("metric must be 'l2', 'ip' or 'cosine'");
}

oryx::Quantization parse_quantization(const std::string& name) {
  if (name == "none") return oryx::Quantization::None;
  if (name == "sq8") return oryx::Quantization::SQ8;
  if (name == "rabitq") return oryx::Quantization::RaBitQ;
  throw std::invalid_argument("quantization must be 'none', 'sq8' or 'rabitq'");
}

oryx::FilterStrategy parse_strategy(const std::string& name) {
  for (const auto s : {oryx::FilterStrategy::Auto, oryx::FilterStrategy::Exact, oryx::FilterStrategy::Inline,
                       oryx::FilterStrategy::TwoHop}) {
    if (name == oryx::to_string(s)) return s;
  }
  throw std::invalid_argument("strategy must be 'auto', 'exact', 'inline' or 'two-hop'");
}

oryx::SyncMode parse_sync(const std::string& name) {
  if (name == "always") return oryx::SyncMode::Always;
  if (name == "batch") return oryx::SyncMode::Batch;
  if (name == "none") return oryx::SyncMode::None;
  throw std::invalid_argument("sync must be 'always', 'batch' or 'none'");
}

oryx::HnswParams make_params(std::size_t dim, const std::string& metric, std::size_t max_elements, std::size_t M,
                             std::size_t ef_construction, std::uint64_t seed, const std::string& quantization,
                             bool keep_vectors) {
  return {.dim = dim, .metric = parse_metric(metric), .max_elements = max_elements, .M = M,
          .ef_construction = ef_construction, .seed = seed, .quantization = parse_quantization(quantization),
          .keep_vectors = keep_vectors};
}

const float* rows_of(const FloatArray& data, std::size_t dim, std::size_t& n) {
  if (data.ndim() == 1 && static_cast<std::size_t>(data.shape(0)) == dim) {
    n = 1;
  } else if (data.ndim() == 2 && static_cast<std::size_t>(data.shape(1)) == dim) {
    n = static_cast<std::size_t>(data.shape(0));
  } else {
    throw std::invalid_argument("expected an array of shape (n, " + std::to_string(dim) + ")");
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
    oryx::parallel_for(0, n, threads, [&](std::size_t q) {
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

class Index {
 public:
  Index(std::size_t dim, const std::string& metric, std::size_t max_elements, std::size_t M,
        std::size_t ef_construction, std::uint64_t seed, const std::string& quantization, bool keep_vectors)
      : index_(std::make_unique<oryx::HnswIndex>(
            make_params(dim, metric, max_elements, M, ef_construction, seed, quantization, keep_vectors))) {}

  explicit Index(std::unique_ptr<oryx::HnswIndex> index) : index_(std::move(index)) {}

  void train(const FloatArray& data) {
    std::size_t n;
    const float* rows = rows_of(data, index_->params().dim, n);
    py::gil_scoped_release release;
    index_->train(rows, n);
  }

  void add(const FloatArray& data, const py::object& labels, std::size_t threads) {
    const std::size_t dim = index_->params().dim;
    std::size_t n;
    const float* rows = rows_of(data, dim, n);
    const auto ids = labels_of(labels, n, index_->size());
    py::gil_scoped_release release;
    oryx::parallel_for(0, n, threads, [&](std::size_t i) { index_->add(ids[i], rows + i * dim); });
  }

  bool remove(std::uint64_t label) { return index_->remove(label); }

  // `filter` is None, an oryxdb.Filter from filter(), or a sequence of allowed labels (compiled
  // for this call; build an oryxdb.Filter once to reuse it across calls).
  // A filter argument: None, an oryxdb.Filter, or a sequence of labels compiled into `storage`.
  const oryx::SearchFilter* resolve_filter(const py::object& filter, oryx::SearchFilter& storage) const {
    if (filter.is_none()) return nullptr;
    if (py::isinstance<oryx::SearchFilter>(filter)) return filter.cast<const oryx::SearchFilter*>();
    storage = make_filter(filter.cast<LabelArray>());
    return &storage;
  }

  py::tuple search(const FloatArray& queries, std::size_t k, std::size_t ef, std::size_t rerank, std::size_t threads,
                   const py::object& filter, const std::string& strategy, std::size_t patience) const {
    const oryx::FilterStrategy chosen = parse_strategy(strategy);
    oryx::SearchFilter compiled;
    const oryx::SearchFilter* active = resolve_filter(filter, compiled);
    return batch_search(queries, index_->params().dim, k, threads, [&](const float* q, std::size_t kk) {
      return index_->search(q, kk, {.ef = ef, .patience = patience, .rerank = rerank, .filter = active, .strategy = chosen});
    });
  }

  // Prefetch settings: depth None means every neighbor up front (the default).
  py::tuple prefetch() const {
    const auto [depth, lines] = index_->prefetch_settings();
    return py::make_tuple(depth == std::numeric_limits<std::size_t>::max() ? py::object(py::none()) : py::int_(depth), lines);
  }
  void set_prefetch(const py::object& depth, std::size_t lines) {
    index_->set_prefetch(depth.is_none() ? std::numeric_limits<std::size_t>::max() : depth.cast<std::size_t>(), lines);
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
    oryx::SearchFilter compiled;
    const oryx::SearchFilter* active = resolve_filter(filter, compiled);
    return batch_search(queries, index_->params().dim, k, threads, [&](const float* q, std::size_t kk) {
      return index_->exact_search(q, kk, active);
    });
  }

  py::dict explain(const FloatArray& query, std::size_t k, std::uint64_t label, std::size_t ef,
                   const py::object& filter) const {
    std::size_t n;
    const float* row = rows_of(query, index_->params().dim, n);
    if (n != 1) throw std::invalid_argument("explain takes exactly one query");
    oryx::SearchFilter compiled;
    const oryx::SearchFilter* active = resolve_filter(filter, compiled);
    oryx::MissExplanation why;
    {
      py::gil_scoped_release release;
      why = index_->explain(row, k, label, {.ef = ef, .filter = active});
    }
    py::dict out;
    out["cause"] = std::string(oryx::to_string(why.cause));
    out["exact_rank"] = why.exact_rank;
    out["distance"] = why.distance;
    out["kth_distance"] = why.kth_distance;
    out["scored"] = why.scored;
    out["ef_needed"] = why.ef_needed;
    return out;
  }

  oryx::SearchFilter make_filter(const LabelArray& labels) const {
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
  py::dict trace(const FloatArray& query, std::size_t k, std::size_t ef) const {
    std::size_t n;
    const float* row = rows_of(query, index_->params().dim, n);
    if (n != 1) throw std::invalid_argument("trace takes exactly one query");
    oryx::SearchTrace trace;
    oryx::SearchStats stats;
    const auto results = index_->search(row, k, {.ef = ef, .stats = &stats, .trace = &trace});

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
    return Index(oryx::HnswIndex::load(path));
  }

  const oryx::HnswIndex& raw() const { return *index_; }

 private:
  std::unique_ptr<oryx::HnswIndex> index_;
};

py::dict estimate_to_dict(const oryx::RecallEstimate& e) {
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
  Guard(const Index& index, double target, std::size_t initial_ef, std::size_t min_ef, std::size_t max_ef,
        std::size_t min_samples, double margin, double sample_rate, std::size_t window, std::size_t reference_ef,
        bool per_query)
      : index_(index),
        guard_(index.raw(), {.target = target, .initial_ef = initial_ef, .min_ef = min_ef, .max_ef = max_ef,
                             .min_samples = min_samples, .margin = margin, .per_query = per_query,
                             .monitor = {.sample_rate = sample_rate, .window = window, .reference_ef = reference_ef}}) {}

  py::tuple search(const FloatArray& queries, std::size_t k, std::size_t threads, const py::object& filter) {
    oryx::SearchFilter compiled;
    const oryx::SearchFilter* active = nullptr;
    if (py::isinstance<oryx::SearchFilter>(filter)) {
      active = filter.cast<const oryx::SearchFilter*>();
    } else if (!filter.is_none()) {
      compiled = index_.make_filter(filter.cast<LabelArray>());
      active = &compiled;
    }
    return batch_search(queries, index_.raw().params().dim, k, threads,
                        [&](const float* q, std::size_t kk) { return guard_.search(q, kk, active); });
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
  oryx::RecallGuard guard_;
};

// Ranked results as (labels, scores) arrays, scores larger-is-better (the negated distances).
py::tuple ranked(const std::vector<oryx::SearchResult>& results) {
  py::array_t<std::int64_t> labels(static_cast<py::ssize_t>(results.size()));
  py::array_t<float> scores(static_cast<py::ssize_t>(results.size()));
  for (std::size_t i = 0; i < results.size(); ++i) {
    labels.mutable_data()[i] = static_cast<std::int64_t>(results[i].label);
    scores.mutable_data()[i] = -results[i].distance;
  }
  return py::make_tuple(labels, scores);
}

class TextIndex {
 public:
  TextIndex(float k1, float b) : index_({.k1 = k1, .b = b}) {}
  explicit TextIndex(oryx::TextIndex index) : index_(std::move(index)) {}

  void add(const py::object& labels, const std::vector<std::string>& texts) {
    const auto ids = labels_of(labels, texts.size(), 0);
    py::gil_scoped_release release;
    for (std::size_t i = 0; i < texts.size(); ++i) index_.add(ids[i], texts[i]);
  }
  bool remove(std::uint64_t label) { return index_.remove(label); }
  py::tuple search(const std::string& query, std::size_t k) const {
    std::vector<oryx::SearchResult> results;
    {
      py::gil_scoped_release release;
      results = index_.search(query, k);
    }
    return ranked(results);
  }
  void save(const std::string& path) const { index_.save(path); }
  static TextIndex load(const std::string& path) { return TextIndex(oryx::TextIndex::load(path)); }
  const oryx::TextIndex& raw() const noexcept { return index_; }

 private:
  oryx::TextIndex index_;
};

class Database {
 public:
  explicit Database(std::unique_ptr<oryx::Database> db) : db_(std::move(db)) {}

  static Database open(const std::string& path, std::size_t dim, const std::string& metric, std::size_t max_elements,
                       std::size_t M, std::size_t ef_construction, const std::string& quantization, bool keep_vectors,
                       const std::string& sync, double checkpoint_mb) {
    oryx::DatabaseOptions options;
    options.sync = parse_sync(sync);
    options.checkpoint_bytes = static_cast<std::size_t>(checkpoint_mb * 1024 * 1024);
    const auto params = make_params(dim, metric, max_elements, M, ef_construction, 42, quantization, keep_vectors);
    py::gil_scoped_release release;
    return Database(oryx::Database::open(path, params, options));
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

  const oryx::Database& raw() const { return *db_; }

 private:
  std::unique_ptr<oryx::Database> db_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "Oryx: HNSW vector search engine";
  m.def("simd_level", [] { return std::string(oryx::to_string(oryx::active_simd_level())); },
        "SIMD instruction set used by the distance kernels on this CPU");

  py::class_<oryx::SearchFilter>(m, "Filter",
                                 "The labels a search may return, compiled by Index.filter() for reuse across queries.")
      .def_property_readonly("allowed", &oryx::SearchFilter::allowed, "How many live vectors pass.")
      .def_property_readonly("selectivity", &oryx::SearchFilter::selectivity, "The share of live vectors that pass.")
      .def("__repr__", [](const oryx::SearchFilter& f) {
        return "<oryxdb.Filter allowing " + std::to_string(f.allowed()) + " vectors (" +
               std::to_string(f.selectivity() * 100.0).substr(0, 5) + "%)>";
      });

  py::class_<Index>(m, "Index")
      .def(py::init<std::size_t, const std::string&, std::size_t, std::size_t, std::size_t, std::uint64_t,
                    const std::string&, bool>(),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("max_elements") = 1'000'000, py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("seed") = 42, py::arg("quantization") = "none",
           py::arg("keep_vectors") = true)
      .def("train", &Index::train, py::arg("data"), "SQ8: learn per-dimension ranges before the first add().")
      .def("add", &Index::add, py::arg("data"), py::arg("labels") = py::none(), py::arg("threads") = 0,
           "Insert rows of `data`; labels default to consecutive ids starting at the number of slots used.")
      .def("remove", &Index::remove, py::arg("label"), "Tombstone a label; returns False if it is absent.")
      .def("search", &Index::search, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64,
           py::arg("rerank") = 0, py::arg("threads") = 1, py::arg("filter") = py::none(), py::arg("strategy") = "auto",
           py::arg("patience") = 0,
           "Returns (labels, distances), each of shape (n, k). With `filter` (an oryxdb.Filter or a list of\n"
           "allowed labels), only those labels are returned; `strategy` forces 'exact', 'inline' or\n"
           "'two-hop' instead of choosing by selectivity. With `patience`, each unfiltered query stops once\n"
           "its best k have gone that many steps unchanged (`ef` then caps the beam): easy queries finish\n"
           "early, hard ones search further.")
      .def_property_readonly("prefetch", &Index::prefetch, "(depth, lines): how far ahead the search prefetches.")
      .def("set_prefetch", &Index::set_prefetch, py::arg("depth") = py::none(), py::arg("lines") = 1,
           "Prefetch `lines` cache lines of `depth` neighbors ahead (None: every neighbor up front).")
      .def("tune_prefetch", &Index::tune_prefetch, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64,
           "Times searches of `queries` (a few hundred rows, e.g. sampled from the data) at a grid of\n"
           "prefetch settings and keeps the fastest if it beats the default by 3%. Returns the setting.")
      .def("filter", &Index::make_filter, py::arg("labels"),
           "Compiles the allowed labels into an oryxdb.Filter, reusable across searches.")
      .def("exact_search", &Index::exact_search, py::arg("queries"), py::arg("k") = 10, py::arg("threads") = 1,
           py::arg("filter") = py::none(), "Exact (labels, distances) by scanning every live vector: the ground truth.")
      .def("explain", &Index::explain, py::arg("query"), py::arg("k"), py::arg("label"), py::arg("ef") = 64,
           py::arg("filter") = py::none(),
           "Why `label` is or isn't among search(query, k, ef)'s results. 'cause' is one of returned,\n"
           "not-in-index, filtered-out, not-near-enough (the search was right) or search-missed, with\n"
           "the label's exact rank, whether the walk scored it, and the beam width that finds it.")
      .def("trace", &Index::trace, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 64)
      .def("compact", &Index::compact, py::arg("threads") = 0, "A new index without deleted vectors.")
      .def("save", &Index::save, py::arg("path"))
      .def_static("load", &Index::load, py::arg("path"))
      .def("level", [](const Index& self, std::uint64_t label) { return self.raw().level_of(label); })
      .def("neighbors", [](const Index& self, std::uint64_t label, int layer) { return self.raw().neighbors_of(label, layer); },
           py::arg("label"), py::arg("layer") = 0)
      .def("__contains__", [](const Index& self, std::uint64_t label) { return self.raw().contains(label); })
      .def("__len__", [](const Index& self) { return self.raw().live_count(); })
      .def_property_readonly("dim", [](const Index& self) { return self.raw().params().dim; })
      .def_property_readonly("max_level", [](const Index& self) { return self.raw().max_level(); })
      .def_property_readonly("deleted_count", [](const Index& self) { return self.raw().deleted_count(); })
      .def_property_readonly("memory_bytes", [](const Index& self) { return self.raw().memory_bytes(); })
      .def_property_readonly("simd", [](const Index& self) { return std::string(oryx::to_string(self.raw().simd_level())); });

  py::class_<Guard>(m, "RecallGuard",
                    "Searches an Index at the smallest beam width that keeps live, measured recall above a\n"
                    "target. A sample of queries is re-answered exactly in the background.")
      .def(py::init<const Index&, double, std::size_t, std::size_t, std::size_t, std::size_t, double, double, std::size_t,
                    std::size_t, bool>(),
           py::arg("index"), py::arg("target") = 0.95, py::arg("initial_ef") = 64, py::arg("min_ef") = 16,
           py::arg("max_ef") = 2048, py::arg("min_samples") = 200, py::arg("margin") = 0.01,
           py::arg("sample_rate") = 0.01, py::arg("window") = 1000, py::arg("reference_ef") = 0,
           py::arg("per_query") = false,
           py::keep_alive<1, 2>())  // the guard holds the index
      .def("search", &Guard::search, py::arg("queries"), py::arg("k") = 10, py::arg("threads") = 1,
           py::arg("filter") = py::none(), "Returns (labels, distances), like Index.search.")
      .def("report", &Guard::report,
           "Per segment: its beam width, how often it moved, and the recall estimate with its interval.")
      .def("drain", &Guard::drain, "Waits for the queued exact checks to finish.");

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
      .def_static("tokenize", &oryx::TextIndex::tokenize, py::arg("text"), "The words a text is indexed by.")
      .def("__len__", [](const TextIndex& self) { return self.raw().size(); })
      .def("__contains__", [](const TextIndex& self, std::uint64_t label) { return self.raw().contains(label); });

  m.def(
      "hybrid_search",
      [](const Index& index, const TextIndex& text, const FloatArray& vector, const std::string& query, std::size_t k,
         std::size_t ef, std::size_t candidates, double rrf_k, double vector_weight, double text_weight) {
        std::size_t n;
        const float* row = rows_of(vector, index.raw().params().dim, n);
        if (n != 1) throw std::invalid_argument("hybrid_search takes one query vector");
        std::vector<oryx::SearchResult> results;
        {
          py::gil_scoped_release release;
          results = oryx::hybrid_search(index.raw(), text.raw(), row, query, k,
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
        std::vector<std::vector<oryx::SearchResult>> lists;
        for (const auto& ranking : rankings) {
          auto& list = lists.emplace_back();
          for (const auto label : ranking) {
            if (label < 0) throw std::invalid_argument("labels must be non-negative");
            list.push_back({static_cast<std::uint64_t>(label), 0.0f});
          }
        }
        return ranked(oryx::fuse_rankings(lists, k, rrf_k, weights));
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
      .def_property_readonly("last_lsn", [](const Database& self) { return self.raw().last_lsn(); });
}
