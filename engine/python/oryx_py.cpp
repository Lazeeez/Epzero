// Python bindings. Batch calls release the GIL and fan out across C++ threads, so Python only
// pays call overhead once per batch.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>

#include "oryx/hnsw.h"
#include "oryx/parallel.h"

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

class Index {
 public:
  Index(std::size_t dim, const std::string& metric, std::size_t max_elements, std::size_t M,
        std::size_t ef_construction, std::uint64_t seed)
      : index_(std::make_unique<oryx::HnswIndex>(oryx::HnswParams{
            .dim = dim, .metric = parse_metric(metric), .max_elements = max_elements, .M = M,
            .ef_construction = ef_construction, .seed = seed})) {}

  explicit Index(std::unique_ptr<oryx::HnswIndex> index) : index_(std::move(index)) {}

  void add(const FloatArray& data, const py::object& labels, std::size_t threads) {
    const std::size_t dim = index_->params().dim;
    std::size_t n;
    const float* rows = rows_of(data, dim, n);
    std::vector<std::int64_t> ids(n);
    if (labels.is_none()) {
      const auto first = static_cast<std::int64_t>(index_->size());
      for (std::size_t i = 0; i < n; ++i) ids[i] = first + static_cast<std::int64_t>(i);
    } else {
      const auto arr = labels.cast<LabelArray>();
      if (arr.ndim() != 1 || static_cast<std::size_t>(arr.shape(0)) != n) {
        throw std::invalid_argument("labels must be a 1-D array with one label per row");
      }
      std::copy(arr.data(), arr.data() + n, ids.begin());
    }
    py::gil_scoped_release release;
    oryx::parallel_for(0, n, threads, [&](std::size_t i) { index_->add(static_cast<std::uint64_t>(ids[i]), rows + i * dim); });
  }

  // Returns (labels, distances), each of shape (n, k). Missing results are -1 / inf.
  py::tuple search(const FloatArray& queries, std::size_t k, std::size_t ef, std::size_t threads) const {
    const std::size_t dim = index_->params().dim;
    std::size_t n;
    const float* rows = rows_of(queries, dim, n);
    LabelArray labels({n, k});
    FloatArray distances({n, k});
    std::int64_t* out_labels = labels.mutable_data();
    float* out_distances = distances.mutable_data();
    {
      py::gil_scoped_release release;
      oryx::parallel_for(0, n, threads, [&](std::size_t q) {
        const auto results = index_->search(rows + q * dim, k, {.ef = ef});
        for (std::size_t i = 0; i < k; ++i) {
          const bool found = i < results.size();
          out_labels[q * k + i] = found ? static_cast<std::int64_t>(results[i].label) : -1;
          out_distances[q * k + i] = found ? results[i].distance : std::numeric_limits<float>::infinity();
        }
      });
    }
    return py::make_tuple(labels, distances);
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

}  // namespace

PYBIND11_MODULE(oryx, m) {
  m.doc() = "Oryx: HNSW vector search engine";
  m.def("simd_level", [] { return std::string(oryx::to_string(oryx::active_simd_level())); },
        "SIMD instruction set used by the distance kernels on this CPU");

  py::class_<Index>(m, "Index")
      .def(py::init<std::size_t, const std::string&, std::size_t, std::size_t, std::size_t, std::uint64_t>(),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("max_elements") = 1'000'000, py::arg("M") = 16,
           py::arg("ef_construction") = 200, py::arg("seed") = 42)
      .def("add", &Index::add, py::arg("data"), py::arg("labels") = py::none(), py::arg("threads") = 0,
           "Insert rows of `data`; labels default to consecutive ids starting at len(index).")
      .def("search", &Index::search, py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 64,
           py::arg("threads") = 1, "Returns (labels, distances), each of shape (n, k).")
      .def("trace", &Index::trace, py::arg("query"), py::arg("k") = 10, py::arg("ef") = 64)
      .def("save", &Index::save, py::arg("path"))
      .def_static("load", &Index::load, py::arg("path"))
      .def("level", [](const Index& self, std::uint64_t label) { return self.raw().level_of(label); })
      .def("neighbors", [](const Index& self, std::uint64_t label, int layer) { return self.raw().neighbors_of(label, layer); },
           py::arg("label"), py::arg("layer") = 0)
      .def("__len__", [](const Index& self) { return self.raw().size(); })
      .def_property_readonly("dim", [](const Index& self) { return self.raw().params().dim; })
      .def_property_readonly("max_level", [](const Index& self) { return self.raw().max_level(); })
      .def_property_readonly("memory_bytes", [](const Index& self) { return self.raw().memory_bytes(); })
      .def_property_readonly("simd", [](const Index& self) { return std::string(oryx::to_string(self.raw().simd_level())); });
}
