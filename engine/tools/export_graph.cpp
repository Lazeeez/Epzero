// Builds the web demo's HNSW graph natively and writes it as a graph-only file (no vectors), so
// the site can load a prebuilt index instead of building one on every visit.
//
//   epzero-export-graph <vectors.i8> <scales.f32> <dim> <M> <efConstruction> <out.graph>
//
// The vectors are dequantized and normalized exactly as the WebAssembly Collection does
// (code * scale[d] / 127, then L2-normalized), and indexed with cosine distance, so the graph
// matches the vectors the browser reconstructs from the same files.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "epzero/distance.h"
#include "epzero/hnsw.h"
#include "epzero/parallel.h"

namespace {

std::vector<char> read_file(const char* path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(std::string("cannot read ") + path);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr, "usage: %s <vectors.i8> <scales.f32> <dim> <M> <efConstruction> <out.graph>\n", argv[0]);
    return 2;
  }
  try {
    const auto codes = read_file(argv[1]);
    const auto scale_bytes = read_file(argv[2]);
    const std::size_t dim = std::stoul(argv[3]);
    const std::size_t M = std::stoul(argv[4]);
    const std::size_t ef_construction = std::stoul(argv[5]);
    if (dim == 0 || codes.size() % dim != 0 || scale_bytes.size() != dim * sizeof(float)) {
      throw std::runtime_error("vector and scale files don't match the dimension");
    }
    const std::size_t n = codes.size() / dim;
    const auto* scales = reinterpret_cast<const float*>(scale_bytes.data());
    std::vector<float> vectors(codes.size());
    for (std::size_t i = 0; i < codes.size(); ++i) {
      vectors[i] = static_cast<float>(static_cast<std::int8_t>(codes[i])) * (scales[i % dim] / 127.0f);
    }
    for (std::size_t i = 0; i < n; ++i) epzero::normalize(vectors.data() + i * dim, dim);

    epzero::HnswIndex index(epzero::HnswParams{.dim = dim, .metric = epzero::Metric::Cosine, .max_elements = n, .M = M,
                                           .ef_construction = ef_construction});
    const auto start = std::chrono::steady_clock::now();
    const std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
    epzero::parallel_for(0, n, threads, [&](std::size_t i) { index.add(i, vectors.data() + i * dim); });
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    const auto bytes = index.export_graph();
    std::ofstream out(argv[6], std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error(std::string("cannot write ") + argv[6]);
    std::printf("%zu vectors x %zu-d, M=%zu, efConstruction=%zu: built in %.1f s on %zu threads, %d layers, %zu bytes\n",
                n, dim, M, ef_construction, seconds, threads, index.max_level() + 1, bytes.size());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
