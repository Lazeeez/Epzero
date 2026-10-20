// Fuzzes HnswIndex::import_graph, which the website feeds a downloaded file: arbitrary bytes must
// either rebuild a graph that is safe to search, or throw.
#include <exception>
#include <vector>

#include "common.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  using namespace oryx::fuzz;
  static const std::vector<float> vectors = [] {
    std::vector<float> v(kMaxElements * kDim);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>((i * 2654435761u) % 1000) / 1000.0f;
    return v;
  }();
  try {
    const auto index = oryx::HnswIndex::import_graph(small_params(), vectors.data(), kMaxElements, data, size);
    exercise(*index);
  } catch (const std::exception&) {
    // Rejected cleanly.
  }
  return 0;
}
