// Microbenchmark: nanoseconds per distance call for each SIMD level the CPU supports.
// The working set is small enough to stay in L1, so this isolates arithmetic throughput from
// memory latency (bench_recall measures the end-to-end effect).

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "oryx/distance.h"

using namespace oryx;

namespace {

double ns_per_call(DistanceFn fn, const std::vector<float>& data, std::size_t dim, std::size_t rows) {
  constexpr std::size_t kCalls = 2'000'000;
  volatile float sink = 0.0f;  // keeps the compiler from deleting the loop
  float acc = 0.0f;
  for (std::size_t i = 0; i < 10'000; ++i) acc += fn(data.data(), data.data() + dim, dim);  // warm up
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < kCalls; ++i) {
    const std::size_t a = i % rows;
    const std::size_t b = (i * 7 + 3) % rows;
    acc += fn(data.data() + a * dim, data.data() + b * dim, dim);
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  sink = acc;
  (void)sink;
  return std::chrono::duration<double, std::nano>(elapsed).count() / kCalls;
}

}  // namespace

int main() {
  constexpr std::size_t kRows = 16;
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::printf("active SIMD level: %s\n\n", std::string(to_string(active_simd_level())).c_str());
  std::printf("%-8s %-6s %6s %12s %10s\n", "metric", "simd", "dim", "ns/call", "speedup");

  for (Metric metric : {Metric::L2, Metric::InnerProduct}) {
    for (std::size_t dim : {128, 384, 768, 960}) {
      std::vector<float> data(kRows * dim);
      for (float& x : data) x = uniform(rng);
      const double scalar = ns_per_call(get_distance_fn(metric, SimdLevel::Scalar), data, dim, kRows);
      for (SimdLevel level : {SimdLevel::Scalar, SimdLevel::AVX2, SimdLevel::AVX512, SimdLevel::NEON}) {
        const DistanceFn fn = get_distance_fn(metric, level);
        if (fn == nullptr) continue;
        const double ns = level == SimdLevel::Scalar ? scalar : ns_per_call(fn, data, dim, kRows);
        std::printf("%-8s %-6s %6zu %12.2f %9.1fx\n", std::string(to_string(metric)).c_str(),
                    std::string(to_string(level)).c_str(), dim, ns, scalar / ns);
      }
    }
  }
  return 0;
}
