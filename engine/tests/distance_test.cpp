#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "oryx/distance.h"

namespace oryx {
namespace {

constexpr SimdLevel kAllLevels[] = {SimdLevel::Scalar, SimdLevel::AVX2, SimdLevel::AVX512, SimdLevel::NEON,
                                    SimdLevel::WasmSimd128};

std::vector<float> random_vector(std::size_t dim, std::mt19937& rng) {
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (float& x : v) x = uniform(rng);
  return v;
}

// Every compiled-in kernel must match the scalar reference, including the tail handling for
// dimensions that are not a multiple of the vector width.
TEST(Distance, SimdKernelsMatchScalarForAllDimensions) {
  std::mt19937 rng(7);
  for (Metric metric : {Metric::L2, Metric::InnerProduct}) {
    const DistanceFn reference = get_distance_fn(metric, SimdLevel::Scalar);
    for (SimdLevel level : kAllLevels) {
      const DistanceFn fn = get_distance_fn(metric, level);
      if (fn == nullptr) continue;
      for (std::size_t dim = 1; dim <= 300; ++dim) {
        const auto a = random_vector(dim, rng);
        const auto b = random_vector(dim, rng);
        const float expected = reference(a.data(), b.data(), dim);
        EXPECT_NEAR(fn(a.data(), b.data(), dim), expected, 1e-4f * std::max(1.0f, std::abs(expected)))
            << to_string(level) << " " << to_string(metric) << " dim=" << dim;
      }
    }
  }
}

TEST(Distance, Sq8KernelsMatchScalarForAllDimensions) {
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> byte(0, 255);
  const Sq8DotFn reference = get_sq8_dot_fn(SimdLevel::Scalar);
  for (SimdLevel level : kAllLevels) {
    const Sq8DotFn fn = get_sq8_dot_fn(level);
    EXPECT_EQ(fn != nullptr, simd_level_supported(level)) << to_string(level);
    if (fn == nullptr) continue;
    for (std::size_t dim = 1; dim <= 300; ++dim) {
      const auto w = random_vector(dim, rng);
      std::vector<std::uint8_t> codes(dim);
      for (auto& c : codes) c = static_cast<std::uint8_t>(byte(rng));
      const float expected = reference(w.data(), codes.data(), dim);
      EXPECT_NEAR(fn(w.data(), codes.data(), dim), expected, 1e-3f * std::max(1.0f, std::abs(expected)))
          << to_string(level) << " dim=" << dim;
    }
  }
}

// Integer kernels are exact: every variant this CPU runs must equal the scalar sum, at the weight
// bound and for dimensions that leave every kind of tail.
TEST(Distance, Sq8IntKernelsMatchScalarExactly) {
  std::mt19937 rng(13);
  std::uniform_int_distribution<int> byte(0, 255);
  std::uniform_int_distribution<int> weight(-kSq8IntWeightMax, kSq8IntWeightMax);
  const auto variants = sq8_int_dot_variants();
  for (std::size_t dim = 1; dim <= 400; ++dim) {
    std::vector<std::int8_t> w(dim);
    std::vector<std::uint8_t> codes(dim);
    for (auto& x : w) x = static_cast<std::int8_t>(weight(rng));
    for (auto& c : codes) c = static_cast<std::uint8_t>(byte(rng));
    if (dim % 7 == 0) {  // the extremes, where the AVX2 pair sums are largest
      for (std::size_t i = 0; i < dim; ++i) {
        w[i] = static_cast<std::int8_t>(i % 2 ? kSq8IntWeightMax : -kSq8IntWeightMax);
        codes[i] = 255;
      }
    }
    const std::int32_t expected = variants[0].second(w.data(), codes.data(), dim);
    for (const auto& [name, fn] : variants) EXPECT_EQ(fn(w.data(), codes.data(), dim), expected) << name << " dim=" << dim;
  }
  for (SimdLevel level : kAllLevels) {
    if (!simd_level_supported(level)) EXPECT_EQ(get_sq8_int_dot_fn(level), nullptr) << to_string(level);
  }
}

TEST(Distance, KnownValues) {
  const float a[] = {1, 2, 3};
  const float b[] = {4, 6, 3};
  EXPECT_FLOAT_EQ(get_distance_fn(Metric::L2)(a, b, 3), 25.0f);          // 9 + 16 + 0
  EXPECT_FLOAT_EQ(get_distance_fn(Metric::InnerProduct)(a, b, 3), -24.0f);  // 1 - (4 + 12 + 9)
}

TEST(Distance, UnsupportedLevelReturnsNull) {
  for (SimdLevel level : kAllLevels) {
    EXPECT_EQ(get_distance_fn(Metric::L2, level) != nullptr, simd_level_supported(level)) << to_string(level);
  }
  EXPECT_TRUE(simd_level_supported(active_simd_level()));
}

TEST(Distance, NormalizeProducesUnitVectorAndIgnoresZero) {
  float v[] = {3, 4};
  normalize(v, 2);
  EXPECT_FLOAT_EQ(v[0], 0.6f);
  EXPECT_FLOAT_EQ(v[1], 0.8f);
  float zero[] = {0, 0};
  normalize(zero, 2);
  EXPECT_EQ(zero[0], 0.0f);
}

}  // namespace
}  // namespace oryx
