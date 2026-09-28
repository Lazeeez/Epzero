#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>

#include "oryx/brute_force.h"
#include "oryx/dataset.h"

namespace oryx {
namespace {

template <typename T>
void write_row(std::ofstream& out, const std::vector<T>& row) {
  const auto dim = static_cast<std::int32_t>(row.size());
  out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
  out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(T)));
}

TEST(Dataset, ReadsFvecsAndIvecs) {
  const auto dir = std::filesystem::temp_directory_path();
  const auto fvecs = (dir / "oryx_test.fvecs").string();
  const auto ivecs = (dir / "oryx_test.ivecs").string();
  {
    std::ofstream out(fvecs, std::ios::binary);
    write_row<float>(out, {1, 2, 3});
    write_row<float>(out, {4, 5, 6});
  }
  {
    std::ofstream out(ivecs, std::ios::binary);
    write_row<std::int32_t>(out, {9, 8});
  }
  const Dataset ds = read_fvecs(fvecs);
  EXPECT_EQ(ds.n, 2u);
  EXPECT_EQ(ds.dim, 3u);
  EXPECT_EQ(ds.row(1)[2], 6.0f);
  EXPECT_EQ(read_fvecs(fvecs, 1).n, 1u);
  const auto gt = read_ivecs(ivecs);
  ASSERT_EQ(gt.size(), 1u);
  EXPECT_EQ(gt[0], (std::vector<std::uint32_t>{9, 8}));
  std::filesystem::remove(fvecs);
  std::filesystem::remove(ivecs);
}

TEST(Dataset, RejectsTruncatedFile) {
  const auto path = (std::filesystem::temp_directory_path() / "oryx_truncated.fvecs").string();
  {
    std::ofstream out(path, std::ios::binary);
    const std::int32_t dim = 4;
    const float one = 1.0f;
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(&one), sizeof(one));
  }
  EXPECT_THROW(read_fvecs(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(BruteForce, FindsExactNeighborsAndPerfectRecall) {
  const auto data = make_clustered(1000, 20, 16, 10, 3);
  const auto truth = compute_ground_truth(data.base.data.data(), data.base.n, 16, data.queries.data.data(),
                                          data.queries.n, 5, Metric::L2, 4);
  std::vector<std::vector<SearchResult>> results;
  for (std::size_t q = 0; q < data.queries.n; ++q) {
    results.push_back(brute_force_search(data.base.data.data(), data.base.n, 16, data.queries.row(q), 5, Metric::L2));
  }
  EXPECT_DOUBLE_EQ(recall_at_k(results, truth, 5), 1.0);
  // Querying with a base vector must return that vector at distance 0.
  const auto self = brute_force_search(data.base.data.data(), data.base.n, 16, data.base.row(42), 1, Metric::L2);
  EXPECT_EQ(self[0].label, 42u);
  EXPECT_EQ(self[0].distance, 0.0f);
}

}  // namespace
}  // namespace oryx
