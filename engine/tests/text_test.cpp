#include "oryx/text.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#include "oryx/dataset.h"

namespace oryx {
namespace {

std::vector<std::uint64_t> labels_of(const std::vector<SearchResult>& results) {
  std::vector<std::uint64_t> out;
  for (const auto& r : results) out.push_back(r.label);
  return out;
}

TEST(TextIndex, TokenizesWordsCaseInsensitively) {
  EXPECT_EQ(TextIndex::tokenize("Error E-1042: Disk FULL!"), (std::vector<std::string>{"error", "e", "1042", "disk", "full"}));
  EXPECT_EQ(TextIndex::tokenize("  "), std::vector<std::string>{});
  // Multi-byte UTF-8 stays inside the word.
  EXPECT_EQ(TextIndex::tokenize("café über"), (std::vector<std::string>{"café", "über"}));
}

TEST(TextIndex, RanksByBm25) {
  TextIndex index;
  index.add(1, "the cat sat on the mat");
  index.add(2, "the dog chased the cat around the cat tree");
  index.add(3, "a dog in the fog");
  index.add(4, "nothing relevant here");
  // "cat" appears twice in 2 (once in 1): 2 first. 3 and 4 share no term with the query.
  EXPECT_EQ(labels_of(index.search("cat", 10)), (std::vector<std::uint64_t>{2, 1}));
  // A rarer term outweighs a common one: "fog" (one document) beats "the" (three).
  EXPECT_EQ(index.search("the fog", 1)[0].label, 3u);
  // Scores are negated (smaller is better) and sorted.
  const auto results = index.search("dog cat", 10);
  for (std::size_t i = 1; i < results.size(); ++i) EXPECT_LE(results[i - 1].distance, results[i].distance);
  EXPECT_LT(results[0].distance, 0.0f);
}

TEST(TextIndex, MatchesTheBm25Formula) {
  TextIndex index({.k1 = 1.2f, .b = 0.75f});
  index.add(7, "alpha beta beta");
  index.add(8, "gamma");
  const double n = 2, df = 1, avgdl = 2, tf = 2, len = 3;
  const double idf = std::log(1 + (n - df + 0.5) / (df + 0.5));
  const double expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * len / avgdl));
  const auto results = index.search("beta", 1);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_NEAR(-results[0].distance, expected, 1e-5);
}

TEST(TextIndex, RemoveAndReplace) {
  TextIndex index;
  index.add(1, "apple banana");
  index.add(2, "banana cherry");
  EXPECT_EQ(index.size(), 2u);
  EXPECT_TRUE(index.remove(1));
  EXPECT_FALSE(index.remove(1));
  EXPECT_EQ(index.size(), 1u);
  EXPECT_TRUE(index.search("apple", 5).empty());
  index.add(2, "durian");  // replaces label 2's text
  EXPECT_TRUE(index.search("banana", 5).empty());
  EXPECT_EQ(labels_of(index.search("durian", 5)), (std::vector<std::uint64_t>{2}));
  // Heavy churn compacts the dead postings away without changing answers.
  for (std::uint64_t round = 0; round < 3000; ++round) index.add(round % 50, "word" + std::to_string(round % 7) + " common");
  EXPECT_EQ(index.size(), 50u);
  std::size_t expected = 0;  // labels whose last text (round 2950 + label) had word3
  for (std::uint64_t label = 0; label < 50; ++label) expected += (2950 + label) % 7 == 3;
  EXPECT_EQ(index.search("word3", 100).size(), expected);
}

TEST(TextIndex, SaveLoadRoundTrip) {
  TextIndex index({.k1 = 1.5f, .b = 0.5f});
  std::mt19937 rng(3);
  const std::vector<std::string> words = {"red", "green", "blue", "car", "tree", "river", "night", "city"};
  for (std::uint64_t i = 0; i < 500; ++i) {
    std::string text;
    for (int w = 0; w < 6; ++w) text += words[rng() % words.size()] + " ";
    index.add(i * 3, text);
  }
  index.remove(9);
  const auto path = (std::filesystem::temp_directory_path() / "oryx_text_test.idx").string();
  index.save(path);
  const TextIndex loaded = TextIndex::load(path);
  EXPECT_EQ(loaded.size(), index.size());
  EXPECT_FALSE(loaded.contains(9));
  for (const char* q : {"red car", "night city river", "blue"}) {
    const auto a = index.search(q, 20);
    const auto b = loaded.search(q, 20);
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      EXPECT_EQ(a[i].label, b[i].label);
      EXPECT_FLOAT_EQ(a[i].distance, b[i].distance);
    }
  }
  // A truncated file is refused.
  std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);
  EXPECT_THROW(TextIndex::load(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(Fusion, ReciprocalRankFusion) {
  const std::vector<SearchResult> a = {{1, 0}, {2, 0}, {3, 0}};
  const std::vector<SearchResult> b = {{3, 0}, {4, 0}};
  const auto fused = fuse_rankings({a, b}, 10, 60.0);
  // 3 is in both lists: 1/63 + 1/61 beats 1's 1/61 alone.
  EXPECT_EQ(labels_of(fused), (std::vector<std::uint64_t>{3, 1, 2, 4}));
  EXPECT_NEAR(-fused[0].distance, 1.0 / 63 + 1.0 / 61, 1e-6);
  // Weights scale each list's votes.
  EXPECT_EQ(fuse_rankings({a, b}, 1, 60.0, {0.1, 1.0})[0].label, 3u);
  EXPECT_EQ(fuse_rankings({a, b}, 1, 60.0, {1.0, 0.0})[0].label, 1u);
  EXPECT_THROW(fuse_rankings({a, b}, 1, 60.0, {1.0}), std::invalid_argument);
}

// The point of hybrid search: a document whose embedding is far away but whose text contains the
// query's exact term is found, and documents good on both counts rank first.
TEST(Hybrid, KeywordMatchesRescueWhatTheVectorMisses) {
  const std::size_t dim = 16, n = 2000;
  auto data = make_clustered(n, 1, dim, 20, 5);
  HnswIndex vectors({.dim = dim, .metric = Metric::L2, .max_elements = n});
  TextIndex text;
  for (std::size_t i = 0; i < n; ++i) {
    vectors.add(i, data.base.row(i));
    text.add(i, "document number " + std::to_string(i));
  }
  const auto nearest = vectors.search(data.queries.row(0), 10, {.ef = 100});
  // The farthest vector, labelled with a rare code that only its text contains.
  const std::uint64_t far = vectors.exact_search(data.queries.row(0), n).back().label;
  text.add(far, "error code XJ-9000 in the billing service");
  const auto hits = hybrid_search(vectors, text, data.queries.row(0), "XJ-9000", 10);
  EXPECT_NE(std::find_if(hits.begin(), hits.end(), [&](const SearchResult& r) { return r.label == far; }), hits.end());
  // With no matching text, hybrid is the vector ranking.
  const auto plain = hybrid_search(vectors, text, data.queries.row(0), "zzz", 10);
  EXPECT_EQ(labels_of(plain), labels_of(nearest));
}

}  // namespace
}  // namespace oryx
