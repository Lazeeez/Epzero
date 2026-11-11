#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <thread>
#include <vector>

#include "epzero/metadata.h"

namespace epzero {
namespace {

using L = std::vector<std::uint64_t>;

MetadataIndex sample() {
  MetadataIndex m;
  m.set(1, {{"source", std::string("faq")}, {"year", std::int64_t{2021}}, {"public", true}});
  m.set(2, {{"source", std::string("blog")}, {"year", std::int64_t{2024}}, {"score", 0.5}});
  m.set(3, {{"source", std::string("faq")}, {"year", 2024.0}, {"public", false}});
  m.set(4, {{"source", std::string("docs")}, {"score", 0.9}});
  return m;
}

TEST(Metadata, EqualityAnyOfAndConjunction) {
  const auto m = sample();
  EXPECT_EQ(m.select({{.field = "source", .any_of = {std::string("faq")}}}), (L{1, 3}));
  EXPECT_EQ(m.select({{.field = "source", .any_of = {std::string("faq"), std::string("docs")}}}), (L{1, 3, 4}));
  // Integers and decimals compare as numbers.
  EXPECT_EQ(m.select({{.field = "year", .any_of = {std::int64_t{2024}}}}), (L{2, 3}));
  EXPECT_EQ(m.select({{.field = "source", .any_of = {std::string("faq")}}, {.field = "year", .any_of = {2024.0}}}), (L{3}));
  EXPECT_EQ(m.select({{.field = "public", .any_of = {true}}}), (L{1}));
  EXPECT_TRUE(m.select({{.field = "missing", .any_of = {std::string("x")}}}).empty());
  EXPECT_EQ(m.select({}), (L{1, 2, 3, 4}));
}

TEST(Metadata, RangesPresenceAndNegation) {
  const auto m = sample();
  EXPECT_EQ(m.select({{.field = "year", .min = 2022.0}}), (L{2, 3}));
  EXPECT_EQ(m.select({{.field = "year", .max = 2024.0, .max_exclusive = true}}), (L{1}));
  EXPECT_EQ(m.select({{.field = "score", .min = 0.5, .max = 0.9, .min_exclusive = true}}), (L{4}));
  EXPECT_EQ(m.select({{.field = "score"}}), (L{2, 4}));  // has the field
  // Negation keeps labels that have the field but fail the condition.
  EXPECT_EQ(m.select({{.field = "source", .any_of = {std::string("faq")}, .negate = true}}), (L{2, 4}));
  EXPECT_EQ(m.select({{.field = "year", .min = 2022.0, .negate = true}}), (L{1}));
  // A range with listed values keeps only the listed values in range.
  EXPECT_EQ(m.select({{.field = "year", .any_of = {std::int64_t{2021}, std::int64_t{2024}}, .min = 2022.0}}), (L{2, 3}));
}

TEST(Metadata, ReplaceRemoveAndGet) {
  auto m = sample();
  m.set(1, {{"source", std::string("blog")}});
  EXPECT_EQ(m.select({{.field = "source", .any_of = {std::string("faq")}}}), (L{3}));
  EXPECT_TRUE(m.select({{.field = "public", .any_of = {true}}}).empty());
  EXPECT_TRUE(m.remove(3));
  EXPECT_FALSE(m.remove(3));
  EXPECT_FALSE(m.contains(3));
  EXPECT_TRUE(m.select({{.field = "source", .any_of = {std::string("faq")}}}).empty());
  EXPECT_EQ(m.size(), 3u);
  const auto fields = m.get(2);
  ASSERT_EQ(fields.size(), 3u);
  EXPECT_EQ(std::get<std::string>(fields[0].second), "blog");
  m.set(5, {{"k", std::int64_t{1}}, {"k", std::int64_t{2}}});  // the last value wins
  EXPECT_EQ(m.select({{.field = "k", .any_of = {std::int64_t{2}}}}), (L{5}));
  EXPECT_TRUE(m.get(99).empty());
}

TEST(Metadata, SaveAndLoad) {
  const auto m = sample();
  const auto path = (std::filesystem::temp_directory_path() / "epzero_meta.bin").string();
  m.save(path);
  const auto loaded = MetadataIndex::load(path);
  std::filesystem::remove(path);
  EXPECT_EQ(loaded.size(), 4u);
  EXPECT_EQ(loaded.select({{.field = "year", .min = 2022.0}}), (L{2, 3}));
  EXPECT_EQ(loaded.get(4), m.get(4));
}

TEST(Metadata, LoadRejectsGarbage) {
  const auto path = (std::filesystem::temp_directory_path() / "epzero_meta_bad.bin").string();
  {
    std::ofstream out(path, std::ios::binary);
    out << "EPZRMETA";
    const std::uint32_t version = 1;
    const std::uint64_t rows = 1'000'000'000;  // far more than the file holds
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
  }
  EXPECT_THROW(MetadataIndex::load(path), std::runtime_error);
  {
    std::ofstream out(path, std::ios::binary);
    out << "NOTMETA!";
  }
  EXPECT_THROW(MetadataIndex::load(path), std::runtime_error);
  std::filesystem::remove(path);
}

TEST(Metadata, ConcurrentWritesAndReads) {
  MetadataIndex m;
  std::vector<std::thread> threads;
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      for (std::uint64_t i = 0; i < 2000; ++i) {
        const std::uint64_t label = t * 10000 + i;
        m.set(label, {{"t", std::int64_t{t}}, {"i", static_cast<std::int64_t>(i)}});
        if (i % 3 == 0) m.remove(label);
        (void)m.select({{.field = "t", .any_of = {std::int64_t{t}}}, {.field = "i", .max = 100.0}});
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(m.size(), 4u * (2000 - 667));
  EXPECT_EQ(m.select({{.field = "t", .any_of = {std::int64_t{0}}}}).size(), 2000u - 667);
}

}  // namespace
}  // namespace epzero
