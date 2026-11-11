#pragma once

// Metadata stored beside the vectors, for filtered search.
//
// Each label carries named fields (text, integers, decimals or booleans). select() turns conditions
// on them into the labels that pass, which HnswIndex::make_filter compiles into a SearchFilter, so
// the engine then picks its filtered-search strategy as for any other filter:
//
//   MetadataIndex meta;
//   meta.set(7, {{"source", std::string("faq")}, {"year", std::int64_t{2024}}});
//   auto labels = meta.select({{.field = "source", .any_of = {std::string("faq")}},
//                              {.field = "year", .min = 2020.0}});
//   auto filter = index.make_filter(labels.data(), labels.size());
//
// Equality looks values up in an inverted index per field; numeric ranges walk an ordered map, so
// neither scans the labels. Integers and decimals compare as numbers (2 equals 2.0), as doubles: exact
// up to 2^53.
//
// Thread safety: set() and remove() may run concurrently with select() and with each other.

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace epzero {

using MetaValue = std::variant<std::int64_t, double, std::string, bool>;

// One condition on a field; select() requires all of them. `any_of`: the field equals one of these
// values. `min` / `max`: a numeric range (inclusive unless *_exclusive). A label without the field
// never passes.
struct MetaCondition {
  std::string field;
  std::vector<MetaValue> any_of;
  std::optional<double> min;
  std::optional<double> max;
  bool min_exclusive = false;
  bool max_exclusive = false;
  bool negate = false;  // the label has the field and fails the rest of the condition
};

class MetadataIndex {
 public:
  using Fields = std::vector<std::pair<std::string, MetaValue>>;

  MetadataIndex() = default;
  MetadataIndex(MetadataIndex&& other) noexcept;
  MetadataIndex& operator=(MetadataIndex&& other) noexcept;

  // Sets `label`'s fields, replacing any it had. A field named twice keeps the last value.
  void set(std::uint64_t label, const Fields& fields);
  // Forgets `label`; returns false if it had no fields.
  bool remove(std::uint64_t label);

  // The labels meeting every condition, in increasing order (no conditions: every label).
  std::vector<std::uint64_t> select(const std::vector<MetaCondition>& conditions) const;
  // `label`'s fields, empty if it has none.
  Fields get(std::uint64_t label) const;

  std::size_t size() const;  // labels with fields
  bool contains(std::uint64_t label) const;

  void save(const std::string& path) const;
  static MetadataIndex load(const std::string& path);

 private:
  // A value's equality key: numbers by their double value, so 2 and 2.0 match.
  static std::string key_of(const MetaValue& value);
  static std::optional<double> number_of(const MetaValue& value);
  void remove_locked(std::uint64_t label);
  std::vector<std::uint64_t> select_one(const MetaCondition& condition) const;

  struct Field {
    std::unordered_map<std::string, std::unordered_set<std::uint64_t>> by_value;  // equality key -> labels
    std::map<double, std::unordered_set<std::uint64_t>> by_number;               // numeric values, ordered
    std::unordered_set<std::uint64_t> labels;                                     // every label that has it
  };

  mutable std::shared_mutex lock_;
  std::unordered_map<std::string, Field> fields_;
  std::unordered_map<std::uint64_t, Fields> rows_;
};

}  // namespace epzero
