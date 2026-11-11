#include "epzero/metadata.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>

namespace epzero {
namespace {

constexpr char kMagic[8] = {'E', 'P', 'Z', 'R', 'M', 'E', 'T', 'A'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kMaxName = 4096;
constexpr std::uint32_t kMaxText = 1u << 20;
constexpr std::uint32_t kMaxFields = 4096;

enum class Type : std::uint8_t { Integer = 0, Decimal = 1, Text = 2, Boolean = 3 };

template <typename T>
void write_pod(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_pod(std::ifstream& in) {
  T value{};
  if (!in.read(reinterpret_cast<char*>(&value), sizeof(T))) throw std::runtime_error("metadata file is truncated");
  return value;
}

std::string read_string(std::ifstream& in, std::uint32_t limit) {
  const auto length = read_pod<std::uint32_t>(in);
  if (length > limit) throw std::runtime_error("metadata file is corrupt");
  std::string text(length, '\0');
  if (length > 0 && !in.read(text.data(), length)) throw std::runtime_error("metadata file is truncated");
  return text;
}

void write_string(std::ofstream& out, const std::string& text) {
  write_pod(out, static_cast<std::uint32_t>(text.size()));
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::vector<std::uint64_t> sorted(const std::unordered_set<std::uint64_t>& labels) {
  std::vector<std::uint64_t> out(labels.begin(), labels.end());
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

MetadataIndex::MetadataIndex(MetadataIndex&& other) noexcept {
  std::unique_lock lock(other.lock_);
  fields_ = std::move(other.fields_);
  rows_ = std::move(other.rows_);
}

MetadataIndex& MetadataIndex::operator=(MetadataIndex&& other) noexcept {
  if (this != &other) {
    std::scoped_lock lock(lock_, other.lock_);
    fields_ = std::move(other.fields_);
    rows_ = std::move(other.rows_);
  }
  return *this;
}

std::optional<double> MetadataIndex::number_of(const MetaValue& value) {
  if (const auto* i = std::get_if<std::int64_t>(&value)) return static_cast<double>(*i);
  if (const auto* d = std::get_if<double>(&value)) return *d;
  return std::nullopt;
}

std::string MetadataIndex::key_of(const MetaValue& value) {
  if (const auto number = number_of(value)) {
    char buffer[32];
    const auto end = std::to_chars(buffer, buffer + sizeof(buffer), *number == 0.0 ? 0.0 : *number).ptr;
    return "n" + std::string(buffer, end);
  }
  if (const auto* text = std::get_if<std::string>(&value)) return "s" + *text;
  return std::get<bool>(value) ? "b1" : "b0";
}

void MetadataIndex::remove_locked(std::uint64_t label) {
  const auto row = rows_.find(label);
  if (row == rows_.end()) return;
  for (const auto& [name, value] : row->second) {
    const auto f = fields_.find(name);
    if (f == fields_.end()) continue;
    Field& field = f->second;
    if (const auto v = field.by_value.find(key_of(value)); v != field.by_value.end()) {
      v->second.erase(label);
      if (v->second.empty()) field.by_value.erase(v);
    }
    if (const auto number = number_of(value); number && !std::isnan(*number)) {
      if (const auto n = field.by_number.find(*number); n != field.by_number.end()) {
        n->second.erase(label);
        if (n->second.empty()) field.by_number.erase(n);
      }
    }
    field.labels.erase(label);
    if (field.labels.empty()) fields_.erase(f);
  }
  rows_.erase(row);
}

void MetadataIndex::set(std::uint64_t label, const Fields& fields) {
  // A field named twice keeps its last value.
  Fields unique;
  for (const auto& [name, value] : fields) {
    if (name.size() > kMaxName) throw std::invalid_argument("metadata field names are limited to 4096 bytes");
    if (const auto* text = std::get_if<std::string>(&value); text != nullptr && text->size() > kMaxText) {
      throw std::invalid_argument("metadata text values are limited to 1 MiB");
    }
    const auto same = std::find_if(unique.begin(), unique.end(), [&](const auto& f) { return f.first == name; });
    if (same != unique.end()) {
      same->second = value;
    } else {
      unique.emplace_back(name, value);
    }
  }
  if (unique.size() > kMaxFields) throw std::invalid_argument("a label may have at most 4096 metadata fields");
  std::unique_lock lock(lock_);
  remove_locked(label);
  if (unique.empty()) return;
  for (const auto& [name, value] : unique) {
    Field& field = fields_[name];
    field.by_value[key_of(value)].insert(label);
    if (const auto number = number_of(value); number && !std::isnan(*number)) field.by_number[*number].insert(label);
    field.labels.insert(label);
  }
  rows_.emplace(label, std::move(unique));
}

bool MetadataIndex::remove(std::uint64_t label) {
  std::unique_lock lock(lock_);
  if (!rows_.contains(label)) return false;
  remove_locked(label);
  return true;
}

std::vector<std::uint64_t> MetadataIndex::select_one(const MetaCondition& c) const {
  const auto f = fields_.find(c.field);
  if (f == fields_.end()) return {};
  const Field& field = f->second;
  const bool ranged = c.min.has_value() || c.max.has_value();
  const auto in_range = [&](double x) {
    if (c.min && (c.min_exclusive ? !(x > *c.min) : !(x >= *c.min))) return false;
    if (c.max && (c.max_exclusive ? !(x < *c.max) : !(x <= *c.max))) return false;
    return true;
  };

  std::unordered_set<std::uint64_t> matching;
  if (!c.any_of.empty()) {
    for (const MetaValue& value : c.any_of) {
      if (ranged) {
        const auto number = number_of(value);
        if (!number || !in_range(*number)) continue;
      }
      if (const auto v = field.by_value.find(key_of(value)); v != field.by_value.end()) {
        matching.insert(v->second.begin(), v->second.end());
      }
    }
  } else if (ranged) {
    auto it = c.min ? field.by_number.lower_bound(*c.min) : field.by_number.begin();
    for (; it != field.by_number.end(); ++it) {
      if (c.max && (c.max_exclusive ? it->first >= *c.max : it->first > *c.max)) break;
      if (in_range(it->first)) matching.insert(it->second.begin(), it->second.end());
    }
  } else {
    matching = field.labels;  // the field is present
  }

  if (!c.negate) return sorted(matching);
  std::vector<std::uint64_t> out;
  out.reserve(field.labels.size() - std::min(field.labels.size(), matching.size()));
  for (const std::uint64_t label : field.labels) {
    if (!matching.contains(label)) out.push_back(label);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::uint64_t> MetadataIndex::select(const std::vector<MetaCondition>& conditions) const {
  std::shared_lock lock(lock_);
  if (conditions.empty()) {
    std::vector<std::uint64_t> all;
    all.reserve(rows_.size());
    for (const auto& [label, fields] : rows_) all.push_back(label);
    std::sort(all.begin(), all.end());
    return all;
  }
  std::vector<std::vector<std::uint64_t>> parts;
  parts.reserve(conditions.size());
  for (const MetaCondition& c : conditions) parts.push_back(select_one(c));
  // Intersect smallest first, so the running result only shrinks.
  std::sort(parts.begin(), parts.end(), [](const auto& a, const auto& b) { return a.size() < b.size(); });
  std::vector<std::uint64_t> result = std::move(parts[0]);
  std::vector<std::uint64_t> next;
  for (std::size_t i = 1; i < parts.size() && !result.empty(); ++i) {
    next.clear();
    std::set_intersection(result.begin(), result.end(), parts[i].begin(), parts[i].end(), std::back_inserter(next));
    result.swap(next);
  }
  return result;
}

MetadataIndex::Fields MetadataIndex::get(std::uint64_t label) const {
  std::shared_lock lock(lock_);
  const auto row = rows_.find(label);
  return row == rows_.end() ? Fields{} : row->second;
}

std::size_t MetadataIndex::size() const {
  std::shared_lock lock(lock_);
  return rows_.size();
}

bool MetadataIndex::contains(std::uint64_t label) const {
  std::shared_lock lock(lock_);
  return rows_.contains(label);
}

// magic, version, row count; per row: label, field count, then per field its name, a type byte and the
// value (int64, double, length-prefixed text, or a byte). The lookup structures are rebuilt on load.
void MetadataIndex::save(const std::string& path) const {
  std::shared_lock lock(lock_);
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open " + path + " for writing");
  out.write(kMagic, sizeof(kMagic));
  write_pod(out, kVersion);
  write_pod(out, static_cast<std::uint64_t>(rows_.size()));
  for (const auto& [label, fields] : rows_) {
    write_pod(out, label);
    write_pod(out, static_cast<std::uint32_t>(fields.size()));
    for (const auto& [name, value] : fields) {
      write_string(out, name);
      if (const auto* i = std::get_if<std::int64_t>(&value)) {
        write_pod(out, Type::Integer);
        write_pod(out, *i);
      } else if (const auto* d = std::get_if<double>(&value)) {
        write_pod(out, Type::Decimal);
        write_pod(out, *d);
      } else if (const auto* text = std::get_if<std::string>(&value)) {
        write_pod(out, Type::Text);
        write_string(out, *text);
      } else {
        write_pod(out, Type::Boolean);
        write_pod(out, static_cast<std::uint8_t>(std::get<bool>(value) ? 1 : 0));
      }
    }
  }
  out.flush();
  if (!out) throw std::runtime_error("failed to write " + path);
}

MetadataIndex MetadataIndex::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  char magic[sizeof(kMagic)];
  if (!in.read(magic, sizeof(magic)) || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error(path + " is not an Epzero metadata file");
  }
  if (read_pod<std::uint32_t>(in) != kVersion) throw std::runtime_error(path + ": unsupported metadata version");
  const auto rows = read_pod<std::uint64_t>(in);
  // Every row takes at least 12 bytes: a header claiming more rows than that is corrupt.
  if (rows > std::filesystem::file_size(path) / 12) throw std::runtime_error(path + ": corrupt metadata header");
  MetadataIndex index;
  for (std::uint64_t r = 0; r < rows; ++r) {
    const auto label = read_pod<std::uint64_t>(in);
    const auto count = read_pod<std::uint32_t>(in);
    if (count > kMaxFields) throw std::runtime_error(path + ": corrupt metadata row");
    Fields fields;
    fields.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
      std::string name = read_string(in, kMaxName);
      switch (static_cast<Type>(read_pod<std::uint8_t>(in))) {
        case Type::Integer: fields.emplace_back(std::move(name), read_pod<std::int64_t>(in)); break;
        case Type::Decimal: fields.emplace_back(std::move(name), read_pod<double>(in)); break;
        case Type::Text: fields.emplace_back(std::move(name), read_string(in, kMaxText)); break;
        case Type::Boolean: fields.emplace_back(std::move(name), read_pod<std::uint8_t>(in) != 0); break;
        default: throw std::runtime_error(path + ": corrupt metadata value");
      }
    }
    if (index.rows_.contains(label)) throw std::runtime_error(path + ": a label is listed twice");
    index.set(label, fields);
  }
  return index;
}

}  // namespace epzero
