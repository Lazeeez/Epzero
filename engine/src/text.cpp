#include "epzero/text.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_set>

namespace epzero {

namespace {

constexpr char kMagic[8] = {'E', 'P', 'Z', 'R', 'T', 'X', 'T', '1'};
constexpr char kLegacyMagic[8] = {'O', 'R', 'Y', 'X', 'T', 'X', 'T', '1'};  // from before the rename

bool word_byte(unsigned char c) noexcept {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

template <typename T>
void put(std::ofstream& out, const T& value) {
  out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T get(std::ifstream& in) {
  T value{};
  if (!in.read(reinterpret_cast<char*>(&value), sizeof(T))) throw std::runtime_error("text index file is truncated");
  return value;
}

}  // namespace

std::vector<std::string> TextIndex::tokenize(std::string_view text) {
  std::vector<std::string> tokens;
  std::string current;
  for (const char ch : text) {
    const auto c = static_cast<unsigned char>(ch);
    if (word_byte(c)) {
      current.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : ch);
    } else if (!current.empty()) {
      tokens.push_back(std::move(current));
      current.clear();
    }
  }
  if (!current.empty()) tokens.push_back(std::move(current));
  return tokens;
}

TextIndex::TextIndex(TextIndex&& other) noexcept {
  std::unique_lock lock(other.lock_);
  options_ = other.options_;
  term_ids_ = std::move(other.term_ids_);
  postings_ = std::move(other.postings_);
  live_df_ = std::move(other.live_df_);
  docs_ = std::move(other.docs_);
  slot_of_ = std::move(other.slot_of_);
  live_ = other.live_;
  total_length_ = other.total_length_;
}

TextIndex& TextIndex::operator=(TextIndex&& other) noexcept {
  if (this != &other) {
    std::scoped_lock lock(lock_, other.lock_);
    options_ = other.options_;
    term_ids_ = std::move(other.term_ids_);
    postings_ = std::move(other.postings_);
    live_df_ = std::move(other.live_df_);
    docs_ = std::move(other.docs_);
    slot_of_ = std::move(other.slot_of_);
    live_ = other.live_;
    total_length_ = other.total_length_;
  }
  return *this;
}

void TextIndex::add(std::uint64_t label, std::string_view text) {
  const auto tokens = tokenize(text);
  std::unordered_map<std::string_view, std::uint32_t> counts;
  for (const auto& t : tokens) ++counts[t];

  std::unique_lock lock(lock_);
  remove_locked(label);
  // Postings of removed documents are skipped at query time; once they are the majority, drop them.
  if (docs_.size() >= 1024 && docs_.size() > 2 * live_) {
    std::vector<std::uint32_t> remap(docs_.size(), UINT32_MAX);
    std::vector<Doc> kept;
    kept.reserve(live_);
    for (std::uint32_t i = 0; i < docs_.size(); ++i) {
      if (!docs_[i].live) continue;
      remap[i] = static_cast<std::uint32_t>(kept.size());
      kept.push_back(std::move(docs_[i]));
    }
    for (auto& list : postings_) {
      std::erase_if(list, [&](const Posting& p) { return remap[p.doc] == UINT32_MAX; });
      for (auto& p : list) p.doc = remap[p.doc];
    }
    docs_ = std::move(kept);
    for (std::uint32_t i = 0; i < docs_.size(); ++i) slot_of_[docs_[i].label] = i;
  }

  const auto slot = static_cast<std::uint32_t>(docs_.size());
  Doc doc{label, static_cast<std::uint32_t>(tokens.size()), true, {}};
  doc.terms.reserve(counts.size());
  for (const auto& [term, tf] : counts) {
    auto [it, inserted] = term_ids_.try_emplace(std::string(term), static_cast<std::uint32_t>(postings_.size()));
    if (inserted) {
      postings_.emplace_back();
      live_df_.push_back(0);
    }
    postings_[it->second].push_back({slot, tf});
    ++live_df_[it->second];
    doc.terms.push_back(it->second);
  }
  docs_.push_back(std::move(doc));
  slot_of_[label] = slot;
  ++live_;
  total_length_ += tokens.size();
}

void TextIndex::remove_locked(std::uint64_t label) {
  const auto it = slot_of_.find(label);
  if (it == slot_of_.end()) return;
  Doc& doc = docs_[it->second];
  doc.live = false;
  for (const auto t : doc.terms) --live_df_[t];
  doc.terms.clear();
  doc.terms.shrink_to_fit();
  --live_;
  total_length_ -= doc.length;
  slot_of_.erase(it);
}

bool TextIndex::remove(std::uint64_t label) {
  std::unique_lock lock(lock_);
  if (!slot_of_.contains(label)) return false;
  remove_locked(label);
  return true;
}

std::size_t TextIndex::size() const {
  std::shared_lock lock(lock_);
  return live_;
}

bool TextIndex::contains(std::uint64_t label) const {
  std::shared_lock lock(lock_);
  return slot_of_.contains(label);
}

std::vector<SearchResult> TextIndex::search(std::string_view query, std::size_t k) const {
  auto tokens = tokenize(query);
  std::sort(tokens.begin(), tokens.end());
  tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());

  std::shared_lock lock(lock_);
  if (k == 0 || live_ == 0) return {};
  const double n = static_cast<double>(live_);
  const double average = static_cast<double>(total_length_) / n;
  std::unordered_map<std::uint32_t, double> scores;
  for (const auto& token : tokens) {
    const auto it = term_ids_.find(token);
    if (it == term_ids_.end()) continue;
    const double df = live_df_[it->second];
    if (df == 0) continue;
    const double idf = std::log(1.0 + (n - df + 0.5) / (df + 0.5));  // never negative (Lucene's form)
    for (const Posting& p : postings_[it->second]) {
      const Doc& doc = docs_[p.doc];
      if (!doc.live) continue;
      const double tf = p.tf;
      const double norm = options_.k1 * (1.0 - options_.b + options_.b * doc.length / std::max(average, 1e-9));
      scores[p.doc] += idf * tf * (options_.k1 + 1.0) / (tf + norm);
    }
  }
  std::vector<SearchResult> results;
  results.reserve(scores.size());
  for (const auto& [slot, score] : scores) results.push_back({docs_[slot].label, static_cast<float>(-score)});
  const auto better = [](const SearchResult& a, const SearchResult& b) {
    return a.distance != b.distance ? a.distance < b.distance : a.label < b.label;
  };
  const std::size_t keep = std::min(k, results.size());
  std::partial_sort(results.begin(), results.begin() + static_cast<std::ptrdiff_t>(keep), results.end(), better);
  results.resize(keep);
  return results;
}

void TextIndex::save(const std::string& path) const {
  std::shared_lock lock(lock_);
  // Per live document: its label, length and (term, tf) pairs; the inverted lists are rebuilt on load.
  std::vector<std::vector<std::pair<std::uint32_t, std::uint32_t>>> terms_of(docs_.size());
  for (std::uint32_t t = 0; t < postings_.size(); ++t) {
    for (const Posting& p : postings_[t]) {
      if (docs_[p.doc].live) terms_of[p.doc].emplace_back(t, p.tf);
    }
  }
  std::vector<const std::string*> names(postings_.size());
  for (const auto& [name, id] : term_ids_) names[id] = &name;

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write " + path);
  out.write(kMagic, sizeof(kMagic));
  put(out, options_.k1);
  put(out, options_.b);
  put(out, static_cast<std::uint64_t>(live_));
  for (std::uint32_t i = 0; i < docs_.size(); ++i) {
    if (!docs_[i].live) continue;
    put(out, docs_[i].label);
    put(out, docs_[i].length);
    put(out, static_cast<std::uint32_t>(terms_of[i].size()));
    for (const auto& [term, tf] : terms_of[i]) {
      put(out, static_cast<std::uint32_t>(names[term]->size()));
      out.write(names[term]->data(), static_cast<std::streamsize>(names[term]->size()));
      put(out, tf);
    }
  }
  if (!out) throw std::runtime_error("failed writing " + path);
}

TextIndex TextIndex::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  char magic[sizeof(kMagic)];
  if (!in.read(magic, sizeof(magic)) ||
      (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 && std::memcmp(magic, kLegacyMagic, sizeof(kMagic)) != 0)) {
    throw std::runtime_error(path + " is not an Epzero text index");
  }
  Options options;
  options.k1 = get<float>(in);
  options.b = get<float>(in);
  TextIndex index(options);
  const auto docs = get<std::uint64_t>(in);
  std::unordered_set<std::uint64_t> seen;
  for (std::uint64_t d = 0; d < docs; ++d) {
    const auto label = get<std::uint64_t>(in);
    const auto length = get<std::uint32_t>(in);
    const auto count = get<std::uint32_t>(in);
    if (!seen.insert(label).second) throw std::runtime_error("text index file repeats a label");
    const auto slot = static_cast<std::uint32_t>(index.docs_.size());
    Doc doc{label, length, true, {}};
    std::uint64_t counted = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
      const auto size = get<std::uint32_t>(in);
      if (size == 0 || size > (1u << 20)) throw std::runtime_error("text index file is corrupt");
      std::string term(size, '\0');
      if (!in.read(term.data(), size)) throw std::runtime_error("text index file is truncated");
      const auto tf = get<std::uint32_t>(in);
      if (tf == 0) throw std::runtime_error("text index file is corrupt");
      counted += tf;
      auto [it, inserted] = index.term_ids_.try_emplace(std::move(term), static_cast<std::uint32_t>(index.postings_.size()));
      if (inserted) {
        index.postings_.emplace_back();
        index.live_df_.push_back(0);
      } else if (!index.postings_[it->second].empty() && index.postings_[it->second].back().doc == slot) {
        throw std::runtime_error("text index file repeats a term");
      }
      index.postings_[it->second].push_back({slot, tf});
      ++index.live_df_[it->second];
      doc.terms.push_back(it->second);
    }
    if (counted != length) throw std::runtime_error("text index file is corrupt");
    index.docs_.push_back(std::move(doc));
    index.slot_of_[label] = slot;
    ++index.live_;
    index.total_length_ += length;
  }
  return index;
}

std::vector<SearchResult> fuse_rankings(const std::vector<std::vector<SearchResult>>& lists, std::size_t k, double rrf_k,
                                        const std::vector<double>& weights) {
  if (!weights.empty() && weights.size() != lists.size()) {
    throw std::invalid_argument("fuse_rankings: one weight per list");
  }
  std::unordered_map<std::uint64_t, double> scores;
  for (std::size_t l = 0; l < lists.size(); ++l) {
    const double weight = weights.empty() ? 1.0 : weights[l];
    for (std::size_t r = 0; r < lists[l].size(); ++r) {
      scores[lists[l][r].label] += weight / (rrf_k + static_cast<double>(r + 1));
    }
  }
  std::vector<SearchResult> fused;
  fused.reserve(scores.size());
  for (const auto& [label, score] : scores) fused.push_back({label, static_cast<float>(-score)});
  const std::size_t keep = std::min(k, fused.size());
  std::partial_sort(fused.begin(), fused.begin() + static_cast<std::ptrdiff_t>(keep), fused.end(),
                    [](const SearchResult& a, const SearchResult& b) {
                      return a.distance != b.distance ? a.distance < b.distance : a.label < b.label;
                    });
  fused.resize(keep);
  return fused;
}

std::vector<SearchResult> hybrid_search(const HnswIndex& vectors, const TextIndex& text, const float* query_vector,
                                        std::string_view query_text, std::size_t k, const HybridOptions& options) {
  const std::size_t candidates = std::max(options.candidates, k);
  return fuse_rankings({vectors.search(query_vector, candidates, {.ef = std::max(options.ef, candidates)}),
                        text.search(query_text, candidates)},
                       k, options.rrf_k, {options.vector_weight, options.text_weight});
}

}  // namespace epzero
