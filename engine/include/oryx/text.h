#pragma once

// Keyword search and hybrid retrieval.
//
// Embeddings capture meaning but blur exact terms: a product code, a person's name or an error
// message can rank below documents that are merely about the same topic. Production retrieval for
// RAG therefore usually runs a keyword search beside the vector search and merges the two rankings.
// TextIndex is the keyword side (Okapi BM25 over an inverted index), and hybrid_search() merges it
// with an HnswIndex search by reciprocal rank fusion (Cormack et al., 2009): each list contributes
// weight / (rrf_k + rank) per label, so neither retriever's raw scores need to be comparable.

#include <cstddef>
#include <cstdint>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "oryx/hnsw.h"

namespace oryx {

// Okapi BM25 over short documents, keyed by the same labels as the vector index.
//
// Tokens are maximal runs of letters and digits, lowercased (ASCII); bytes of multi-byte UTF-8
// characters count as letters, so words in other scripts stay whole (they are matched exactly, not
// case-folded). There is no stemming or stop-word list: BM25's idf already discounts common words.
//
// Thread safety: add() and remove() may run concurrently with search() and with each other.
class TextIndex {
 public:
  struct Options {
    float k1 = 1.2f;   // term-frequency saturation
    float b = 0.75f;   // document-length normalization
  };

  TextIndex() : TextIndex(Options{}) {}
  explicit TextIndex(Options options) : options_(options) {}

  // Indexes `text` under `label`, replacing any text the label had.
  void add(std::uint64_t label, std::string_view text);
  // Forgets `label`; returns false if it is not present.
  bool remove(std::uint64_t label);

  // The k best matches for `query`, best first. SearchResult::distance is the negated BM25 score,
  // so smaller is better, as everywhere else. Documents sharing no term with the query are omitted.
  std::vector<SearchResult> search(std::string_view query, std::size_t k) const;

  std::size_t size() const;
  bool contains(std::uint64_t label) const;

  void save(const std::string& path) const;
  static TextIndex load(const std::string& path);

  // The tokens `text` is indexed by (exposed for tests and for explaining matches).
  static std::vector<std::string> tokenize(std::string_view text);

  TextIndex(TextIndex&& other) noexcept;
  TextIndex& operator=(TextIndex&& other) noexcept;

 private:
  struct Posting {
    std::uint32_t doc;  // slot in docs_
    std::uint32_t tf;
  };
  struct Doc {
    std::uint64_t label;
    std::uint32_t length;  // tokens
    bool live;
    std::vector<std::uint32_t> terms;  // distinct term ids, to update document frequencies on remove
  };
  void remove_locked(std::uint64_t label);

  Options options_;
  mutable std::shared_mutex lock_;
  std::unordered_map<std::string, std::uint32_t> term_ids_;
  std::vector<std::vector<Posting>> postings_;  // by term id; postings of removed docs are skipped
  std::vector<std::uint32_t> live_df_;          // documents containing each term, among live ones
  std::vector<Doc> docs_;
  std::unordered_map<std::uint64_t, std::uint32_t> slot_of_;
  std::size_t live_ = 0;
  std::uint64_t total_length_ = 0;  // tokens over live documents
};

// Reciprocal rank fusion of ranked lists (each best first): a label scores the sum over lists of
// weights[i] / (rrf_k + rank), rank starting at 1; returns the k best, with distance = -score.
// `weights` may be empty (all 1).
std::vector<SearchResult> fuse_rankings(const std::vector<std::vector<SearchResult>>& lists, std::size_t k,
                                        double rrf_k = 60.0, const std::vector<double>& weights = {});

struct HybridOptions {
  std::size_t ef = 64;            // vector search beam width (at least `candidates`)
  std::size_t candidates = 50;    // how many results each retriever contributes to the fusion
  double rrf_k = 60.0;            // the fusion's rank offset (60 is the paper's default)
  double vector_weight = 1.0;
  double text_weight = 1.0;
};

// The k best labels for a query given both as a vector and as text: a vector search and a keyword
// search, merged by reciprocal rank fusion.
std::vector<SearchResult> hybrid_search(const HnswIndex& vectors, const TextIndex& text, const float* query_vector,
                                        std::string_view query_text, std::size_t k, const HybridOptions& options = {});

}  // namespace oryx
