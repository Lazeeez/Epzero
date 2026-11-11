#pragma once

// The Porter stemmer (M. F. Porter, "An algorithm for suffix stripping", 1980), as in Porter's own
// reference implementation, with its departures from the paper (-bli to -ble, -logi to -log): the
// version Lucene's PorterStemFilter and so Anserini's BM25 baselines use. It reduces English words to a
// common stem ("connections", "connected" and "connecting" to "connect"), which lifted BM25 by 0.017-0.027
// nDCG@10 on BEIR's SciFact and NFCorpus. Words of anything but the letters a-z are returned whole.

#include <string>
#include <string_view>

namespace epzero::porter {

class Stemmer {
 public:
  std::string stem(std::string_view word) {
    if (word.size() <= 2) return std::string(word);
    for (const char c : word) {
      if (c < 'a' || c > 'z') return std::string(word);
    }
    b_.assign(word);
    k_ = static_cast<int>(b_.size()) - 1;
    step1ab();
    if (k_ > 0) {
      step1c();
      step2();
      step3();
      step4();
      step5();
    }
    b_.resize(static_cast<std::size_t>(k_ + 1));
    return b_;
  }

 private:
  // b_[0..k_] is the word being stemmed; j_ marks the end of its stem before a matched suffix.
  std::string b_;
  int k_ = 0;
  int j_ = 0;

  bool cons(int i) const {
    switch (b_[i]) {
      case 'a': case 'e': case 'i': case 'o': case 'u': return false;
      case 'y': return i == 0 ? true : !cons(i - 1);
      default: return true;
    }
  }

  // The number of vowel-consonant sequences in b_[0..j_]: [C](VC)^m[V].
  int m() const {
    int n = 0;
    int i = 0;
    while (true) {
      if (i > j_) return n;
      if (!cons(i)) break;
      ++i;
    }
    ++i;
    while (true) {
      while (true) {
        if (i > j_) return n;
        if (cons(i)) break;
        ++i;
      }
      ++i;
      ++n;
      while (true) {
        if (i > j_) return n;
        if (!cons(i)) break;
        ++i;
      }
      ++i;
    }
  }

  bool vowel_in_stem() const {
    for (int i = 0; i <= j_; ++i) {
      if (!cons(i)) return true;
    }
    return false;
  }

  bool double_consonant(int j) const { return j >= 1 && b_[j] == b_[j - 1] && cons(j); }

  // consonant-vowel-consonant ending at i, the last not w, x or y: "hop" (to restore "hope").
  bool cvc(int i) const {
    if (i < 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
    return b_[i] != 'w' && b_[i] != 'x' && b_[i] != 'y';
  }

  bool ends(std::string_view s) {
    const int length = static_cast<int>(s.size());
    if (length > k_ + 1) return false;
    if (std::string_view(b_).substr(static_cast<std::size_t>(k_ - length + 1), s.size()) != s) return false;
    j_ = k_ - length;
    return true;
  }

  void set_to(std::string_view s) {
    b_.replace(static_cast<std::size_t>(j_ + 1), static_cast<std::size_t>(k_ - j_), s);
    k_ = j_ + static_cast<int>(s.size());
  }

  void replace_if_measured(std::string_view s) {
    if (m() > 0) set_to(s);
  }

  // Plurals and -ed or -ing: caresses -> caress, ponies -> poni, meeting -> meet, hopping -> hop.
  void step1ab() {
    if (b_[k_] == 's') {
      if (ends("sses")) {
        k_ -= 2;
      } else if (ends("ies")) {
        set_to("i");
      } else if (b_[k_ - 1] != 's') {
        --k_;
      }
    }
    if (ends("eed")) {
      if (m() > 0) --k_;
    } else if ((ends("ed") || ends("ing")) && vowel_in_stem()) {
      k_ = j_;
      if (ends("at")) {
        set_to("ate");
      } else if (ends("bl")) {
        set_to("ble");
      } else if (ends("iz")) {
        set_to("ize");
      } else if (double_consonant(k_)) {
        --k_;
        const char c = b_[k_];
        if (c == 'l' || c == 's' || c == 'z') ++k_;
      } else {
        j_ = k_;
        if (m() == 1 && cvc(k_)) set_to("e");
      }
    }
  }

  // A final y after a vowel in the stem becomes i: happy -> happi.
  void step1c() {
    if (ends("y") && vowel_in_stem()) b_[k_] = 'i';
  }

  // Double suffixes to single ones, when the stem has a measure above 0: -ization -> -ize.
  void step2() {
    struct Rule {
      std::string_view from, to;
    };
    const auto apply = [&](std::initializer_list<Rule> rules) {
      for (const Rule& r : rules) {
        if (ends(r.from)) {
          replace_if_measured(r.to);
          return;
        }
      }
    };
    switch (b_[k_ - 1]) {
      case 'a': apply({{"ational", "ate"}, {"tional", "tion"}}); break;
      case 'c': apply({{"enci", "ence"}, {"anci", "ance"}}); break;
      case 'e': apply({{"izer", "ize"}}); break;
      case 'l': apply({{"bli", "ble"}, {"alli", "al"}, {"entli", "ent"}, {"eli", "e"}, {"ousli", "ous"}}); break;
      case 'o': apply({{"ization", "ize"}, {"ation", "ate"}, {"ator", "ate"}}); break;
      case 's': apply({{"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"}, {"ousness", "ous"}}); break;
      case 't': apply({{"aliti", "al"}, {"iviti", "ive"}, {"biliti", "ble"}}); break;
      case 'g': apply({{"logi", "log"}}); break;
      default: break;
    }
  }

  // -ic-, -full, -ness and the like.
  void step3() {
    struct Rule {
      std::string_view from, to;
    };
    const auto apply = [&](std::initializer_list<Rule> rules) {
      for (const Rule& r : rules) {
        if (ends(r.from)) {
          replace_if_measured(r.to);
          return;
        }
      }
    };
    switch (b_[k_]) {
      case 'e': apply({{"icate", "ic"}, {"ative", ""}, {"alize", "al"}}); break;
      case 'i': apply({{"iciti", "ic"}}); break;
      case 'l': apply({{"ical", "ic"}, {"ful", ""}}); break;
      case 's': apply({{"ness", ""}}); break;
      default: break;
    }
  }

  // -ant, -ence and the like, when the stem has a measure above 1.
  void step4() {
    const auto any = [&](std::initializer_list<std::string_view> suffixes) {
      for (const auto s : suffixes) {
        if (ends(s)) return true;
      }
      return false;
    };
    bool matched = false;
    switch (b_[k_ - 1]) {
      case 'a': matched = any({"al"}); break;
      case 'c': matched = any({"ance", "ence"}); break;
      case 'e': matched = any({"er"}); break;
      case 'i': matched = any({"ic"}); break;
      case 'l': matched = any({"able", "ible"}); break;
      case 'n': matched = any({"ant", "ement", "ment", "ent"}); break;
      case 'o':
        matched = (ends("ion") && j_ >= 0 && (b_[j_] == 's' || b_[j_] == 't')) || ends("ou");
        break;
      case 's': matched = any({"ism"}); break;
      case 't': matched = any({"ate", "iti"}); break;
      case 'u': matched = any({"ous"}); break;
      case 'v': matched = any({"ive"}); break;
      case 'z': matched = any({"ize"}); break;
      default: break;
    }
    if (matched && m() > 1) k_ = j_;
  }

  // A final -e, and -ll to -l, when the measure allows.
  void step5() {
    j_ = k_;
    if (b_[k_] == 'e') {
      const int a = m();
      if (a > 1 || (a == 1 && !cvc(k_ - 1))) --k_;
    }
    if (b_[k_] == 'l' && double_consonant(k_) && m() > 1) --k_;
  }
};

}  // namespace epzero::porter
