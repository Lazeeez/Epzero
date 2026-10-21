// Fuzzes TextIndex::load: arbitrary bytes as a text index file must either load into an index
// that is safe to search and save, or throw. Never crash, read out of bounds or hit undefined behavior.
#include <exception>

#include "common.h"
#include "oryx/text.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  static const auto path = oryx::fuzz::scratch("text") / "input.idx";
  oryx::fuzz::write_bytes(path, data, size);
  try {
    oryx::TextIndex index = oryx::TextIndex::load(path.string());
    for (const char* query : {"red car", "a", "night city river 42"}) index.search(query, 10);
    index.add(1, "a new document");
    index.remove(1);
    index.save(path.string());
  } catch (const std::exception&) {
    // Rejected cleanly: the expected outcome for corrupt input.
  }
  return 0;
}
