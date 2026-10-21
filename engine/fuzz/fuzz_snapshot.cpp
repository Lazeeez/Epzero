// Fuzzes HnswIndex::load: arbitrary bytes as a snapshot file must either load into an index that
// is safe to search, or throw. Never crash, read out of bounds or hit undefined behavior.
#include <exception>

#include "common.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  static const auto path = epzero::fuzz::scratch("snapshot") / "input.epzero";
  epzero::fuzz::write_bytes(path, data, size);
  try {
    // An untrusted snapshot is loaded with a memory cap: a header may legitimately declare a large
    // capacity, so without one, loading a tiny file can still allocate gigabytes.
    const auto index = epzero::HnswIndex::load(path.string(), std::size_t{256} << 20);
    epzero::fuzz::exercise(*index);
  } catch (const std::exception&) {
    // Rejected cleanly: the expected outcome for corrupt input.
  }
  return 0;
}
