// Fuzzes MetadataIndex::load: arbitrary bytes as a metadata file must either load into an index that is
// safe to query, change and save, or throw. Never crash, read out of bounds or hit undefined behavior.
#include <exception>
#include <string>

#include "common.h"
#include "epzero/metadata.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  static const auto path = epzero::fuzz::scratch("metadata") / "input.meta";
  epzero::fuzz::write_bytes(path, data, size);
  try {
    epzero::MetadataIndex index = epzero::MetadataIndex::load(path.string());
    (void)index.select({{.field = "source", .any_of = {std::string("faq")}}, {.field = "year", .min = 2000.0}});
    (void)index.select({{.field = "year", .max = 1e9, .negate = true}});
    index.set(1, {{"source", std::string("new")}});
    index.remove(1);
    index.save(path.string());
  } catch (const std::exception&) {
    // Rejected cleanly: the expected outcome for corrupt input.
  }
  return 0;
}
