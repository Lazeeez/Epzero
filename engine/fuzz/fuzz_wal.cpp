// Fuzzes crash recovery: arbitrary bytes as a database's write-ahead log. Opening the database
// must recover a consistent prefix (truncating a torn or corrupt tail) or throw, and the
// recovered collection must be safe to search and to write to.
#include <exception>
#include <filesystem>

#include "common.h"
#include "oryx/database.h"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  using namespace oryx::fuzz;
  static const auto dir = scratch("wal");
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  write_bytes(dir / "wal.log", data, size);
  try {
    oryx::DatabaseOptions options;
    options.sync = oryx::SyncMode::None;
    options.checkpoint_bytes = 0;
    auto db = oryx::Database::open(dir.string(), small_params(), options);
    const float query[kDim] = {0.1f, 0.2f, 0.3f, 0.4f};
    for (const auto& hit : db->search(query, 5, {.ef = 16})) (void)hit;
    db->upsert(7, query);  // appending after recovery must still work
  } catch (const std::exception&) {
    // Rejected cleanly (e.g. more records than the index can hold).
  }
  return 0;
}
