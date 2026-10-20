// Writes one valid input per fuzz target, so fuzzing starts from real files and mutates them,
// rather than having to discover the formats from nothing:
//   fuzz_seeds <dir>  ->  <dir>/snapshot/valid, <dir>/graph/valid, <dir>/wal/valid
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#include "common.h"
#include "oryx/database.h"

namespace fs = std::filesystem;
using namespace oryx::fuzz;

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: fuzz_seeds <corpus dir>\n");
    return 2;
  }
  const fs::path out = argv[1];
  for (const char* name : {"snapshot", "graph", "wal"}) fs::create_directories(out / name);

  std::vector<float> vectors(40 * kDim);
  for (std::size_t i = 0; i < vectors.size(); ++i) vectors[i] = static_cast<float>((i * 7919) % 101) / 101.0f;

  oryx::HnswIndex index(small_params());
  for (std::size_t i = 0; i < 40; ++i) index.add(i, vectors.data() + i * kDim);
  index.save((out / "snapshot" / "valid").string());

  const auto graph = index.export_graph();
  std::ofstream(out / "graph" / "valid", std::ios::binary)
      .write(reinterpret_cast<const char*>(graph.data()), static_cast<std::streamsize>(graph.size()));

  // A log with upserts, a replacement and a remove.
  const fs::path dir = scratch("seed-wal");
  fs::remove_all(dir);
  {
    oryx::DatabaseOptions options;
    options.checkpoint_bytes = 0;
    auto db = oryx::Database::open(dir.string(), small_params(), options);
    for (std::size_t i = 0; i < 12; ++i) db->upsert(i, vectors.data() + i * kDim);
    db->upsert(3, vectors.data() + 20 * kDim);
    db->remove(5);
  }
  fs::copy_file(dir / "wal.log", out / "wal" / "valid", fs::copy_options::overwrite_existing);
  fs::remove_all(dir.parent_path());
  std::printf("seeds written to %s\n", out.string().c_str());
  return 0;
}
