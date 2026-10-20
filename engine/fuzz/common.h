// Shared helpers for the fuzz targets: a scratch directory per process and a small index shape.
#pragma once

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include "oryx/hnsw.h"

namespace oryx::fuzz {

constexpr std::size_t kDim = 4;
constexpr std::size_t kMaxElements = 64;

inline HnswParams small_params() {
  return {.dim = kDim, .metric = Metric::L2, .max_elements = kMaxElements, .M = 4, .ef_construction = 16};
}

// A directory of this process's own (fuzzers can run several jobs side by side).
inline std::filesystem::path scratch(const std::string& name) {
  auto dir = std::filesystem::temp_directory_path() / ("oryx-fuzz-" + std::to_string(getpid())) / name;
  std::filesystem::create_directories(dir);
  return dir;
}

inline void write_bytes(const std::filesystem::path& path, const std::uint8_t* data, std::size_t size) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
}

// Whatever was loaded must also be safe to use: a search touches every structure it rebuilt.
inline void exercise(const HnswIndex& index) {
  const float query[kDim] = {0.1f, 0.2f, 0.3f, 0.4f};
  for (const auto& hit : index.search(query, 5, {.ef = 16})) (void)hit;
}

}  // namespace oryx::fuzz
