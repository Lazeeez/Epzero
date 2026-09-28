#include "oryx/database.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include "oryx/parallel.h"

namespace oryx {
namespace {

namespace fs = std::filesystem;

constexpr std::uint8_t kUpsert = 1;
constexpr std::uint8_t kRemove = 2;
constexpr std::size_t kHeaderBytes = 8;                    // u32 length + u32 crc
constexpr std::size_t kFixedPayload = 1 + 8 + 8;           // type + lsn + label

std::array<std::uint32_t, 256> make_crc_table() {
  std::array<std::uint32_t, 256> table{};
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;  // reflected Castagnoli
    table[i] = c;
  }
  return table;
}

[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error(what + ": " + std::strerror(errno));
}

void write_all(int fd, const void* data, std::size_t size) {
  const auto* p = static_cast<const std::uint8_t*>(data);
  while (size > 0) {
    const ssize_t n = ::write(fd, p, size);
    if (n < 0) {
      if (errno == EINTR) continue;
      fail("write-ahead log write failed");
    }
    p += n;
    size -= static_cast<std::size_t>(n);
  }
}

void sync_fd(int fd) {
#if defined(__linux__)
  if (::fdatasync(fd) != 0) fail("fdatasync failed");
#else
  if (::fsync(fd) != 0) fail("fsync failed");
#endif
}

void sync_path(const fs::path& path, bool directory) {
  const int fd = ::open(path.c_str(), directory ? O_RDONLY | O_DIRECTORY : O_RDONLY);
  if (fd < 0) fail("open " + path.string());
  const int result = ::fsync(fd);
  ::close(fd);
  if (result != 0) fail("fsync " + path.string());
}

template <typename T>
void put(std::vector<std::uint8_t>& out, const T& value) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), p, p + sizeof(T));
}

// Appends one framed record: [u32 payload length][u32 crc32c(payload)][payload].
void frame(std::vector<std::uint8_t>& out, std::uint8_t type, std::uint64_t lsn, std::uint64_t label,
           const float* vector, std::size_t dim) {
  const std::size_t start = out.size();
  const auto length = static_cast<std::uint32_t>(kFixedPayload + (vector ? dim * sizeof(float) : 0));
  put(out, length);
  put(out, std::uint32_t{0});  // crc, filled in below
  put(out, type);
  put(out, lsn);
  put(out, label);
  if (vector) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(vector);
    out.insert(out.end(), p, p + dim * sizeof(float));
  }
  const std::uint32_t crc = crc32c(out.data() + start + kHeaderBytes, length);
  std::memcpy(out.data() + start + 4, &crc, sizeof(crc));
}

fs::path snapshot_path(const fs::path& dir, std::uint64_t lsn) {
  return dir / ("snapshot-" + std::to_string(lsn) + ".oryx");
}

}  // namespace

std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc) noexcept {
  static const std::array<std::uint32_t, 256> table = make_crc_table();
  const auto* p = static_cast<const std::uint8_t*>(data);
  crc = ~crc;
  for (std::size_t i = 0; i < size; ++i) crc = table[(crc ^ p[i]) & 0xff] ^ (crc >> 8);
  return ~crc;
}

Database::Database(std::string dir, const HnswParams& params, const DatabaseOptions& options)
    : dir_(std::move(dir)), params_(params), options_(options) {}

Database::~Database() {
  if (wal_fd_ >= 0) ::close(wal_fd_);
}

std::unique_ptr<Database> Database::open(const std::string& dir, const HnswParams& params,
                                         const DatabaseOptions& options) {
  fs::create_directories(dir);
  std::unique_ptr<Database> db(new Database(dir, params, options));
  db->recover();
  return db;
}

void Database::recover() {
  const fs::path dir(dir_);

  // 1. The current snapshot, if a checkpoint has ever completed.
  std::uint64_t snapshot_lsn = 0;
  const fs::path checkpoint = dir / "CHECKPOINT";
  if (fs::exists(checkpoint)) {
    std::uint8_t buffer[12];
    const int fd = ::open(checkpoint.c_str(), O_RDONLY);
    if (fd < 0) fail("open CHECKPOINT");
    const ssize_t n = ::read(fd, buffer, sizeof(buffer));
    ::close(fd);
    std::uint32_t crc;
    std::memcpy(&snapshot_lsn, buffer, 8);
    std::memcpy(&crc, buffer + 8, 4);
    if (n != 12 || crc != crc32c(buffer, 8)) throw std::runtime_error(dir_ + ": corrupt CHECKPOINT file");
    index_ = HnswIndex::load(snapshot_path(dir, snapshot_lsn).string());
    params_ = index_->params();
    recovery_.loaded_snapshot = true;
    recovery_.snapshot_lsn = snapshot_lsn;
  } else {
    index_ = std::make_unique<HnswIndex>(params_);
  }
  lsn_ = snapshot_lsn;

  // 2. Replay the log past the snapshot, stopping at the first torn or corrupt record.
  const fs::path wal = dir / "wal.log";
  wal_fd_ = ::open(wal.c_str(), O_RDWR | O_CREAT, 0644);
  if (wal_fd_ < 0) fail("open " + wal.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(fs::file_size(wal)));
  for (std::size_t done = 0; done < bytes.size();) {
    const ssize_t n = ::pread(wal_fd_, bytes.data() + done, bytes.size() - done, static_cast<off_t>(done));
    if (n <= 0) fail("read " + wal.string());
    done += static_cast<std::size_t>(n);
  }

  const std::size_t max_payload = kFixedPayload + params_.dim * sizeof(float);
  std::size_t pos = 0;
  while (pos + kHeaderBytes <= bytes.size()) {
    std::uint32_t length, crc;
    std::memcpy(&length, bytes.data() + pos, 4);
    std::memcpy(&crc, bytes.data() + pos + 4, 4);
    if (length < kFixedPayload || length > max_payload || pos + kHeaderBytes + length > bytes.size()) break;
    const std::uint8_t* payload = bytes.data() + pos + kHeaderBytes;
    if (crc32c(payload, length) != crc) break;
    const std::uint8_t type = payload[0];
    std::uint64_t lsn, label;
    std::memcpy(&lsn, payload + 1, 8);
    std::memcpy(&label, payload + 9, 8);
    const bool upsert = type == kUpsert && length == max_payload;
    if (!upsert && !(type == kRemove && length == kFixedPayload)) break;
    if (lsn > snapshot_lsn) {
      if (upsert) {
        std::vector<float> vector(params_.dim);
        std::memcpy(vector.data(), payload + kFixedPayload, params_.dim * sizeof(float));
        apply_upsert(label, vector.data());
      } else {
        index_->remove(label);
      }
      ++recovery_.replayed;
      lsn_ = std::max(lsn_, lsn);
    }
    pos += kHeaderBytes + length;
  }

  // 3. Drop the torn tail so new records are appended after the last good one.
  if (pos < bytes.size()) {
    recovery_.truncated_bytes = bytes.size() - pos;
    if (::ftruncate(wal_fd_, static_cast<off_t>(pos)) != 0) fail("truncate " + wal.string());
    sync_fd(wal_fd_);
  }
  if (::lseek(wal_fd_, static_cast<off_t>(pos), SEEK_SET) < 0) fail("seek " + wal.string());
  wal_bytes_ = pos;
}

void Database::append(const std::vector<std::uint8_t>& bytes, bool sync) {
  write_all(wal_fd_, bytes.data(), bytes.size());
  wal_bytes_ += bytes.size();
  if (sync) sync_fd(wal_fd_);
}

void Database::train(const float* data, std::size_t n) {
  std::lock_guard lock(write_lock_);
  index_->train(data, n);
  checkpoint_locked();  // the learned ranges live in the snapshot
}

void Database::apply_upsert(std::uint64_t label, const float* vector) {
  {
    std::shared_lock lock(index_lock_);
    if (index_->size() < params_.max_elements) {
      index_->remove(label);  // replace: the old vector becomes a tombstone
      index_->add(label, vector);
      return;
    }
  }
  compact_if_needed(/*force=*/true);  // every slot is used; reclaim tombstones
  std::shared_lock lock(index_lock_);
  if (index_->size() >= params_.max_elements) throw std::length_error("database is full");
  index_->remove(label);
  index_->add(label, vector);
}

void Database::upsert(std::uint64_t label, const float* vector) {
  std::lock_guard lock(write_lock_);
  std::vector<std::uint8_t> record;
  frame(record, kUpsert, lsn_ + 1, label, vector, params_.dim);
  append(record, options_.sync == SyncMode::Always);
  ++lsn_;
  apply_upsert(label, vector);
  maybe_checkpoint();
}

void Database::upsert_batch(const std::uint64_t* labels, const float* vectors, std::size_t n, std::size_t threads) {
  std::lock_guard lock(write_lock_);
  {
    std::shared_lock shared(index_lock_);
    if (index_->live_count() + n > params_.max_elements) throw std::length_error("database is full");
  }
  if (index_->size() + n > params_.max_elements) compact_if_needed(/*force=*/true);

  std::vector<std::uint8_t> records;
  records.reserve(n * (kHeaderBytes + kFixedPayload + params_.dim * sizeof(float)));
  for (std::size_t i = 0; i < n; ++i) frame(records, kUpsert, lsn_ + 1 + i, labels[i], vectors + i * params_.dim, params_.dim);
  append(records, options_.sync != SyncMode::None);  // one write, one fsync for the whole batch
  lsn_ += n;

  std::shared_lock shared(index_lock_);
  for (std::size_t i = 0; i < n; ++i) index_->remove(labels[i]);
  parallel_for(0, n, threads, [&](std::size_t i) { index_->add(labels[i], vectors + i * params_.dim); });
  shared.unlock();
  maybe_checkpoint();
}

bool Database::remove(std::uint64_t label) {
  std::lock_guard lock(write_lock_);
  {
    std::shared_lock shared(index_lock_);
    if (!index_->contains(label)) return false;
  }
  std::vector<std::uint8_t> record;
  frame(record, kRemove, lsn_ + 1, label, nullptr, params_.dim);
  append(record, options_.sync == SyncMode::Always);
  ++lsn_;
  {
    std::shared_lock shared(index_lock_);
    index_->remove(label);
  }
  maybe_checkpoint();
  return true;
}

std::vector<SearchResult> Database::search(const float* query, std::size_t k, const SearchOptions& options) const {
  std::shared_lock lock(index_lock_);
  return index_->search(query, k, options);
}

void Database::flush() {
  std::lock_guard lock(write_lock_);
  sync_fd(wal_fd_);
}

void Database::checkpoint() {
  std::lock_guard lock(write_lock_);
  checkpoint_locked();
}

void Database::maybe_checkpoint() {
  if (options_.checkpoint_bytes > 0 && wal_bytes_ >= options_.checkpoint_bytes) checkpoint_locked();
}

void Database::compact_if_needed(bool force) {
  const double deleted = static_cast<double>(index_->deleted_count());
  if (deleted == 0 || (!force && deleted < options_.compact_ratio * static_cast<double>(index_->size()))) return;
  auto compacted = index_->compacted();
  std::unique_lock lock(index_lock_);  // waits for in-flight searches, then swaps
  index_ = std::move(compacted);
}

void Database::checkpoint_locked() {
  compact_if_needed(/*force=*/false);
  const fs::path dir(dir_);
  const std::uint64_t lsn = lsn_;

  // 1. New snapshot under a temporary name, fsynced, then renamed into place.
  const fs::path snapshot = snapshot_path(dir, lsn);
  const fs::path temp = dir / "snapshot.tmp";
  {
    std::shared_lock lock(index_lock_);
    index_->save(temp.string());
  }
  sync_path(temp, false);
  fs::rename(temp, snapshot);
  sync_path(dir, true);

  // 2. Atomically point CHECKPOINT at it. Before this rename a crash recovers from the old
  //    snapshot + full log; after it, from the new snapshot, skipping records it contains.
  std::uint8_t buffer[12];
  std::memcpy(buffer, &lsn, 8);
  const std::uint32_t crc = crc32c(buffer, 8);
  std::memcpy(buffer + 8, &crc, 4);
  const fs::path checkpoint_temp = dir / "CHECKPOINT.tmp";
  {
    const int fd = ::open(checkpoint_temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) fail("open CHECKPOINT.tmp");
    write_all(fd, buffer, sizeof(buffer));
    sync_fd(fd);
    ::close(fd);
  }
  fs::rename(checkpoint_temp, dir / "CHECKPOINT");
  sync_path(dir, true);

  // 3. The log is now redundant: truncate it, and drop older snapshots.
  if (::ftruncate(wal_fd_, 0) != 0) fail("truncate wal.log");
  if (::lseek(wal_fd_, 0, SEEK_SET) < 0) fail("seek wal.log");
  sync_fd(wal_fd_);
  wal_bytes_ = 0;
  for (const auto& entry : fs::directory_iterator(dir)) {
    const std::string name = entry.path().filename().string();
    if (name.starts_with("snapshot-") && entry.path() != snapshot) fs::remove(entry.path());
  }
}

std::uint64_t Database::last_lsn() const {
  std::lock_guard lock(write_lock_);
  return lsn_;
}

std::size_t Database::size() const {
  std::shared_lock lock(index_lock_);
  return index_->live_count();
}

}  // namespace oryx
