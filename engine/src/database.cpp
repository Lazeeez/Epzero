#include "epzero/database.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>

#include "epzero/parallel.h"

namespace epzero {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kHeaderBytes = 8;           // u32 length + u32 crc
constexpr std::size_t kFixedPayload = 1 + 8 + 8;  // type + lsn + label

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
      fail("write failed");
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

// Writes `bytes` to `path` crash-safely: temporary file, fsync, rename, fsync the directory.
void write_file_atomically(const fs::path& path, const void* bytes, std::size_t size) {
  const fs::path temp = path.string() + ".tmp";
  const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) fail("open " + temp.string());
  write_all(fd, bytes, size);
  sync_fd(fd);
  ::close(fd);
  fs::rename(temp, path);
  sync_path(path.parent_path(), true);
}

std::vector<std::uint8_t> read_file(int fd, std::size_t size, const std::string& name) {
  std::vector<std::uint8_t> bytes(size);
  for (std::size_t done = 0; done < size;) {
    const ssize_t n = ::pread(fd, bytes.data() + done, size - done, static_cast<off_t>(done));
    if (n <= 0) fail("read " + name);
    done += static_cast<std::size_t>(n);
  }
  return bytes;
}

template <typename T>
void put(std::vector<std::uint8_t>& out, const T& value) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
  out.insert(out.end(), p, p + sizeof(T));
}

// Appends one framed record: [u32 payload length][u32 crc32c(payload)][type][lsn][label][vector].
void frame(std::vector<std::uint8_t>& out, const LogRecord& record) {
  const std::size_t start = out.size();
  const auto length = static_cast<std::uint32_t>(kFixedPayload + record.vector.size() * sizeof(float));
  put(out, length);
  put(out, std::uint32_t{0});  // crc, filled in below
  put(out, static_cast<std::uint8_t>(record.type));
  put(out, record.lsn);
  put(out, record.label);
  const auto* p = reinterpret_cast<const std::uint8_t*>(record.vector.data());
  out.insert(out.end(), p, p + record.vector.size() * sizeof(float));
  const std::uint32_t crc = crc32c(out.data() + start + kHeaderBytes, length);
  std::memcpy(out.data() + start + 4, &crc, sizeof(crc));
}

// Parses records from the start of `bytes` until the first torn or corrupt one, calling
// `on_record` for each good record. Returns the length of the good prefix.
std::size_t parse_log(const std::vector<std::uint8_t>& bytes, std::size_t dim,
                      const std::function<void(LogRecord&&)>& on_record) {
  const std::size_t upsert_payload = kFixedPayload + dim * sizeof(float);
  std::size_t pos = 0;
  while (pos + kHeaderBytes <= bytes.size()) {
    std::uint32_t length, crc;
    std::memcpy(&length, bytes.data() + pos, 4);
    std::memcpy(&crc, bytes.data() + pos + 4, 4);
    if (length < kFixedPayload || length > upsert_payload || pos + kHeaderBytes + length > bytes.size()) break;
    const std::uint8_t* payload = bytes.data() + pos + kHeaderBytes;
    if (crc32c(payload, length) != crc) break;
    LogRecord record;
    record.type = static_cast<LogRecord::Type>(payload[0]);
    std::memcpy(&record.lsn, payload + 1, 8);
    std::memcpy(&record.label, payload + 9, 8);
    const bool upsert = record.type == LogRecord::Type::Upsert && length == upsert_payload;
    if (!upsert && !(record.type == LogRecord::Type::Remove && length == kFixedPayload)) break;
    if (upsert) {
      record.vector.resize(dim);
      std::memcpy(record.vector.data(), payload + kFixedPayload, dim * sizeof(float));
    }
    on_record(std::move(record));
    pos += kHeaderBytes + length;
  }
  return pos;
}

fs::path snapshot_path(const fs::path& dir, std::uint64_t lsn) {
  return dir / ("snapshot-" + std::to_string(lsn) + ".epzero");
}

void remove_other_snapshots(const fs::path& dir, const fs::path& keep) {
  for (const auto& entry : fs::directory_iterator(dir)) {
    if (entry.path().filename().string().starts_with("snapshot-") && entry.path() != keep) fs::remove(entry.path());
  }
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
  lsn_ = checkpoint_lsn_ = snapshot_lsn;

  // 2. Replay the log past the snapshot, stopping at the first torn or corrupt record.
  const fs::path wal = dir / "wal.log";
  wal_fd_ = ::open(wal.c_str(), O_RDWR | O_CREAT, 0644);
  if (wal_fd_ < 0) fail("open " + wal.string());
  const auto bytes = read_file(wal_fd_, static_cast<std::size_t>(fs::file_size(wal)), wal.string());
  const std::size_t good = parse_log(bytes, params_.dim, [&](LogRecord&& record) {
    if (record.lsn <= snapshot_lsn) return;  // already in the snapshot
    apply_record(record);
    ++recovery_.replayed;
    lsn_ = std::max(lsn_, record.lsn);
  });

  // 3. Drop the torn tail so new records are appended after the last good one.
  if (good < bytes.size()) {
    recovery_.truncated_bytes = bytes.size() - good;
    if (::ftruncate(wal_fd_, static_cast<off_t>(good)) != 0) fail("truncate " + wal.string());
    sync_fd(wal_fd_);
  }
  if (::lseek(wal_fd_, static_cast<off_t>(good), SEEK_SET) < 0) fail("seek " + wal.string());
  wal_bytes_ = good;
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

void Database::apply_record(const LogRecord& record) {
  if (record.type == LogRecord::Type::Upsert) {
    apply_upsert(record.label, record.vector.data());
  } else {
    std::shared_lock lock(index_lock_);
    index_->remove(record.label);
  }
}

LogRecord Database::upsert(std::uint64_t label, const float* vector) {
  std::lock_guard lock(write_lock_);
  LogRecord record{LogRecord::Type::Upsert, lsn_ + 1, label, std::vector<float>(vector, vector + params_.dim)};
  std::vector<std::uint8_t> bytes;
  frame(bytes, record);
  append(bytes, options_.sync == SyncMode::Always);
  ++lsn_;
  apply_upsert(label, vector);
  maybe_checkpoint();
  return record;
}

std::vector<LogRecord> Database::upsert_batch(const std::uint64_t* labels, const float* vectors, std::size_t n,
                                              std::size_t threads) {
  std::lock_guard lock(write_lock_);
  {
    std::shared_lock shared(index_lock_);
    if (index_->live_count() + n > params_.max_elements) throw std::length_error("database is full");
  }
  if (index_->size() + n > params_.max_elements) compact_if_needed(/*force=*/true);

  std::vector<LogRecord> records(n);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(n * (kHeaderBytes + kFixedPayload + params_.dim * sizeof(float)));
  for (std::size_t i = 0; i < n; ++i) {
    const float* v = vectors + i * params_.dim;
    records[i] = {LogRecord::Type::Upsert, lsn_ + 1 + i, labels[i], std::vector<float>(v, v + params_.dim)};
    frame(bytes, records[i]);
  }
  append(bytes, options_.sync != SyncMode::None);  // one write, one fsync for the whole batch
  lsn_ += n;

  std::shared_lock shared(index_lock_);
  for (std::size_t i = 0; i < n; ++i) index_->remove(labels[i]);
  parallel_for(0, n, threads, [&](std::size_t i) { index_->add(labels[i], vectors + i * params_.dim); });
  shared.unlock();
  maybe_checkpoint();
  return records;
}

std::optional<LogRecord> Database::remove(std::uint64_t label) {
  std::lock_guard lock(write_lock_);
  {
    std::shared_lock shared(index_lock_);
    if (!index_->contains(label)) return std::nullopt;
  }
  LogRecord record{LogRecord::Type::Remove, lsn_ + 1, label, {}};
  std::vector<std::uint8_t> bytes;
  frame(bytes, record);
  append(bytes, options_.sync == SyncMode::Always);
  ++lsn_;
  apply_record(record);
  maybe_checkpoint();
  return record;
}

void Database::apply(const std::vector<LogRecord>& records) {
  if (records.empty()) return;
  std::lock_guard lock(write_lock_);
  std::vector<std::uint8_t> bytes;
  for (std::size_t i = 0; i < records.size(); ++i) {
    if (records[i].lsn != lsn_ + 1 + i) {
      throw std::runtime_error("replica out of sync: expected lsn " + std::to_string(lsn_ + 1 + i) + ", got " +
                               std::to_string(records[i].lsn));
    }
    if (records[i].type == LogRecord::Type::Upsert && records[i].vector.size() != params_.dim) {
      throw std::invalid_argument("replicated vector has the wrong dimension");
    }
    frame(bytes, records[i]);
  }
  append(bytes, options_.sync != SyncMode::None);
  for (const LogRecord& record : records) apply_record(record);
  lsn_ += records.size();
  maybe_checkpoint();
}

std::optional<std::vector<LogRecord>> Database::records_since(std::uint64_t lsn) const {
  std::lock_guard lock(write_lock_);
  if (lsn < checkpoint_lsn_) return std::nullopt;  // folded into the snapshot; ship that instead
  std::vector<LogRecord> out;
  if (lsn >= lsn_) return out;
  const auto bytes = read_file(wal_fd_, wal_bytes_, "wal.log");
  parse_log(bytes, params_.dim, [&](LogRecord&& record) {
    if (record.lsn > lsn) out.push_back(std::move(record));
  });
  return out;
}

std::pair<std::uint64_t, std::string> Database::export_snapshot() {
  std::lock_guard lock(write_lock_);
  checkpoint_locked();
  std::ifstream in(snapshot_path(dir_, lsn_), std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (!in.good() && !in.eof()) throw std::runtime_error("cannot read snapshot");
  return {lsn_, std::move(bytes)};
}

void Database::import_snapshot(std::uint64_t lsn, const std::string& bytes) {
  std::lock_guard lock(write_lock_);
  const fs::path dir(dir_);
  const fs::path snapshot = snapshot_path(dir, lsn);
  write_file_atomically(snapshot, bytes.data(), bytes.size());
  auto loaded = HnswIndex::load(snapshot.string());  // validates before anything is replaced
  write_checkpoint_file(lsn);

  if (::ftruncate(wal_fd_, 0) != 0) fail("truncate wal.log");
  if (::lseek(wal_fd_, 0, SEEK_SET) < 0) fail("seek wal.log");
  sync_fd(wal_fd_);
  wal_bytes_ = 0;
  {
    std::unique_lock exclusive(index_lock_);
    index_ = std::move(loaded);
    params_ = index_->params();
  }
  lsn_ = checkpoint_lsn_ = lsn;
  remove_other_snapshots(dir, snapshot);
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

void Database::write_checkpoint_file(std::uint64_t lsn) {
  std::uint8_t buffer[12];
  std::memcpy(buffer, &lsn, 8);
  const std::uint32_t crc = crc32c(buffer, 8);
  std::memcpy(buffer + 8, &crc, 4);
  write_file_atomically(fs::path(dir_) / "CHECKPOINT", buffer, sizeof(buffer));
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
  write_checkpoint_file(lsn);
  checkpoint_lsn_ = lsn;

  // 3. The log is now redundant: truncate it, and drop older snapshots.
  if (::ftruncate(wal_fd_, 0) != 0) fail("truncate wal.log");
  if (::lseek(wal_fd_, 0, SEEK_SET) < 0) fail("seek wal.log");
  sync_fd(wal_fd_);
  wal_bytes_ = 0;
  remove_other_snapshots(dir, snapshot);
}

std::uint64_t Database::last_lsn() const {
  std::lock_guard lock(write_lock_);
  return lsn_;
}

std::size_t Database::size() const {
  std::shared_lock lock(index_lock_);
  return index_->live_count();
}

}  // namespace epzero
