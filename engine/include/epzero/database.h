#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <vector>

#include "epzero/hnsw.h"

namespace epzero {

enum class SyncMode {
  Always,  // fsync after every write: an acknowledged write survives power loss
  Batch,   // group commit: fsync on flush() (and at checkpoints); fastest durable bulk loading
  None,    // leave it to the OS: survives a process crash, not a power loss
};

struct DatabaseOptions {
  SyncMode sync = SyncMode::Always;
  // Checkpoint automatically once the write-ahead log grows past this many bytes (0 = never).
  std::size_t checkpoint_bytes = std::size_t{256} << 20;
  // Rebuild without tombstones at a checkpoint once this fraction of slots is deleted.
  double compact_ratio = 0.25;
};

// One logged write. Replication ships these verbatim, so a replica's log is the primary's log.
struct LogRecord {
  enum class Type : std::uint8_t { Upsert = 1, Remove = 2 };
  Type type = Type::Upsert;
  std::uint64_t lsn = 0;
  std::uint64_t label = 0;
  std::vector<float> vector;  // empty for Remove
};

struct RecoveryInfo {
  bool loaded_snapshot = false;
  std::uint64_t snapshot_lsn = 0;
  std::size_t replayed = 0;           // WAL records applied on top of the snapshot
  std::size_t truncated_bytes = 0;    // torn or corrupt tail removed from the WAL
};

// A durable vector collection: an HnswIndex plus a write-ahead log in a directory.
//
//   dir/snapshot-<lsn>.epzero  index snapshot containing every write up to log sequence number lsn
//   dir/CHECKPOINT           which lsn (and so which snapshot) is current, checksummed
//   dir/wal.log              records: [u32 length][u32 crc32c][u8 type][u64 lsn][u64 label][vector]
//
// Every write is appended to the log (and, with SyncMode::Always, fsynced) before it is applied,
// so a crash at any point loses nothing that was acknowledged. Recovery loads the current
// snapshot, replays records with a newer lsn, and truncates the log at the first incomplete or
// corrupt record (a torn final write). A checkpoint writes and fsyncs a new snapshot, then
// atomically renames a new CHECKPOINT over the old one, and only then truncates the log, so a
// crash at any step leaves a consistent snapshot + log pair.
//
// Thread safety: writes are serialized (log order is apply order); search() runs concurrently
// with writes and with other searches.
class Database {
 public:
  // Opens `dir`, creating it with `params` if it holds no database.
  static std::unique_ptr<Database> open(const std::string& dir, const HnswParams& params,
                                        const DatabaseOptions& options = {});
  ~Database();

  Database(const Database&) = delete;
  Database& operator=(const Database&) = delete;

  // SQ8 collections must be trained before the first write (the ranges are part of the snapshot
  // and are persisted by an immediate checkpoint).
  void train(const float* data, std::size_t n);

  // Inserts or replaces the vector for `label`; returns the logged record (for replication).
  LogRecord upsert(std::uint64_t label, const float* vector);
  // Many upserts with one log write and one fsync, applied on `threads` threads. Labels must be
  // distinct within the batch. Returns the logged records.
  std::vector<LogRecord> upsert_batch(const std::uint64_t* labels, const float* vectors, std::size_t n,
                                      std::size_t threads = 0);
  // Returns nothing if `label` is not present (nothing is logged).
  std::optional<LogRecord> remove(std::uint64_t label);

  // --- Replication ---
  // Replica side: applies records written by a primary, keeping their LSNs. Each batch must
  // continue exactly where this log ends (first lsn == last_lsn() + 1); a gap means this replica
  // missed writes and must be re-seeded with import_snapshot(). Durable per the sync mode.
  void apply(const std::vector<LogRecord>& records);
  // Primary side: every record after `lsn`, if the log still holds all of them (a checkpoint may
  // have folded older ones into the snapshot); otherwise nothing, and a snapshot must be shipped.
  std::optional<std::vector<LogRecord>> records_since(std::uint64_t lsn) const;
  // Checkpoints and returns the snapshot bytes and the LSN they contain.
  std::pair<std::uint64_t, std::string> export_snapshot();
  // Replaces all state with a snapshot taken on another node (installs it as this node's checkpoint).
  void import_snapshot(std::uint64_t lsn, const std::string& bytes);

  std::vector<SearchResult> search(const float* query, std::size_t k, const SearchOptions& options = {}) const;

  void flush();       // makes every write so far durable (SyncMode::Batch group commit)
  void checkpoint();  // snapshot + log truncation (+ compaction if many tombstones)

  const RecoveryInfo& recovery() const noexcept { return recovery_; }
  std::uint64_t last_lsn() const;
  std::size_t size() const;  // live vectors
  const HnswIndex& index() const { return *index_; }

 private:
  Database(std::string dir, const HnswParams& params, const DatabaseOptions& options);
  void recover();
  void append(const std::vector<std::uint8_t>& bytes, bool sync);
  void apply_upsert(std::uint64_t label, const float* vector);
  void apply_record(const LogRecord& record);
  void write_checkpoint_file(std::uint64_t lsn);
  void maybe_checkpoint();
  void checkpoint_locked();
  void compact_if_needed(bool force);

  std::string dir_;
  HnswParams params_;
  DatabaseOptions options_;
  std::unique_ptr<HnswIndex> index_;
  int wal_fd_ = -1;
  std::size_t wal_bytes_ = 0;
  std::uint64_t lsn_ = 0;
  std::uint64_t checkpoint_lsn_ = 0;  // everything <= this is in the snapshot, not the log
  RecoveryInfo recovery_;
  mutable std::mutex write_lock_;          // serializes writers: log order == apply order
  mutable std::shared_mutex index_lock_;   // exclusive only to swap in a compacted index
};

// CRC-32C (Castagnoli), as used by the log records.
std::uint32_t crc32c(const void* data, std::size_t size, std::uint32_t crc = 0) noexcept;

}  // namespace epzero
