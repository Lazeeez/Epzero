#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <set>

#if !defined(__EMSCRIPTEN__)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "epzero/database.h"
#include "epzero/dataset.h"

namespace epzero {
namespace {

namespace fs = std::filesystem;
constexpr std::size_t kDim = 16;

class DatabaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = fs::temp_directory_path() / ("epzero_db_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                                        "_" + ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::remove_all(dir_);
    data_ = make_clustered(2000, 50, kDim, 20, 5);
  }
  void TearDown() override { fs::remove_all(dir_); }

  HnswParams params(std::size_t max_elements = 3000) const {
    return {.dim = kDim, .metric = Metric::L2, .max_elements = max_elements, .M = 12, .ef_construction = 64};
  }
  std::unique_ptr<Database> open(DatabaseOptions options = {}) { return Database::open(dir_.string(), params(), options); }
  const float* row(std::size_t i) const { return data_.base.row(i); }
  fs::path wal() const { return dir_ / "wal.log"; }

  fs::path dir_;
  SyntheticData data_;
};

// Same labels, and each label's nearest neighbor is itself (so its vector survived intact).
// `replaced` counts live labels left out of `live` because their vector changed.
void expect_contents(const Database& db, const SyntheticData& data, const std::set<std::size_t>& live,
                     std::size_t replaced = 0) {
  EXPECT_EQ(db.size(), live.size() + replaced);
  for (std::size_t i : live) {
    EXPECT_TRUE(db.index().contains(i)) << i;
    const auto hits = db.search(data.base.row(i), 1, {.ef = 64});
    ASSERT_FALSE(hits.empty());
    EXPECT_EQ(hits[0].label, i);
    EXPECT_FLOAT_EQ(hits[0].distance, 0.0f);
  }
}

TEST_F(DatabaseTest, RecoversWritesFromTheLog) {
  std::set<std::size_t> live;
  {
    auto db = open();
    for (std::size_t i = 0; i < 500; ++i) {
      db->upsert(i, row(i));
      live.insert(i);
    }
    for (std::size_t i = 0; i < 500; i += 7) {
      EXPECT_TRUE(db->remove(i));
      live.erase(i);
    }
    EXPECT_FALSE(db->remove(0));        // already gone: nothing logged
    db->upsert(3, row(1000));           // replace a vector
    live.erase(3);                      // still live, but no longer its own nearest neighbor
  }  // no checkpoint, no clean shutdown step: everything must come back from wal.log
  auto db = open();
  EXPECT_FALSE(db->recovery().loaded_snapshot);
  EXPECT_EQ(db->recovery().replayed, 500 + 72 + 1);
  EXPECT_EQ(db->recovery().truncated_bytes, 0u);
  expect_contents(*db, data_, live, /*replaced=*/1);
  EXPECT_EQ(db->search(row(1000), 1, {.ef = 64})[0].label, 3u);  // replaced vector
  EXPECT_EQ(db->last_lsn(), 500u + 72 + 1);
}

TEST_F(DatabaseTest, TornFinalRecordIsTruncatedAtEveryLength) {
  {
    auto db = open();
    for (std::size_t i = 0; i < 20; ++i) db->upsert(i, row(i));
  }
  const auto full = fs::file_size(wal());
  const std::size_t record = 8 + 17 + kDim * sizeof(float);
  std::vector<char> original(full);
  std::ifstream(wal(), std::ios::binary).read(original.data(), static_cast<std::streamsize>(full));

  // Cut the last record at every possible byte: recovery keeps the first 19 and drops the rest.
  for (std::size_t cut = 1; cut < record; ++cut) {
    std::ofstream(wal(), std::ios::binary | std::ios::trunc).write(original.data(), static_cast<std::streamsize>(full - cut));
    auto db = open();
    EXPECT_EQ(db->recovery().replayed, 19u) << "cut " << cut;
    EXPECT_EQ(db->recovery().truncated_bytes, record - cut);
    EXPECT_FALSE(db->index().contains(19));
    EXPECT_EQ(fs::file_size(wal()), full - record);
    db->upsert(19, row(19));  // appends cleanly after the truncation point
  }
  auto db = open();
  EXPECT_EQ(db->recovery().replayed, 20u);
  EXPECT_TRUE(db->index().contains(19));
}

TEST_F(DatabaseTest, CorruptRecordStopsReplay) {
  {
    auto db = open();
    for (std::size_t i = 0; i < 10; ++i) db->upsert(i, row(i));
  }
  const std::size_t record = 8 + 17 + kDim * sizeof(float);
  {
    std::fstream f(wal(), std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(static_cast<std::streamoff>(record * 6 + 30));  // flip a byte inside record #7's vector
    char byte;
    f.read(&byte, 1);
    byte ^= 0x5a;
    f.seekp(static_cast<std::streamoff>(record * 6 + 30));
    f.write(&byte, 1);
  }
  auto db = open();
  EXPECT_EQ(db->recovery().replayed, 6u);  // everything before the checksum mismatch
  EXPECT_EQ(db->recovery().truncated_bytes, record * 4);
  EXPECT_TRUE(db->index().contains(5));
  EXPECT_FALSE(db->index().contains(6));
}

TEST_F(DatabaseTest, CheckpointThenMoreWritesThenCrash) {
  std::set<std::size_t> live;
  {
    auto db = open();
    for (std::size_t i = 0; i < 300; ++i) {
      db->upsert(i, row(i));
      live.insert(i);
    }
    db->checkpoint();
    EXPECT_EQ(fs::file_size(wal()), 0u);
    for (std::size_t i = 300; i < 400; ++i) {
      db->upsert(i, row(i));
      live.insert(i);
    }
    db->remove(10);
    live.erase(10);
  }
  // A crashed checkpoint may leave a temporary snapshot behind; it must be ignored.
  std::ofstream(dir_ / "snapshot.tmp") << "half-written";
  auto db = open();
  EXPECT_TRUE(db->recovery().loaded_snapshot);
  EXPECT_EQ(db->recovery().snapshot_lsn, 300u);
  EXPECT_EQ(db->recovery().replayed, 101u);
  expect_contents(*db, data_, live);
}

TEST_F(DatabaseTest, AutomaticCheckpointAndCompaction) {
  DatabaseOptions options;
  options.checkpoint_bytes = 20'000;  // a checkpoint every ~240 records
  options.compact_ratio = 0.2;
  std::set<std::size_t> live;
  {
    auto db = open(options);
    for (std::size_t round = 0; round < 3; ++round) {
      for (std::size_t i = 0; i < 400; ++i) {
        db->upsert(i, row(i + round * 400));  // rewrite the same labels: tombstones pile up
        live.insert(i);
      }
    }
    EXPECT_LT(db->index().deleted_count(), 400u);  // compaction reclaimed most tombstones
  }
  auto db = open(options);
  EXPECT_TRUE(db->recovery().loaded_snapshot);
  EXPECT_EQ(db->size(), 400u);
  for (std::size_t i = 0; i < 400; i += 37) EXPECT_EQ(db->search(row(i + 800), 1, {.ef = 64})[0].label, i);
}

TEST_F(DatabaseTest, BatchUpsertAndSq8Persist) {
  HnswParams p = params();
  p.quantization = Quantization::SQ8;
  std::vector<std::uint64_t> labels(1000);
  for (std::size_t i = 0; i < labels.size(); ++i) labels[i] = i;
  {
    DatabaseOptions options;
    options.sync = SyncMode::Batch;
    auto db = Database::open(dir_.string(), p, options);
    db->train(data_.base.data.data(), 2000);  // checkpoints the trained ranges
    db->upsert_batch(labels.data(), data_.base.data.data(), labels.size(), 4);
  }
  auto db = Database::open(dir_.string(), params(), {});  // params ignored: the snapshot wins
  EXPECT_EQ(db->index().params().quantization, Quantization::SQ8);
  EXPECT_EQ(db->recovery().replayed, 1000u);
  for (std::size_t i = 0; i < 1000; i += 50) EXPECT_EQ(db->search(row(i), 1, {.ef = 64})[0].label, i);
}

// ---------- Replication primitives ----------

TEST_F(DatabaseTest, ReplicaMirrorsPrimaryLog) {
  auto primary = Database::open((dir_ / "primary").string(), params());
  auto replica = Database::open((dir_ / "replica").string(), params());
  for (std::size_t i = 0; i < 200; ++i) replica->apply({primary->upsert(i, row(i))});
  replica->apply({*primary->remove(7)});
  std::vector<std::uint64_t> labels = {500, 501, 502};
  replica->apply(primary->upsert_batch(labels.data(), row(500), labels.size()));

  EXPECT_EQ(replica->last_lsn(), primary->last_lsn());
  EXPECT_EQ(replica->size(), primary->size());
  for (std::size_t q = 0; q < 20; ++q) {
    const auto a = primary->search(data_.queries.row(q), 5, {.ef = 64});
    const auto b = replica->search(data_.queries.row(q), 5, {.ef = 64});
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_EQ(a[i].label, b[i].label);
  }
  // The replica's own log recovers it just like a primary's.
  replica.reset();
  replica = Database::open((dir_ / "replica").string(), params());
  EXPECT_EQ(replica->last_lsn(), primary->last_lsn());
}

TEST_F(DatabaseTest, ReplicaRejectsGapsAndCatchesUp) {
  auto primary = Database::open((dir_ / "primary").string(), params());
  auto replica = Database::open((dir_ / "replica").string(), params());
  replica->apply({primary->upsert(0, row(0))});
  primary->upsert(1, row(1));                 // the replica misses this one
  const auto third = primary->upsert(2, row(2));
  EXPECT_THROW(replica->apply({third}), std::runtime_error);

  // Catch up from the primary's log.
  const auto missing = primary->records_since(replica->last_lsn());
  ASSERT_TRUE(missing.has_value());
  EXPECT_EQ(missing->size(), 2u);
  replica->apply(*missing);
  EXPECT_EQ(replica->last_lsn(), 3u);
  EXPECT_TRUE(replica->index().contains(1));
}

TEST_F(DatabaseTest, SnapshotBootstrapsAReplicaPastACheckpoint) {
  auto primary = Database::open((dir_ / "primary").string(), params());
  for (std::size_t i = 0; i < 300; ++i) primary->upsert(i, row(i));
  primary->checkpoint();
  primary->upsert(300, row(300));
  EXPECT_FALSE(primary->records_since(10).has_value());  // folded into the snapshot
  EXPECT_EQ(primary->records_since(300)->size(), 1u);

  auto replica = Database::open((dir_ / "replica").string(), params());
  replica->upsert(9999, row(1999));  // stale state that must be discarded
  const auto [lsn, bytes] = primary->export_snapshot();
  replica->import_snapshot(lsn, bytes);
  EXPECT_EQ(replica->last_lsn(), 301u);
  EXPECT_FALSE(replica->index().contains(9999));
  replica->apply({primary->upsert(301, row(301))});  // and then follows the live stream

  replica.reset();  // the imported snapshot is this node's durable checkpoint now
  replica = Database::open((dir_ / "replica").string(), params());
  EXPECT_EQ(replica->size(), 302u);
  EXPECT_EQ(replica->last_lsn(), 302u);
}

TEST(Crc32c, KnownVector) {
  // RFC 3720 test vector: CRC-32C of "123456789".
  EXPECT_EQ(crc32c("123456789", 9), 0xE3069283u);
}

#if !defined(__EMSCRIPTEN__)
// A real crash: a child process writes with fsync-per-write and reports each acknowledged label
// through a pipe; the parent SIGKILLs it at a random moment. Every acknowledged write must be
// there after recovery, and the database must still open cleanly.
TEST_F(DatabaseTest, SurvivesKillNineMidWrite) {
  for (int trial = 0; trial < 3; ++trial) {
    fs::remove_all(dir_);
    int fds[2];
    ASSERT_EQ(pipe(fds), 0);
    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      close(fds[0]);
      DatabaseOptions options;
      options.checkpoint_bytes = 30'000;  // also crash in and around checkpoints
      auto db = Database::open(dir_.string(), params(), options);
      for (std::uint64_t i = 0;; i = (i + 1) % 2000) {
        db->upsert(i, data_.base.row(i));
        if (write(fds[1], &i, sizeof(i)) != sizeof(i)) _exit(1);
      }
    }
    close(fds[1]);
    std::mt19937 rng(static_cast<unsigned>(trial) + 1);
    std::set<std::uint64_t> acknowledged;
    std::uint64_t label;
    const std::size_t stop_after = 200 + rng() % 800;
    while (acknowledged.size() < stop_after && read(fds[0], &label, sizeof(label)) == sizeof(label)) {
      acknowledged.insert(label);
    }
    kill(child, SIGKILL);
    waitpid(child, nullptr, 0);
    close(fds[0]);

    auto db = open();
    for (std::uint64_t l : acknowledged) EXPECT_TRUE(db->index().contains(l)) << "lost acknowledged write " << l;
  }
}
#endif

}  // namespace
}  // namespace epzero
