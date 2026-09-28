// oryx-node: a data node of the Oryx cluster. Hosts shard copies (each a durable oryx::Database)
// and serves the gRPC protocol in proto/oryx/v1/oryx.proto.
//
// Replication is primary/backup per shard. The primary logs and applies a write, forwards the
// same log records (same LSNs) to its replica and waits for it before acknowledging, so an
// acknowledged write normally lives on two machines. If the replica is down the primary keeps
// accepting writes ("replicated": false in the reply) and the replica catches up when it returns,
// from the primary's log or, if a checkpoint folded that away, from a snapshot. Epochs fence off
// stale primaries after a failover.

#include <grpcpp/grpcpp.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>

#include "oryx/database.h"
#include "oryx/v1/oryx.grpc.pb.h"

namespace fs = std::filesystem;
namespace pb = oryx::v1;
using grpc::ServerContext;
using grpc::StatusCode;
using Clock = std::chrono::steady_clock;

namespace {

struct Config {
  std::string listen = "0.0.0.0:7000";
  std::string advertise;  // address peers use to reach this node (default: listen)
  std::string name = "node";
  std::string data = "/var/lib/oryx/data";
  std::string token;      // required by Crash
  oryx::HnswParams params{.dim = 512, .metric = oryx::Metric::Cosine, .max_elements = 200'000, .M = 16,
                          .ef_construction = 128};
  oryx::SyncMode sync = oryx::SyncMode::Always;
};

pb::Record to_proto(const oryx::LogRecord& r) {
  pb::Record out;
  out.set_type(static_cast<std::uint32_t>(r.type));
  out.set_lsn(r.lsn);
  out.set_label(r.label);
  out.mutable_vector()->Assign(r.vector.begin(), r.vector.end());
  return out;
}

oryx::LogRecord from_proto(const pb::Record& r) {
  return {static_cast<oryx::LogRecord::Type>(r.type()), r.lsn(), r.label(),
          std::vector<float>(r.vector().begin(), r.vector().end())};
}

// Role, epoch and peer survive restarts so a node comes back knowing what it was; it still waits
// for the router to confirm (Assign) before taking writes as a primary.
struct Assignment {
  pb::Role role = pb::ROLE_UNSPECIFIED;
  std::uint64_t epoch = 0;
  std::string peer;
};

class Shard {
 public:
  Shard(std::uint32_t id, const Config& config) : id_(id), dir_(fs::path(config.data) / ("shard-" + std::to_string(id))) {
    oryx::DatabaseOptions options;
    options.sync = config.sync;
    db_ = oryx::Database::open(dir_.string(), config.params, options);
    std::ifstream in(dir_ / "ASSIGNMENT");
    int role = 0;
    if (in >> role >> assignment_.epoch) {
      assignment_.role = static_cast<pb::Role>(role);
      in >> assignment_.peer;
    }
  }

  std::uint32_t id() const { return id_; }
  oryx::Database& db() { return *db_; }

  std::mutex mu;                      // writes, role changes and forwarding, in log order
  std::condition_variable caught_up;  // signalled when a catch-up finishes
  bool catching_up = false;
  bool confirmed = false;             // the router has (re)assigned us since start
  bool in_sync = false;               // primary: forwarding to replica; replica: caught up
  std::unique_ptr<pb::Node::Stub> peer_stub;

  const Assignment& assignment() const { return assignment_; }
  void set_assignment(const Assignment& a) {
    assignment_ = a;
    const fs::path temp = dir_ / "ASSIGNMENT.tmp";
    {
      std::ofstream out(temp, std::ios::trunc);
      out << static_cast<int>(a.role) << ' ' << a.epoch << ' ' << (a.peer.empty() ? "-" : a.peer) << '\n';
    }
    fs::rename(temp, dir_ / "ASSIGNMENT");
  }

 private:
  std::uint32_t id_;
  fs::path dir_;
  std::unique_ptr<oryx::Database> db_;
  Assignment assignment_;
};

class NodeService final : public pb::Node::Service {
 public:
  explicit NodeService(Config config) : config_(std::move(config)), started_(now_ms()) {
    // Re-open every shard this node held before a restart.
    if (fs::exists(config_.data)) {
      for (const auto& entry : fs::directory_iterator(config_.data)) {
        const std::string name = entry.path().filename().string();
        if (name.starts_with("shard-")) shard(static_cast<std::uint32_t>(std::stoul(name.substr(6))), true);
      }
    }
  }

  grpc::Status Upsert(ServerContext*, const pb::UpsertRequest* req, pb::WriteReply* reply) override {
    Shard* s = shard(req->shard(), false);
    if (!s) return {StatusCode::NOT_FOUND, "shard not hosted here"};
    const std::size_t dim = config_.params.dim;
    const std::size_t n = static_cast<std::size_t>(req->labels_size());
    if (n == 0 || static_cast<std::size_t>(req->vectors_size()) != n * dim) {
      return {StatusCode::INVALID_ARGUMENT, "expected labels.size() * dim vector values"};
    }
    std::unique_lock lock(s->mu);
    if (grpc::Status st = check_primary(*s, req->epoch()); !st.ok()) return st;
    try {
      std::vector<oryx::LogRecord> records;
      if (n == 1) {
        records.push_back(s->db().upsert(req->labels(0), req->vectors().data()));
      } else {
        records = s->db().upsert_batch(req->labels().data(), req->vectors().data(), n);
      }
      reply->set_lsn(s->db().last_lsn());
      reply->set_replicated(forward(*s, records));
    } catch (const std::exception& e) {
      return {StatusCode::RESOURCE_EXHAUSTED, e.what()};
    }
    return grpc::Status::OK;
  }

  grpc::Status Remove(ServerContext*, const pb::RemoveRequest* req, pb::WriteReply* reply) override {
    Shard* s = shard(req->shard(), false);
    if (!s) return {StatusCode::NOT_FOUND, "shard not hosted here"};
    std::unique_lock lock(s->mu);
    if (grpc::Status st = check_primary(*s, req->epoch()); !st.ok()) return st;
    const auto record = s->db().remove(req->label());
    reply->set_lsn(s->db().last_lsn());
    reply->set_replicated(record ? forward(*s, {*record}) : true);
    return grpc::Status::OK;
  }

  grpc::Status Search(ServerContext*, const pb::SearchRequest* req, pb::SearchReply* reply) override {
    Shard* s = shard(req->shard(), false);
    if (!s) return {StatusCode::NOT_FOUND, "shard not hosted here"};
    if (static_cast<std::size_t>(req->query_size()) != config_.params.dim) {
      return {StatusCode::INVALID_ARGUMENT, "query has the wrong dimension"};
    }
    oryx::SearchStats stats;
    const auto start = Clock::now();
    const auto hits = s->db().search(req->query().data(), std::max(1u, req->k()),
                                     {.ef = std::max<std::size_t>(req->ef(), 16), .stats = &stats});
    reply->set_micros(static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count()));
    reply->set_distance_computations(stats.distance_computations);
    for (const auto& h : hits) {
      auto* hit = reply->add_hits();
      hit->set_label(h.label);
      hit->set_distance(h.distance);
    }
    return grpc::Status::OK;
  }

  grpc::Status Replicate(ServerContext*, const pb::ReplicateRequest* req, pb::ReplicateReply* reply) override {
    Shard* s = shard(req->shard(), false);
    if (!s) return {StatusCode::NOT_FOUND, "shard not hosted here"};
    std::unique_lock lock(s->mu);
    // A forward can race with our own catch-up; wait for it rather than report a gap.
    s->caught_up.wait_for(lock, std::chrono::seconds(20), [&] { return !s->catching_up; });
    const Assignment& a = s->assignment();
    if (a.role != pb::ROLE_REPLICA || req->epoch() != a.epoch) {
      return {StatusCode::FAILED_PRECONDITION, "not a replica in this epoch"};
    }
    std::vector<oryx::LogRecord> records;
    records.reserve(static_cast<std::size_t>(req->records_size()));
    for (const auto& r : req->records()) records.push_back(from_proto(r));
    try {
      s->db().apply(records);
    } catch (const std::exception& e) {
      s->in_sync = false;
      catch_up_async(s->id());
      return {StatusCode::OUT_OF_RANGE, e.what()};  // gap: the primary stops forwarding until we fetch
    }
    reply->set_lsn(s->db().last_lsn());
    return grpc::Status::OK;
  }

  grpc::Status Fetch(ServerContext*, const pb::FetchRequest* req, pb::FetchReply* reply) override {
    Shard* s = shard(req->shard(), false);
    if (!s) return {StatusCode::NOT_FOUND, "shard not hosted here"};
    std::unique_lock lock(s->mu);
    const Assignment& a = s->assignment();
    if (a.role != pb::ROLE_PRIMARY || req->epoch() != a.epoch || req->replica_address() != a.peer) {
      return {StatusCode::FAILED_PRECONDITION, "not this replica's primary in this epoch"};
    }
    std::optional<std::vector<oryx::LogRecord>> records;
    if (!req->force_snapshot() && req->after_lsn() <= s->db().last_lsn()) records = s->db().records_since(req->after_lsn());
    if (records) {
      for (const auto& r : *records) *reply->add_records() = to_proto(r);
    } else {
      auto [lsn, bytes] = s->db().export_snapshot();
      reply->set_has_snapshot(true);
      reply->set_snapshot_lsn(lsn);
      reply->set_snapshot(std::move(bytes));
    }
    // From here on every write is forwarded; it queues behind the replica applying this reply.
    s->in_sync = true;
    return grpc::Status::OK;
  }

  grpc::Status Assign(ServerContext*, const pb::AssignRequest* req, pb::AssignReply*) override {
    Shard* s = shard(req->shard(), true);
    bool start_catch_up = false;
    {
      std::unique_lock lock(s->mu);
      const Assignment previous = s->assignment();
      if (req->epoch() < previous.epoch) return {StatusCode::FAILED_PRECONDITION, "stale epoch"};
      s->set_assignment({req->role(), req->epoch(), req->peer()});
      s->confirmed = true;
      s->in_sync = false;
      s->peer_stub = req->peer().empty() ? nullptr : stub(req->peer());
      if (req->role() == pb::ROLE_REPLICA) {
        // An ex-primary may hold acknowledged-only-locally writes the new primary never saw.
        force_snapshot_[s->id()] = previous.role == pb::ROLE_PRIMARY && previous.epoch < req->epoch();
        start_catch_up = true;
      }
    }
    if (start_catch_up) catch_up_async(s->id());
    return grpc::Status::OK;
  }

  grpc::Status Status(ServerContext*, const pb::StatusRequest*, pb::StatusReply* reply) override {
    reply->set_node(config_.name);
    reply->set_started_unix_ms(started_);
    reply->set_simd(std::string(oryx::to_string(oryx::active_simd_level())));
    std::shared_lock lock(shards_mu_);
    for (const auto& [id, s] : shards_) {
      std::unique_lock shard_lock(s->mu, std::try_to_lock);  // never block health checks on a write
      auto* out = reply->add_shards();
      out->set_shard(id);
      const Assignment a = s->assignment();
      out->set_role(s->confirmed ? a.role : pb::ROLE_UNSPECIFIED);
      out->set_epoch(a.epoch);
      out->set_peer(a.peer);
      out->set_lsn(s->db().last_lsn());
      out->set_vectors(s->db().size());
      out->set_in_sync(s->in_sync && !s->catching_up);
    }
    return grpc::Status::OK;
  }

  grpc::Status Crash(ServerContext*, const pb::CrashRequest* req, pb::CrashReply*) override {
    if (config_.token.empty() || req->token() != config_.token) return {StatusCode::PERMISSION_DENIED, "bad token"};
    std::cerr << "chaos: crashing on request\n";
    std::thread([] {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));  // let the reply leave
      std::_Exit(137);                                              // no destructors, no flush: a crash
    }).detach();
    return grpc::Status::OK;
  }

 private:
  static std::uint64_t now_ms() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
  }

  std::unique_ptr<pb::Node::Stub> stub(const std::string& address) {
    return pb::Node::NewStub(grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));
  }

  Shard* shard(std::uint32_t id, bool create) {
    {
      std::shared_lock lock(shards_mu_);
      if (auto it = shards_.find(id); it != shards_.end()) return it->second.get();
    }
    if (!create) return nullptr;
    std::unique_lock lock(shards_mu_);
    auto& slot = shards_[id];
    if (!slot) slot = std::make_unique<Shard>(id, config_);
    return slot.get();
  }

  grpc::Status check_primary(Shard& s, std::uint64_t epoch) {
    const Assignment& a = s.assignment();
    if (!s.confirmed || a.role != pb::ROLE_PRIMARY) return {StatusCode::FAILED_PRECONDITION, "not the primary"};
    if (epoch != a.epoch) return {StatusCode::FAILED_PRECONDITION, "stale epoch"};
    return grpc::Status::OK;
  }

  // Caller holds s.mu. Returns whether the replica applied the records.
  bool forward(Shard& s, const std::vector<oryx::LogRecord>& records) {
    if (!s.peer_stub || !s.in_sync) return false;
    pb::ReplicateRequest req;
    req.set_shard(s.id());
    req.set_epoch(s.assignment().epoch);
    for (const auto& r : records) *req.add_records() = to_proto(r);
    grpc::ClientContext ctx;
    ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
    pb::ReplicateReply reply;
    const auto status = s.peer_stub->Replicate(&ctx, req, &reply);
    if (!status.ok()) s.in_sync = false;  // degraded until the replica fetches again
    return status.ok();
  }

  // Replica: pull what we missed from the primary, retrying until caught up or reassigned.
  void catch_up_async(std::uint32_t id) {
    std::thread([this, id] {
      Shard* s = shard(id, false);
      for (int attempt = 0; attempt < 600; ++attempt) {
        pb::FetchRequest req;
        std::unique_ptr<pb::Node::Stub> primary;
        {
          std::unique_lock lock(s->mu);
          const Assignment a = s->assignment();
          if (a.role != pb::ROLE_REPLICA || a.peer.empty() || s->in_sync) return;
          s->catching_up = true;
          req.set_shard(id);
          req.set_epoch(a.epoch);
          req.set_after_lsn(s->db().last_lsn());
          req.set_replica_address(config_.advertise);
          req.set_force_snapshot(force_snapshot_[id]);
          primary = stub(a.peer);
        }
        // The RPC runs without our lock: the primary may be blocked forwarding to us.
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));
        pb::FetchReply reply;
        const auto status = primary->Fetch(&ctx, req, &reply);
        {
          std::unique_lock lock(s->mu);
          try {
            if (status.ok()) {
              if (reply.has_snapshot()) s->db().import_snapshot(reply.snapshot_lsn(), reply.snapshot());
              std::vector<oryx::LogRecord> records;
              for (const auto& r : reply.records()) records.push_back(from_proto(r));
              s->db().apply(records);
              force_snapshot_[id] = false;
              s->in_sync = true;
            }
          } catch (const std::exception& e) {
            std::cerr << "shard " << id << ": catch-up failed: " << e.what() << '\n';
            force_snapshot_[id] = true;  // start over from a snapshot next time
          }
          s->catching_up = false;
        }
        s->caught_up.notify_all();
        if (s->in_sync) {
          std::cerr << "shard " << id << ": caught up at lsn " << s->db().last_lsn() << '\n';
          return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }).detach();
  }

  Config config_;
  std::uint64_t started_;
  std::shared_mutex shards_mu_;
  std::map<std::uint32_t, std::unique_ptr<Shard>> shards_;
  std::map<std::uint32_t, bool> force_snapshot_;
};

Config parse(int argc, char** argv) {
  Config c;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
      return argv[++i];
    };
    if (arg == "--listen") c.listen = value();
    else if (arg == "--advertise") c.advertise = value();
    else if (arg == "--name") c.name = value();
    else if (arg == "--data") c.data = value();
    else if (arg == "--token-file") {
      std::ifstream in(value());
      std::getline(in, c.token);
    } else if (arg == "--dim") c.params.dim = std::stoul(value());
    else if (arg == "--max") c.params.max_elements = std::stoul(value());
    else if (arg == "--metric") {
      const std::string m = value();
      c.params.metric = m == "l2" ? oryx::Metric::L2 : m == "ip" ? oryx::Metric::InnerProduct : oryx::Metric::Cosine;
    } else if (arg == "--sync") {
      const std::string m = value();
      c.sync = m == "none" ? oryx::SyncMode::None : m == "batch" ? oryx::SyncMode::Batch : oryx::SyncMode::Always;
    } else {
      throw std::invalid_argument("unknown argument " + arg);
    }
  }
  if (c.advertise.empty()) c.advertise = c.listen;
  return c;
}

}  // namespace

int main(int argc, char** argv) try {
  const Config config = parse(argc, argv);
  fs::create_directories(config.data);
  NodeService service(config);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(config.listen, grpc::InsecureServerCredentials());
  builder.SetMaxReceiveMessageSize(256 << 20);  // snapshots and batch upserts
  builder.SetMaxSendMessageSize(256 << 20);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  if (!server) throw std::runtime_error("cannot listen on " + config.listen);
  std::cerr << config.name << " listening on " << config.listen << " (simd " << oryx::to_string(oryx::active_simd_level())
            << ")\n";
  server->Wait();
  return 0;
} catch (const std::exception& e) {
  std::cerr << "oryx-node: " << e.what() << '\n';
  return 1;
}
