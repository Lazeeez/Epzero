# Read consistency in the Epzero cluster

What a search sees after a write, what can go wrong when nodes crash or the network misbehaves,
and how that is tested. Short version: **`strong` reads are linearizable** under crashes,
partitions and lossy links (checked with Porcupine on every push); **`eventual` reads** (the
default) are faster at the tail and can briefly miss writes around a fault.

## The setup

Each shard has two copies on different nodes: a **primary** and a **replica**. Writes go to the
primary, fenced by the shard's **epoch** (a node refuses writes from an older epoch). The primary
writes to its log, then forwards the write to its replica if the replica is **in sync**, and only
then acknowledges. The router reports whether that happened:

| `WriteResult.replicated` | Meaning |
|---|---|
| `true` | Both copies have the write. It survives losing either node. |
| `false` | Only the primary has it (the replica was down or catching up). It is lost if the primary dies before the replica catches up. |

Writes stay available with one copy down, and the client is told so. A client that needs
durability across a single failure retries, or alerts, on `replicated: false`.

## Consistency levels

Every search picks one (`"consistency"` in the HTTP body, `Router.SearchWith` in Go). The router's
default is `Config.Consistency`, which is `eventual` unless set.

| Level | Reads | Guarantee | Cost |
|---|---|---|---|
| `strong` | Each shard's primary only | Linearizable: a search sees every write acknowledged before it began, and never goes back in time | A shard is unreadable from when its primary dies until the router fails it over (three missed health checks); no hedging to the replica |
| `eventual` | Primary or replica, hedged when the first is slow | A replica answers only while both copies report it in sync, so in the steady state it sees every replicated write | Around a fault, a search can miss writes the primary acknowledged alone, or lag by up to one health-check interval |

## How it is tested

`cluster/sim` runs the real router (package `cluster/router`, unchanged) against simulated data
nodes whose replication mirrors `epzero-node`: forwarding, catch-up by `Fetch`, snapshots for a
returning ex-primary, and epoch fencing. A simulated network can delay, drop or cut any link. Under
`testing/synctest` the whole cluster runs on a virtual clock, so a minute of traffic and faults
takes well under a second.

Four clients read and write 24 keys through the router while a nemesis injects one kind of fault:

| Scenario | Fault |
|---|---|
| no faults | none |
| crashes | a random node dies for 2–5 s, then restarts with its disk intact |
| partitions | a node is cut off from everyone, or only from its peers (the router still reaches it) |
| lossy links | 30% loss between two nodes, plus 5–65 ms of jitter between the router and one of them |

[Porcupine](https://github.com/anishathalye/porcupine) then checks the history for
linearizability. A write that failed, or that was acknowledged with `replicated: false`, counts as
**ambiguous**: it may take effect at any point after it began, or never. Everything else must fit
one total order consistent with real time.

Results (3 seeds each, about 68,000 operations, `go test -race ./sim`):

| Scenario | `strong` | `eventual` |
|---|---|---|
| no faults | linearizable | linearizable |
| crashes | linearizable | linearizable |
| partitions | linearizable | not linearizable (expected) |
| lossy links | linearizable | not linearizable (expected) |

The site's [cluster page](https://epzero.dev/cluster) runs the same simulation in the
browser, with a button that runs Porcupine over the live history.

## Bugs it found

Both are fixed. Before the first fix, even primary-only reads failed the check under lossy links.

1. **A retried write could be applied twice.** When a write timed out, the router sent it again.
   But a timeout does not mean the write failed: on a lossy link the primary often applied it and
   only the reply was lost. The second copy then landed after newer writes, so a key went
   4 → 2 → 4 with nobody writing 4 the second time. Now the router retries only writes the node
   certainly did not apply (`router.ErrRejected`: a stale epoch, not the primary, connection
   refused). Any other error goes back to the client as "outcome unknown".
2. **A replica could serve stale reads indefinitely.** If a forward to the replica was lost, the
   primary marked the replica out of sync and stopped forwarding. But the replica never heard
   about it: it kept reporting itself in sync, so it neither caught up nor stopped answering reads.
   The router now (a) reads a replica only when the primary also reports it in sync, and
   (b) re-assigns such a replica, which makes it fetch what it missed.

## Not covered

- **Durability of `replicated: false` writes.** These are lost if the primary dies before its
  replica catches up. That is by design (available, and reported to the client), not a bug.
  Refusing writes without a live replica, like Kafka's `min.insync.replicas = 2`, would be the
  alternative.
- **Many routers.** Several routers can share one etcd cluster (`-etcd`). They all serve searches
  and writes from the shard map, which etcd pushes to them; an election picks one leader to
  health-check the nodes and fail shards over. Its changes are compare-and-swap writes fenced by
  its election key, so a leader cut off from etcd can't save once its lease (`-lease`, default
  5 s) expires, even before it notices. Tests run two routers against an embedded etcd: the
  follower serves strong reads through a failover, takes over when the leader stops, and fails
  over the next node itself; a deposed leader's save is refused. The linearizability runs above
  use a single router.
- **Simulation fidelity.** The simulated nodes reproduce `epzero-node`'s replication protocol, not
  its code. The real nodes are covered by the three-node failover test in CI (`-tags integration`).
