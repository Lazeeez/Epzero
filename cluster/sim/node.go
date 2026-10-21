package sim

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/Lazeeez/Epzero/cluster/router"
)

// Node is a simulated epzero-node. Its shard copies replicate the way the real node's do
// (engine/server/node_main.cpp): the primary forwards each write to an in-sync replica before
// acknowledging; a replica that falls behind pulls what it missed (Fetch) and is in sync again;
// an ex-primary rejoining as a replica starts over from a snapshot. Assignments and data survive
// a crash, as they do on the real node's disk; roles wait for the router to confirm them.
type Node struct {
	Name    string
	cluster *Cluster

	mu         sync.Mutex
	up         bool
	generation int // bumped by each crash, to stop the dead process's background work
	started    time.Time
	shards     map[int]*shard
	// Extra time every search takes on this node (a slow or overloaded machine).
	slow time.Duration
}

type record struct {
	lsn    uint64
	remove bool
	label  uint64
	vector []float32
}

// lock is a mutex made of a channel. A shard's lock is held across network calls (writes queue
// behind the forward to the replica, as on the real node), and testing/synctest only advances its
// virtual clock when every goroutine is durably blocked: waiting on a channel counts, waiting on a
// sync.Mutex does not.
type lock chan struct{}

func newLock() lock    { return make(lock, 1) }
func (l lock) Lock()   { l <- struct{}{} }
func (l lock) Unlock() { <-l }
func (l lock) TryLock() bool {
	select {
	case l <- struct{}{}:
		return true
	default:
		return false
	}
}

type shard struct {
	id int

	mu            lock
	role          router.Role
	epoch         uint64
	peer          string
	confirmed     bool // the router assigned the role since the last restart
	inSync        bool // primary: forwarding to its replica; replica: caught up
	catchingUp    bool
	forceSnapshot bool

	lsn  uint64
	base uint64   // log holds lsn base+1 .. lsn (older ones are in the snapshot)
	log  []record // since base
	data map[uint64][]float32

	lastMu sync.Mutex
	last   router.ShardStatus // reported while a write holds the lock
}

var (
	errNotPrimary  = fmt.Errorf("%w: not the primary", router.ErrRejected)
	errStaleEpoch  = fmt.Errorf("%w: stale epoch", router.ErrRejected)
	errNotReplica  = errors.New("failed precondition: not a replica in this epoch")
	errNotPeer     = errors.New("failed precondition: not this replica's primary in this epoch")
	errNoShard     = fmt.Errorf("%w: shard not hosted here", router.ErrRejected)
	errGap         = errors.New("out of range: gap in the log")
	errWrongDim    = errors.New("invalid argument: wrong dimension")
	errNotAssigned = errors.New("failed precondition: not assigned")
)

func newNode(name string, c *Cluster) *Node {
	return &Node{Name: name, cluster: c, up: true, started: time.Now(), shards: map[int]*shard{}}
}

func (n *Node) alive() bool {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.up
}

func (n *Node) gen() int {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.generation
}

// Shards reports the node's shard copies as its health check would.
func (n *Node) Shards() []router.ShardStatus { return n.status().Shards }

// Up reports whether the node's process is running.
func (n *Node) Up() bool { return n.alive() }

// SetSlow makes every search on this node take d longer.
func (n *Node) SetSlow(d time.Duration) {
	n.mu.Lock()
	n.slow = d
	n.mu.Unlock()
}

// Kill stops the node at once, as if its machine lost power. Data and assignments stay on "disk".
func (n *Node) Kill() {
	n.mu.Lock()
	n.up = false
	n.generation++
	n.mu.Unlock()
}

// Restart starts the node again: same disk, new process (roles unconfirmed, nothing in sync).
func (n *Node) Restart() {
	// Lock order is always node, then shard, never the other way while holding a shard (see
	// catchUp and replicate), so the shards are reset without holding the node's lock.
	n.mu.Lock()
	if n.up {
		n.mu.Unlock()
		return
	}
	shards := make([]*shard, 0, len(n.shards))
	for _, s := range n.shards {
		shards = append(shards, s)
	}
	n.mu.Unlock()
	for _, s := range shards {
		s.mu.Lock()
		s.confirmed, s.inSync, s.catchingUp = false, false, false
		s.mu.Unlock()
	}
	n.mu.Lock()
	n.up = true
	n.started = time.Now()
	n.mu.Unlock()
}

func (n *Node) shard(id int, create bool) *shard {
	n.mu.Lock()
	defer n.mu.Unlock()
	s := n.shards[id]
	if s == nil && create {
		s = &shard{id: id, mu: newLock(), data: map[uint64][]float32{}}
		n.shards[id] = s
	}
	return s
}

// ---- The node's RPC handlers (called through the network by client) ----

func (n *Node) status() *router.NodeStatus {
	n.mu.Lock()
	st := &router.NodeStatus{StartedUnixMs: uint64(n.started.UnixMilli()), SIMD: "simulated"}
	ids := make([]int, 0, len(n.shards))
	for id := range n.shards {
		ids = append(ids, id)
	}
	n.mu.Unlock()
	sort.Ints(ids)
	for _, id := range ids {
		s := n.shard(id, false)
		// Never block a health check on a write (the real node try-locks too).
		if s.mu.TryLock() {
			role := router.RoleNone
			if s.confirmed {
				role = s.role
			}
			now := router.ShardStatus{
				Shard: id, Role: role, Epoch: s.epoch, LSN: s.lsn, Vectors: uint64(len(s.data)), InSync: s.inSync && !s.catchingUp,
			}
			s.mu.Unlock()
			s.lastMu.Lock()
			s.last = now
			s.lastMu.Unlock()
		}
		s.lastMu.Lock()
		st.Shards = append(st.Shards, s.last)
		s.lastMu.Unlock()
	}
	return st
}

func (n *Node) assign(a router.Assignment) error {
	s := n.shard(a.Shard, true)
	s.mu.Lock()
	if a.Epoch < s.epoch {
		s.mu.Unlock()
		return errStaleEpoch
	}
	previousRole, previousEpoch := s.role, s.epoch
	s.role, s.epoch, s.peer = a.Role, a.Epoch, a.Peer
	s.confirmed = true
	s.inSync = false
	catchUp := false
	if a.Role == router.RoleReplica {
		// An ex-primary may hold acknowledged-only-locally writes the new primary never saw.
		s.forceSnapshot = previousRole == router.RolePrimary && previousEpoch < a.Epoch
		catchUp = true
	}
	s.mu.Unlock()
	if catchUp {
		go n.catchUp(a.Shard, n.gen())
	}
	return nil
}

func (n *Node) checkPrimary(s *shard, epoch uint64) error {
	if !s.confirmed || s.role != router.RolePrimary {
		return errNotPrimary
	}
	if epoch != s.epoch {
		return errStaleEpoch
	}
	return nil
}

func (n *Node) write(ctx context.Context, shardID int, epoch uint64, records []record) (bool, error) {
	s := n.shard(shardID, false)
	if s == nil {
		return false, errNoShard
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := n.checkPrimary(s, epoch); err != nil {
		return false, err
	}
	for i := range records {
		s.lsn++
		records[i].lsn = s.lsn
		s.applyLocked(records[i])
		s.log = append(s.log, records[i])
	}
	return n.forwardLocked(ctx, s, records), nil
}

func (s *shard) applyLocked(r record) {
	if r.remove {
		delete(s.data, r.label)
	} else {
		s.data[r.label] = r.vector
	}
}

// forwardLocked sends records to the replica if it is in sync, holding the shard's lock as the
// real node does (writes queue behind it). Reports whether the replica applied them.
func (n *Node) forwardLocked(ctx context.Context, s *shard, records []record) bool {
	if s.peer == "" || !s.inSync {
		return false
	}
	peer := n.cluster.Nodes[s.peer]
	if peer == nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 3*time.Second)
	defer cancel()
	epoch := s.epoch
	err := n.cluster.Net.call(ctx, n.Name, s.peer, peer, func() error { return peer.replicate(s.id, epoch, records) })
	if err != nil {
		s.inSync = false // degraded until the replica fetches again
		return false
	}
	return true
}

func (n *Node) replicate(shardID int, epoch uint64, records []record) error {
	s := n.shard(shardID, false)
	if s == nil {
		return errNoShard
	}
	generation := n.gen() // before taking the shard's lock (lock order: node, then shard)
	s.mu.Lock()
	// A forward can race with our own catch-up; wait for it rather than report a gap.
	for deadline := time.Now().Add(20 * time.Second); s.catchingUp && time.Now().Before(deadline); {
		s.mu.Unlock()
		time.Sleep(5 * time.Millisecond)
		s.mu.Lock()
	}
	defer s.mu.Unlock()
	if s.role != router.RoleReplica || epoch != s.epoch {
		return errNotReplica
	}
	if err := s.appendLocked(records); err != nil {
		s.inSync = false
		go n.catchUp(shardID, generation)
		return err // the primary stops forwarding until we fetch
	}
	return nil
}

// appendLocked applies a primary's records, which must continue the log exactly (as
// Database::apply requires): a gap or a repeat is an error.
func (s *shard) appendLocked(records []record) error {
	for _, r := range records {
		if r.lsn != s.lsn+1 {
			return errGap
		}
		s.lsn = r.lsn
		s.applyLocked(r)
		s.log = append(s.log, r)
	}
	return nil
}

type fetchReply struct {
	records  []record
	snapshot map[uint64][]float32 // set when the replica must start over
	snapLSN  uint64
}

func (n *Node) fetch(shardID int, epoch uint64, after uint64, replica string, force bool) (*fetchReply, error) {
	s := n.shard(shardID, false)
	if s == nil {
		return nil, errNoShard
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.confirmed || s.role != router.RolePrimary || epoch != s.epoch || replica != s.peer {
		return nil, errNotPeer
	}
	reply := &fetchReply{}
	if !force && after >= s.base && after <= s.lsn {
		reply.records = append([]record(nil), s.log[after-s.base:]...)
	} else {
		reply.snapshot = make(map[uint64][]float32, len(s.data))
		for k, v := range s.data {
			reply.snapshot[k] = v
		}
		reply.snapLSN = s.lsn
	}
	// From here on every write is forwarded; it queues behind the replica applying this reply.
	s.inSync = true
	return reply, nil
}

// catchUp is the replica's pull loop: fetch what was missed until caught up or reassigned.
func (n *Node) catchUp(shardID int, generation int) {
	s := n.shard(shardID, false)
	for attempt := 0; attempt < 600; attempt++ {
		if n.gen() != generation || !n.alive() || n.cluster.stopped() {
			return
		}
		s.mu.Lock()
		if s.role != router.RoleReplica || s.peer == "" || s.inSync {
			s.mu.Unlock()
			return
		}
		s.catchingUp = true
		epoch, after, force, peerName := s.epoch, s.lsn, s.forceSnapshot, s.peer
		s.mu.Unlock()

		peer := n.cluster.Nodes[peerName]
		var reply *fetchReply
		err := errNotAssigned
		if peer != nil {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			err = n.cluster.Net.call(ctx, n.Name, peerName, peer, func() error {
				var e error
				reply, e = peer.fetch(shardID, epoch, after, n.Name, force)
				return e
			})
			cancel()
		}
		current := n.gen() == generation // read before taking the shard's lock (lock order)
		s.mu.Lock()
		if err == nil && current {
			if reply.snapshot != nil {
				s.data, s.lsn, s.base, s.log = reply.snapshot, reply.snapLSN, reply.snapLSN, nil
			}
			if e := s.appendLocked(reply.records); e != nil {
				s.forceSnapshot = true // start over from a snapshot next time
			} else {
				s.forceSnapshot = false
				s.inSync = true
			}
		}
		s.catchingUp = false
		done := s.inSync
		s.mu.Unlock()
		if done {
			return
		}
		time.Sleep(500 * time.Millisecond)
	}
}

func (n *Node) search(shardID int, query []float32, k int) (*router.ShardReply, error) {
	s := n.shard(shardID, false)
	if s == nil {
		return nil, errNoShard
	}
	start := time.Now()
	s.mu.Lock()
	hits := make([]router.NodeHit, 0, len(s.data))
	for label, v := range s.data {
		if len(v) != len(query) {
			s.mu.Unlock()
			return nil, errWrongDim
		}
		var d float32
		for i, x := range v {
			t := x - query[i]
			d += t * t
		}
		hits = append(hits, router.NodeHit{Label: label, Distance: d})
	}
	scored := uint64(len(hits))
	s.mu.Unlock()
	sort.Slice(hits, func(a, b int) bool {
		if hits[a].Distance != hits[b].Distance {
			return hits[a].Distance < hits[b].Distance
		}
		return hits[a].Label < hits[b].Label
	})
	if len(hits) > k {
		hits = hits[:k]
	}
	n.mu.Lock()
	slow := n.slow
	n.mu.Unlock()
	if slow > 0 {
		time.Sleep(slow)
	}
	return &router.ShardReply{Hits: hits, Micros: uint64(time.Since(start).Microseconds()), DistanceComputations: scored}, nil
}

// String describes the node's shard copies (for logs and test failures).
func (n *Node) String() string {
	st := n.status()
	out := n.Name
	for _, s := range st.Shards {
		out += fmt.Sprintf(" [s%d %s e%d lsn%d n%d sync=%v]", s.Shard, s.Role, s.Epoch, s.LSN, s.Vectors, s.InSync)
	}
	return out
}
