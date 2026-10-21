// Package router is the Epzero cluster's coordinator and query front end.
//
// It owns the shard map (which node is primary and which is replica for each shard, and the
// shard's epoch), health-checks every data node, fails over a shard to its replica when the
// primary stops answering, and re-integrates nodes that come back. Queries fan out to one live
// copy of every shard (hedged to the other copy when slow) and the per-shard top-k lists are
// merged; writes are routed by label hash to each shard's primary.
package router

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// NodeConfig names a data node and where to reach it.
type NodeConfig struct {
	Name string `json:"name"`
	Addr string `json:"addr"`
}

// Config configures a Router.
type Config struct {
	Nodes          []NodeConfig
	Shards         int
	Dim            int
	Token          string        // required for writes and chaos; also passed to nodes' Crash
	StatePath      string        // with no Store: the file the shard map is kept in ("" = memory only)
	HedgeAfter     time.Duration // send a query to the other copy if the first is this slow
	HealthInterval time.Duration
	DownAfter      int // consecutive failed health checks before a node is declared down
	AllowedOrigins []string
	Logger         *slog.Logger
	// Dial connects to a node; nil means gRPC (DialGRPC). The simulator passes its in-memory nodes.
	Dial func(NodeConfig) (Node, error)
	// Consistency is the default for searches that don't choose one (default Eventual).
	Consistency Consistency
	// Store holds the shard map; nil means a FileStore at StatePath (a single router). With an
	// EtcdStore several routers share the map and elect one leader to change it.
	Store StateStore
	// ID names this router (in leader elections and events); default "router".
	ID string
}

// Consistency says which copies of a shard a search may read.
type Consistency string

const (
	// Strong reads only each shard's primary. Every search sees every acknowledged write, and the
	// read and write history is linearizable under crashes, partitions and lossy links (checked
	// with Porcupine in package sim). A shard is unreadable while its primary is down, until the
	// router fails it over.
	Strong Consistency = "strong"
	// Eventual also reads a shard's replica while both copies say it is in sync, hedging slow
	// answers across the two: lower tail latency, but a search can miss writes the primary
	// acknowledged without its replica (WriteResult.Replicated false) and, around a fault, lag
	// by up to a health-check interval.
	Eventual Consistency = "eventual"
)

// ShardMap is one shard's placement. Epoch increases on every failover.
type ShardMap struct {
	ID      int    `json:"id"`
	Epoch   uint64 `json:"epoch"`
	Primary string `json:"primary"`
	Replica string `json:"replica"`
}

type node struct {
	cfg    NodeConfig
	client Node

	mu       sync.Mutex
	up       bool
	fails    int
	started  uint64
	rtt      time.Duration
	status   *NodeStatus
	lastSeen time.Time
}

// Event is a notable cluster change, kept for the UI.
type Event struct {
	Time    time.Time `json:"time"`
	Kind    string    `json:"kind"` // node-down, node-up, failover, crash, degraded, ...
	Message string    `json:"message"`
}

// Router coordinates the cluster. Create with New, then Run.
type Router struct {
	cfg   Config
	log   *slog.Logger
	nodes map[string]*node
	names []string

	mu      sync.RWMutex // guards shards and version
	shards  []ShardMap
	version int64 // the store's version of shards

	store   StateStore
	leading atomic.Bool
	joined  sync.Once // the election and the watch, joined by Bootstrap or Run

	eventsMu sync.Mutex
	events   []Event

	metrics *metrics
	rr      uint64 // round-robin counter for choosing a copy to query first
}

// New connects to the nodes (lazily) and loads or creates the shard map.
func New(cfg Config) (*Router, error) {
	if len(cfg.Nodes) < 2 {
		return nil, fmt.Errorf("need at least 2 nodes for replication, got %d", len(cfg.Nodes))
	}
	if cfg.Shards <= 0 {
		cfg.Shards = len(cfg.Nodes)
	}
	if cfg.HedgeAfter == 0 {
		cfg.HedgeAfter = 40 * time.Millisecond
	}
	if cfg.HealthInterval == 0 {
		cfg.HealthInterval = 500 * time.Millisecond
	}
	if cfg.DownAfter == 0 {
		cfg.DownAfter = 3
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.Consistency == "" {
		cfg.Consistency = Eventual
	}
	if cfg.Consistency != Strong && cfg.Consistency != Eventual {
		return nil, fmt.Errorf("unknown consistency %q", cfg.Consistency)
	}
	if cfg.Dial == nil {
		cfg.Dial = defaultDial
	}
	if cfg.Store == nil {
		cfg.Store = NewFileStore(cfg.StatePath)
	}
	if cfg.ID == "" {
		cfg.ID = "router"
	}
	r := &Router{cfg: cfg, log: cfg.Logger, nodes: map[string]*node{}, metrics: newMetrics(), store: cfg.Store}
	for _, nc := range cfg.Nodes {
		client, err := cfg.Dial(nc)
		if err != nil {
			return nil, fmt.Errorf("node %s: %w", nc.Name, err)
		}
		r.nodes[nc.Name] = &node{cfg: nc, client: client}
		r.names = append(r.names, nc.Name)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := r.loadState(ctx); err != nil {
		return nil, err
	}
	return r, nil
}

// Initial placement: shard i's primary on node i, its replica on the next node, so every node
// holds two shard copies and losing any single node loses no shard.
func (r *Router) initialPlacement() []ShardMap {
	shards := make([]ShardMap, r.cfg.Shards)
	for i := range shards {
		n := len(r.names)
		shards[i] = ShardMap{ID: i, Epoch: 1, Primary: r.names[i%n], Replica: r.names[(i+1)%n]}
	}
	return shards
}

// loadState reads the shard map from the store, creating the initial placement if there is none.
// Several routers starting together may race to create it; the losers read the winner's.
func (r *Router) loadState(ctx context.Context) error {
	for attempt := 0; attempt < 5; attempt++ {
		shards, version, err := r.store.Load(ctx)
		if err != nil {
			return err
		}
		if shards != nil {
			if len(shards) != r.cfg.Shards {
				return fmt.Errorf("the stored shard map has %d shards, config says %d", len(shards), r.cfg.Shards)
			}
			r.shards, r.version = shards, version
			return nil
		}
		initial := r.initialPlacement()
		version, err = r.store.Save(ctx, initial, 0)
		if errors.Is(err, ErrConflict) {
			continue
		}
		if err != nil {
			return err
		}
		r.shards, r.version = initial, version
		return nil
	}
	return errors.New("could not create the shard map: it kept changing")
}

// applyMap installs a shard map from the store if it is newer than the one in use.
func (r *Router) applyMap(shards []ShardMap, version int64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if version > r.version && len(shards) == r.cfg.Shards {
		r.shards, r.version = shards, version
	}
}

// Leading reports whether this router is the leader, the one that changes the shard map.
func (r *Router) Leading() bool { return r.leading.Load() }

func (r *Router) event(kind, format string, args ...any) {
	e := Event{Time: time.Now(), Kind: kind, Message: fmt.Sprintf(format, args...)}
	r.log.Info(e.Message, "kind", kind)
	r.eventsMu.Lock()
	r.events = append(r.events, e)
	if len(r.events) > 100 {
		r.events = r.events[len(r.events)-100:]
	}
	r.eventsMu.Unlock()
}

// Run health-checks the nodes until ctx is cancelled.
func (r *Router) Run(ctx context.Context) {
	r.join(ctx)
	ticker := time.NewTicker(r.cfg.HealthInterval)
	defer ticker.Stop()
	for {
		r.checkAll(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (r *Router) checkAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, name := range r.names {
		wg.Add(1)
		go func(n *node) {
			defer wg.Done()
			r.check(ctx, n)
		}(r.nodes[name])
	}
	wg.Wait()
}

func (r *Router) check(ctx context.Context, n *node) {
	cctx, cancel := context.WithTimeout(ctx, 800*time.Millisecond)
	defer cancel()
	start := time.Now()
	status, err := n.client.Status(cctx)
	rtt := time.Since(start)

	n.mu.Lock()
	if err != nil {
		n.fails++
		wentDown := n.up && n.fails >= r.cfg.DownAfter
		if wentDown {
			n.up = false
		}
		n.mu.Unlock()
		if wentDown {
			r.metrics.nodeUp(n.cfg.Name, false)
			r.event("node-down", "%s stopped answering health checks", n.cfg.Name)
			r.failoverFrom(ctx, n.cfg.Name)
		}
		return
	}
	cameUp := !n.up
	restarted := n.started != 0 && status.StartedUnixMs != n.started
	n.up, n.fails, n.rtt, n.status, n.lastSeen = true, 0, rtt, status, time.Now()
	n.started = status.StartedUnixMs
	n.mu.Unlock()

	r.metrics.nodeUp(n.cfg.Name, true)
	r.repairReplicas(ctx, n.cfg.Name, status)
	if cameUp || restarted {
		if restarted && !cameUp {
			r.event("node-up", "%s restarted", n.cfg.Name)
		} else {
			r.event("node-up", "%s is up", n.cfg.Name)
		}
		r.reconcile(ctx, n.cfg.Name)
		return
	}
	// A node that lost its assignment (e.g. restarted between two health checks) reports no role.
	for _, s := range status.Shards {
		if s.Role == RoleNone {
			r.reconcile(ctx, n.cfg.Name)
			return
		}
	}
}

// repairReplicas re-assigns the replica of each shard whose primary (`name`) has stopped
// forwarding to it while the replica still reports itself in sync. That happens when a forward is
// lost: the primary marks the replica out of sync, but the replica never hears of it, so it would
// neither fetch what it missed nor stop answering reads. Assigning it again makes it catch up.
func (r *Router) repairReplicas(ctx context.Context, name string, status *NodeStatus) {
	if !r.leading.Load() {
		return
	}
	for _, st := range status.Shards {
		if st.Role != RolePrimary || st.InSync {
			continue
		}
		r.mu.RLock()
		var s ShardMap
		if st.Shard >= 0 && st.Shard < len(r.shards) {
			s = r.shards[st.Shard]
		}
		r.mu.RUnlock()
		if s.Primary != name || s.Epoch != st.Epoch || !r.isUp(s.Replica) || !r.reportsInSync(s.Replica, s.ID) {
			continue
		}
		r.event("repair", "shard %d: %s stopped forwarding to %s; re-assigning the replica to catch up", s.ID, name, s.Replica)
		go r.assign(ctx, s.Replica, s, RoleReplica, r.nodes[s.Primary].cfg.Addr)
	}
}

func (r *Router) isUp(name string) bool {
	n := r.nodes[name]
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.up
}

// failoverFrom promotes the replica of every shard whose primary is `name`. Only the leader
// changes the map, and only by compare-and-swap: a router that lost the lead without knowing it
// yet (say, partitioned from etcd) fails to save, and so never hands out a stale epoch.
func (r *Router) failoverFrom(ctx context.Context, name string) {
	if !r.leading.Load() {
		return
	}
	for attempt := 0; attempt < 3; attempt++ {
		r.mu.RLock()
		shards := append([]ShardMap(nil), r.shards...)
		version := r.version
		r.mu.RUnlock()
		var promoted []ShardMap
		for i := range shards {
			s := &shards[i]
			if s.Primary != name {
				continue
			}
			if !r.isUp(s.Replica) {
				r.event("unavailable", "shard %d has no live copy (%s and %s are down)", s.ID, s.Primary, s.Replica)
				continue
			}
			s.Epoch++
			s.Primary, s.Replica = s.Replica, s.Primary
			promoted = append(promoted, *s)
		}
		if len(promoted) == 0 {
			return
		}
		sctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		newVersion, err := r.store.Save(sctx, shards, version)
		cancel()
		if errors.Is(err, ErrConflict) {
			r.reloadMap(ctx) // someone else changed it: recompute from theirs
			continue
		}
		if err != nil {
			r.log.Error("saving the shard map", "err", err)
			return
		}
		r.applyMap(shards, newVersion)
		for _, s := range promoted {
			r.metrics.failovers.Add(1)
			r.event("failover", "shard %d: %s promoted to primary (epoch %d)", s.ID, s.Primary, s.Epoch)
			// Its replica (the failed node) gets its role back when it returns.
			r.assign(ctx, s.Primary, s, RolePrimary, r.nodes[s.Replica].cfg.Addr)
		}
		return
	}
}

func (r *Router) reloadMap(ctx context.Context) {
	lctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if shards, version, err := r.store.Load(lctx); err == nil && shards != nil {
		r.applyMap(shards, version)
	}
}

// reconcile (re)assigns every shard copy involving `name`: primaries first, then replicas, so a
// replica's catch-up finds its primary ready. A primary's replica is re-assigned too, because a
// restarted primary no longer forwards to it until the replica fetches again.
func (r *Router) reconcile(ctx context.Context, name string) {
	if !r.leading.Load() {
		return // the leader assigns roles; followers only route
	}
	r.mu.RLock()
	var involved []ShardMap
	for _, s := range r.shards {
		if s.Primary == name || s.Replica == name {
			involved = append(involved, s)
		}
	}
	r.mu.RUnlock()
	for _, s := range involved {
		if r.isUp(s.Primary) {
			r.assign(ctx, s.Primary, s, RolePrimary, r.nodes[s.Replica].cfg.Addr)
		}
	}
	for _, s := range involved {
		if r.isUp(s.Replica) {
			r.assign(ctx, s.Replica, s, RoleReplica, r.nodes[s.Primary].cfg.Addr)
		}
	}
}

func (r *Router) assign(ctx context.Context, name string, s ShardMap, role Role, peer string) {
	cctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	err := r.nodes[name].client.Assign(cctx, Assignment{Shard: s.ID, Epoch: s.Epoch, Role: role, Peer: peer})
	if err != nil {
		r.log.Warn("assign failed", "node", name, "shard", s.ID, "role", role.String(), "err", err)
	}
}

// Bootstrap joins the leader election and the shard-map watch, then checks every node once (the
// leader also assigns their roles). Run continues from there.
func (r *Router) Bootstrap(ctx context.Context) {
	r.join(ctx)
	r.checkAll(ctx) // marks reachable nodes up and, on the leader, reconciles them
}

func (r *Router) join(ctx context.Context) {
	r.joined.Do(func() {
		lead := r.store.Lead(ctx, r.cfg.ID)
		select {
		case leading, ok := <-lead: // a single router (FileStore) leads at once
			if ok && leading {
				r.leading.Store(true)
				r.event("leader", "%s is the leader", r.cfg.ID)
			}
		default:
		}
		go r.followLeadership(ctx, lead)
		go r.store.Watch(ctx, r.applyMap)
	})
}

// followLeadership tracks this router's leadership. On taking over, it re-reads the map and takes
// charge: re-assigns every live node's roles and fails over the shards of nodes already down.
func (r *Router) followLeadership(ctx context.Context, lead <-chan bool) {
	for leading := range lead {
		if leading == r.leading.Load() {
			continue
		}
		r.leading.Store(leading)
		if !leading {
			r.event("leader", "%s lost the lead", r.cfg.ID)
			continue
		}
		r.event("leader", "%s took over as leader", r.cfg.ID)
		r.reloadMap(ctx)
		for _, name := range r.names {
			if r.isUp(name) {
				r.reconcile(ctx, name)
			} else {
				r.failoverFrom(ctx, name)
			}
		}
	}
}

// ClusterView is the JSON shape of GET /v1/cluster.
type ClusterView struct {
	Router string      `json:"router"`
	Leader bool        `json:"leader"`
	Nodes  []NodeView  `json:"nodes"`
	Shards []ShardView `json:"shards"`
	Events []Event     `json:"events"`
}

type NodeView struct {
	Name     string  `json:"name"`
	Up       bool    `json:"up"`
	RTTMs    float64 `json:"rtt_ms"`
	UptimeS  float64 `json:"uptime_s"`
	SIMD     string  `json:"simd"`
	LastSeen string  `json:"last_seen"`
}

type ShardView struct {
	ShardMap
	Vectors       uint64 `json:"vectors"`
	PrimaryLSN    uint64 `json:"primary_lsn"`
	ReplicaLSN    uint64 `json:"replica_lsn"`
	ReplicaInSync bool   `json:"replica_in_sync"`
}

// View snapshots the cluster for the API and UI.
func (r *Router) View() ClusterView {
	v := ClusterView{Router: r.cfg.ID, Leader: r.leading.Load()}
	copies := map[string]map[int]ShardStatus{}
	for _, name := range r.names {
		n := r.nodes[name]
		n.mu.Lock()
		nv := NodeView{Name: name, Up: n.up, RTTMs: float64(n.rtt.Microseconds()) / 1000}
		if n.status != nil {
			nv.SIMD = n.status.SIMD
			nv.UptimeS = time.Since(time.UnixMilli(int64(n.status.StartedUnixMs))).Seconds()
			copies[name] = map[int]ShardStatus{}
			if n.up {
				for _, s := range n.status.Shards {
					copies[name][s.Shard] = s
				}
			}
		}
		if !n.lastSeen.IsZero() {
			nv.LastSeen = n.lastSeen.UTC().Format(time.RFC3339)
		}
		n.mu.Unlock()
		v.Nodes = append(v.Nodes, nv)
	}
	r.mu.RLock()
	for _, s := range r.shards {
		sv := ShardView{ShardMap: s}
		if p, ok := copies[s.Primary][s.ID]; ok {
			sv.PrimaryLSN, sv.Vectors = p.LSN, p.Vectors
		}
		if rep, ok := copies[s.Replica][s.ID]; ok {
			sv.ReplicaLSN, sv.ReplicaInSync = rep.LSN, rep.InSync
		}
		v.Shards = append(v.Shards, sv)
	}
	r.mu.RUnlock()
	r.eventsMu.Lock()
	v.Events = append([]Event(nil), r.events...)
	r.eventsMu.Unlock()
	sort.Slice(v.Events, func(i, j int) bool { return v.Events[i].Time.After(v.Events[j].Time) })
	return v
}
