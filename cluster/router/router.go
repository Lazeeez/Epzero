// Package router is the Oryx cluster's coordinator and query front end.
//
// It owns the shard map (which node is primary and which is replica for each shard, and the
// shard's epoch), health-checks every data node, fails over a shard to its replica when the
// primary stops answering, and re-integrates nodes that come back. Queries fan out to one live
// copy of every shard (hedged to the other copy when slow) and the per-shard top-k lists are
// merged; writes are routed by label hash to each shard's primary.
package router

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"sync"
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
	StatePath      string        // where the shard map is persisted ("" = memory only)
	HedgeAfter     time.Duration // send a query to the other copy if the first is this slow
	HealthInterval time.Duration
	DownAfter      int // consecutive failed health checks before a node is declared down
	AllowedOrigins []string
	Logger         *slog.Logger
	// Dial connects to a node; nil means gRPC (DialGRPC). The simulator passes its in-memory nodes.
	Dial func(NodeConfig) (Node, error)
	// Consistency is the default for searches that don't choose one (default Eventual).
	Consistency Consistency
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

	mu     sync.RWMutex // guards shards
	shards []ShardMap

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
	r := &Router{cfg: cfg, log: cfg.Logger, nodes: map[string]*node{}, metrics: newMetrics()}
	for _, nc := range cfg.Nodes {
		client, err := cfg.Dial(nc)
		if err != nil {
			return nil, fmt.Errorf("node %s: %w", nc.Name, err)
		}
		r.nodes[nc.Name] = &node{cfg: nc, client: client}
		r.names = append(r.names, nc.Name)
	}
	if err := r.loadState(); err != nil {
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

func (r *Router) loadState() error {
	if r.cfg.StatePath == "" {
		r.shards = r.initialPlacement()
		return nil
	}
	data, err := os.ReadFile(r.cfg.StatePath)
	if os.IsNotExist(err) {
		r.shards = r.initialPlacement()
		return r.saveStateLocked()
	}
	if err != nil {
		return err
	}
	var saved struct {
		Shards []ShardMap `json:"shards"`
	}
	if err := json.Unmarshal(data, &saved); err != nil {
		return fmt.Errorf("%s: %w", r.cfg.StatePath, err)
	}
	if len(saved.Shards) != r.cfg.Shards {
		return fmt.Errorf("%s has %d shards, config says %d", r.cfg.StatePath, len(saved.Shards), r.cfg.Shards)
	}
	r.shards = saved.Shards
	return nil
}

// saveStateLocked persists the shard map atomically (write + fsync + rename). Caller holds r.mu
// or is single-threaded.
func (r *Router) saveStateLocked() error {
	if r.cfg.StatePath == "" {
		return nil
	}
	data, _ := json.MarshalIndent(struct {
		Shards []ShardMap `json:"shards"`
	}{r.shards}, "", "  ")
	tmp := r.cfg.StatePath + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	f.Close()
	if err := os.Rename(tmp, r.cfg.StatePath); err != nil {
		return err
	}
	if dir, err := os.Open(filepath.Dir(r.cfg.StatePath)); err == nil {
		dir.Sync()
		dir.Close()
	}
	return nil
}

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

// failoverFrom promotes the replica of every shard whose primary is `name`.
func (r *Router) failoverFrom(ctx context.Context, name string) {
	r.mu.Lock()
	var promoted []ShardMap
	for i := range r.shards {
		s := &r.shards[i]
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
		r.metrics.failovers.Add(1)
		r.event("failover", "shard %d: %s promoted to primary (epoch %d)", s.ID, s.Primary, s.Epoch)
	}
	if err := r.saveStateLocked(); err != nil {
		r.log.Error("persisting shard map", "err", err)
	}
	r.mu.Unlock()
	for _, s := range promoted {
		// Its replica (the failed node) gets its role back when it returns.
		r.assign(ctx, s.Primary, s, RolePrimary, r.nodes[s.Replica].cfg.Addr)
	}
}

// reconcile (re)assigns every shard copy involving `name`: primaries first, then replicas, so a
// replica's catch-up finds its primary ready. A primary's replica is re-assigned too, because a
// restarted primary no longer forwards to it until the replica fetches again.
func (r *Router) reconcile(ctx context.Context, name string) {
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

// Bootstrap assigns every shard on every reachable node once at startup.
func (r *Router) Bootstrap(ctx context.Context) {
	r.checkAll(ctx) // marks reachable nodes up and reconciles them
}

// ClusterView is the JSON shape of GET /v1/cluster.
type ClusterView struct {
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
	var v ClusterView
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
