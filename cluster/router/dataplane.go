package router

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	pb "github.com/Lazeeez/Oryx/cluster/gen/oryxv1"
)

// ShardOf maps a label to its shard: splitmix64 finalizer, so sequential labels spread evenly.
func ShardOf(label uint64, shards int) int {
	x := label + 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	x ^= x >> 31
	return int(x % uint64(shards))
}

// Hit is one merged search result.
type Hit struct {
	Label    uint64  `json:"label"`
	Distance float32 `json:"distance"`
	Shard    int     `json:"shard"`
}

// ShardTiming reports how one shard's part of a query was served.
type ShardTiming struct {
	Shard    int     `json:"shard"`
	Node     string  `json:"node"`
	Role     string  `json:"role"`
	NodeUs   uint64  `json:"node_us"`  // search time inside the node
	RoundMs  float64 `json:"round_ms"` // router <-> node round trip
	Hedged   bool    `json:"hedged"`   // the backup copy was also asked
	Scored   uint64  `json:"scored"`   // distance computations on the node
	Error    string  `json:"error,omitempty"`
	Degraded bool    `json:"degraded"` // answered by the only live copy
}

// SearchResult is the merged answer to one query.
type SearchResult struct {
	Hits    []Hit         `json:"hits"`
	Shards  []ShardTiming `json:"shards"`
	TotalMs float64       `json:"total_ms"`
	Partial bool          `json:"partial"` // some shard had no live copy; results cover the rest
}

type copyAnswer struct {
	node  string
	role  string
	reply *pb.SearchReply
	err   error
	took  time.Duration
}

// Search queries every shard (hedged across its two copies) and merges the top k.
func (r *Router) Search(ctx context.Context, query []float32, k, ef int) (*SearchResult, error) {
	if len(query) != r.cfg.Dim {
		return nil, fmt.Errorf("query has %d dimensions, expected %d", len(query), r.cfg.Dim)
	}
	start := time.Now()
	r.mu.RLock()
	shards := append([]ShardMap(nil), r.shards...)
	r.mu.RUnlock()

	result := &SearchResult{Shards: make([]ShardTiming, len(shards))}
	hits := make([][]Hit, len(shards))
	var wg sync.WaitGroup
	for i, s := range shards {
		wg.Add(1)
		go func(i int, s ShardMap) {
			defer wg.Done()
			timing, shardHits := r.searchShard(ctx, s, query, k, ef)
			result.Shards[i] = timing
			hits[i] = shardHits
		}(i, s)
	}
	wg.Wait()

	for i, shardHits := range hits {
		if result.Shards[i].Error != "" {
			result.Partial = true
		}
		result.Hits = append(result.Hits, shardHits...)
	}
	if len(result.Hits) == 0 && result.Partial {
		return nil, errors.New("no shard could answer")
	}
	sort.Slice(result.Hits, func(a, b int) bool { return result.Hits[a].Distance < result.Hits[b].Distance })
	if len(result.Hits) > k {
		result.Hits = result.Hits[:k]
	}
	result.TotalMs = float64(time.Since(start).Microseconds()) / 1000
	r.metrics.observeSearch(time.Since(start), result.Partial)
	return result, nil
}

func (r *Router) searchShard(ctx context.Context, s ShardMap, query []float32, k, ef int) (ShardTiming, []Hit) {
	timing := ShardTiming{Shard: s.ID}
	// Live copies, rotating which goes first to spread load across primary and replica.
	var copies []string
	for _, name := range []string{s.Primary, s.Replica} {
		if r.isUp(name) && (name == s.Primary || r.replicaInSync(name, s.ID)) {
			copies = append(copies, name)
		}
	}
	if len(copies) == 0 {
		timing.Error = "no live copy"
		return timing, nil
	}
	if len(copies) == 2 && atomic.AddUint64(&r.rr, 1)%2 == 1 {
		copies[0], copies[1] = copies[1], copies[0]
	}
	timing.Degraded = len(copies) == 1

	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	answers := make(chan copyAnswer, len(copies))
	ask := func(name string) {
		role := "replica"
		if name == s.Primary {
			role = "primary"
		}
		t := time.Now()
		reply, err := r.nodes[name].client.Search(ctx, &pb.SearchRequest{Shard: uint32(s.ID), Query: query, K: uint32(k), Ef: uint32(ef)})
		answers <- copyAnswer{node: name, role: role, reply: reply, err: err, took: time.Since(t)}
	}
	go ask(copies[0])
	pending := 1
	var hedge <-chan time.Time
	if len(copies) > 1 {
		timer := time.NewTimer(r.cfg.HedgeAfter)
		defer timer.Stop()
		hedge = timer.C
	}
	var lastErr error
	for pending > 0 || hedge != nil {
		select {
		case <-hedge:
			// The first copy is slow (GC pause, loaded node, packet loss): race the other one.
			hedge = nil
			timing.Hedged = true
			r.metrics.hedges.Add(1)
			pending++
			go ask(copies[1])
		case a := <-answers:
			pending--
			if a.err != nil {
				lastErr = a.err
				if hedge != nil { // fail over immediately instead of waiting for the hedge timer
					hedge = nil
					pending++
					go ask(copies[1])
				}
				continue
			}
			timing.Node, timing.Role = a.node, a.role
			timing.NodeUs, timing.Scored = a.reply.Micros, a.reply.DistanceComputations
			timing.RoundMs = float64(a.took.Microseconds()) / 1000
			hits := make([]Hit, len(a.reply.Hits))
			for i, h := range a.reply.Hits {
				hits[i] = Hit{Label: h.Label, Distance: h.Distance, Shard: s.ID}
			}
			return timing, hits
		case <-ctx.Done():
			timing.Error = ctx.Err().Error()
			return timing, nil
		}
	}
	timing.Error = fmt.Sprint(lastErr)
	return timing, nil
}

func (r *Router) replicaInSync(name string, shard int) bool {
	n := r.nodes[name]
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.status == nil {
		return false
	}
	for _, s := range n.status.Shards {
		if int(s.Shard) == shard {
			return s.InSync
		}
	}
	return false
}

// WriteResult reports a write's outcome per shard.
type WriteResult struct {
	Written    int  `json:"written"`
	Replicated bool `json:"replicated"` // every shard's replica acknowledged too
}

// Upsert routes each vector to its shard's primary. vectors is len(labels)*Dim values.
func (r *Router) Upsert(ctx context.Context, labels []uint64, vectors []float32) (*WriteResult, error) {
	if len(vectors) != len(labels)*r.cfg.Dim {
		return nil, fmt.Errorf("expected %d vector values, got %d", len(labels)*r.cfg.Dim, len(vectors))
	}
	groups := map[int]*pb.UpsertRequest{}
	for i, label := range labels {
		shard := ShardOf(label, r.cfg.Shards)
		g := groups[shard]
		if g == nil {
			g = &pb.UpsertRequest{Shard: uint32(shard)}
			groups[shard] = g
		}
		g.Labels = append(g.Labels, label)
		g.Vectors = append(g.Vectors, vectors[i*r.cfg.Dim:(i+1)*r.cfg.Dim]...)
	}
	result := &WriteResult{Replicated: true}
	var mu sync.Mutex
	var firstErr error
	var wg sync.WaitGroup
	for shard, req := range groups {
		wg.Add(1)
		go func(shard int, req *pb.UpsertRequest) {
			defer wg.Done()
			reply, err := r.writeToPrimary(ctx, shard, func(ctx context.Context, client pb.NodeClient, epoch uint64) (*pb.WriteReply, error) {
				req.Epoch = epoch
				return client.Upsert(ctx, req)
			})
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				if firstErr == nil {
					firstErr = fmt.Errorf("shard %d: %w", shard, err)
				}
				return
			}
			result.Written += len(req.Labels)
			result.Replicated = result.Replicated && reply.Replicated
		}(shard, req)
	}
	wg.Wait()
	r.metrics.writes.Add(uint64(result.Written))
	return result, firstErr
}

// Remove deletes a label from its shard.
func (r *Router) Remove(ctx context.Context, label uint64) (*WriteResult, error) {
	shard := ShardOf(label, r.cfg.Shards)
	reply, err := r.writeToPrimary(ctx, shard, func(ctx context.Context, client pb.NodeClient, epoch uint64) (*pb.WriteReply, error) {
		return client.Remove(ctx, &pb.RemoveRequest{Shard: uint32(shard), Epoch: epoch, Label: label})
	})
	if err != nil {
		return nil, err
	}
	return &WriteResult{Written: 1, Replicated: reply.Replicated}, nil
}

// writeToPrimary sends a write to the shard's current primary, retrying once after a short wait if
// the shard is mid-failover (the primary refused with a stale epoch or is unreachable).
func (r *Router) writeToPrimary(ctx context.Context, shard int,
	call func(context.Context, pb.NodeClient, uint64) (*pb.WriteReply, error)) (*pb.WriteReply, error) {
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		r.mu.RLock()
		s := r.shards[shard]
		r.mu.RUnlock()
		if r.isUp(s.Primary) {
			cctx, cancel := context.WithTimeout(ctx, 10*time.Second)
			var reply *pb.WriteReply
			reply, err = call(cctx, r.nodes[s.Primary].client, s.Epoch)
			cancel()
			if err == nil {
				return reply, nil
			}
		} else {
			err = fmt.Errorf("primary %s is down", s.Primary)
		}
		time.Sleep(time.Duration(500*(attempt+1)) * time.Millisecond)
	}
	return nil, err
}

// Crash asks a node to die abruptly (chaos testing).
func (r *Router) Crash(ctx context.Context, name string) error {
	n, ok := r.nodes[name]
	if !ok {
		return fmt.Errorf("unknown node %q", name)
	}
	cctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if _, err := n.client.Crash(cctx, &pb.CrashRequest{Token: r.cfg.Token}); err != nil {
		return err
	}
	r.event("crash", "chaos: crashed %s on request", name)
	return nil
}
