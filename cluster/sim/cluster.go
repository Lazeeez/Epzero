package sim

import (
	"context"
	"io"
	"log/slog"
	"time"

	"github.com/Lazeeez/Oryx/cluster/router"
)

// RouterName is the router's endpoint on the simulated network.
const RouterName = "router"

// Cluster is a router and its simulated nodes on one network.
type Cluster struct {
	Net    *Network
	Nodes  map[string]*Node
	Names  []string
	Router *router.Router
	done   <-chan struct{} // closed when the cluster stops (Start's context ends)
}

// Options configures a simulated cluster.
type Options struct {
	Nodes  []string // node names (at least 2)
	Shards int      // default: one per node
	Dim    int
	Link   Link // every link's behavior until changed
	Seed   uint64
	// Router timing; zero values take the router's defaults.
	HealthInterval time.Duration
	HedgeAfter     time.Duration
	Consistency    router.Consistency
	Logger         *slog.Logger
}

// New builds the cluster; call Start to run the router's health checks.
func New(opts Options) (*Cluster, error) {
	c := &Cluster{Net: NewNetwork(opts.Link, opts.Seed), Nodes: map[string]*Node{}, Names: opts.Nodes}
	cfg := router.Config{
		Shards: opts.Shards, Dim: opts.Dim, Token: "sim", HealthInterval: opts.HealthInterval, HedgeAfter: opts.HedgeAfter,
		Logger: opts.Logger, Dial: c.dial, Consistency: opts.Consistency,
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
	}
	for _, name := range opts.Nodes {
		c.Nodes[name] = newNode(name, c)
		cfg.Nodes = append(cfg.Nodes, router.NodeConfig{Name: name, Addr: name})
	}
	r, err := router.New(cfg)
	if err != nil {
		return nil, err
	}
	c.Router = r
	return c, nil
}

// Start assigns every shard and runs the router's health checks until ctx ends.
func (c *Cluster) Start(ctx context.Context) {
	c.done = ctx.Done()
	c.Router.Bootstrap(ctx)
	go c.Router.Run(ctx)
}

// Partition cuts every link between the two groups (both directions).
func (c *Cluster) Partition(a, b []string) {
	for _, x := range a {
		for _, y := range b {
			c.Net.SetBoth(x, y, Link{Cut: true})
		}
	}
}

func (c *Cluster) dial(nc router.NodeConfig) (router.Node, error) {
	return client{c: c, from: RouterName, to: c.Nodes[nc.Name]}, nil
}

// client is the router's handle on one node: every call crosses the simulated network.
type client struct {
	c    *Cluster
	from string
	to   *Node
}

func (cl client) call(ctx context.Context, handler func() error) error {
	return cl.c.Net.call(ctx, cl.from, cl.to.Name, cl.to, handler)
}

func (cl client) Status(ctx context.Context) (*router.NodeStatus, error) {
	var st *router.NodeStatus
	err := cl.call(ctx, func() error { st = cl.to.status(); return nil })
	return st, err
}

func (cl client) Assign(ctx context.Context, a router.Assignment) error {
	return cl.call(ctx, func() error { return cl.to.assign(a) })
}

func (cl client) Search(ctx context.Context, shard int, query []float32, k, ef int) (*router.ShardReply, error) {
	var reply *router.ShardReply
	err := cl.call(ctx, func() error {
		var e error
		reply, e = cl.to.search(shard, query, k)
		return e
	})
	return reply, err
}

func (cl client) Upsert(ctx context.Context, shard int, epoch uint64, labels []uint64, vectors []float32) (bool, error) {
	var replicated bool
	err := cl.call(ctx, func() error {
		dim := len(vectors) / max(1, len(labels))
		records := make([]record, len(labels))
		for i, label := range labels {
			records[i] = record{label: label, vector: append([]float32(nil), vectors[i*dim:(i+1)*dim]...)}
		}
		var e error
		replicated, e = cl.to.write(ctx, shard, epoch, records)
		return e
	})
	return replicated, err
}

func (cl client) Remove(ctx context.Context, shard int, epoch uint64, label uint64) (bool, error) {
	var replicated bool
	err := cl.call(ctx, func() error {
		var e error
		replicated, e = cl.to.write(ctx, shard, epoch, []record{{remove: true, label: label}})
		return e
	})
	return replicated, err
}

func (cl client) Crash(ctx context.Context, token string) error {
	return cl.call(ctx, func() error {
		go func() {
			time.Sleep(50 * time.Millisecond) // let the reply leave
			cl.to.Kill()
		}()
		return nil
	})
}

// stopped reports whether the cluster's context has ended (background loops then exit).
func (c *Cluster) stopped() bool {
	select {
	case <-c.done:
		return true
	default:
		return false
	}
}
