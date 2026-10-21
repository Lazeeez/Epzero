//go:build !js

package router_test

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/embed"

	"github.com/Lazeeez/Epzero/cluster/router"
	"github.com/Lazeeez/Epzero/cluster/sim"
)

// startEtcd runs a single-member etcd in a temporary directory and returns a client factory.
func startEtcd(t *testing.T) func() *clientv3.Client {
	t.Helper()
	cfg := embed.NewConfig()
	cfg.Dir = t.TempDir()
	cfg.LogLevel = "error"
	client, _ := url.Parse("http://127.0.0.1:0")
	peer, _ := url.Parse("http://127.0.0.1:0")
	cfg.ListenClientUrls, cfg.AdvertiseClientUrls = []url.URL{*client}, []url.URL{*client}
	cfg.ListenPeerUrls, cfg.AdvertisePeerUrls = []url.URL{*peer}, []url.URL{*peer}
	cfg.InitialCluster = cfg.Name + "=" + peer.String()
	server, err := embed.StartEtcd(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(server.Close)
	select {
	case <-server.Server.ReadyNotify():
	case <-time.After(20 * time.Second):
		t.Fatal("etcd did not start")
	}
	endpoint := server.Clients[0].Addr().String()
	return func() *clientv3.Client {
		c, err := router.DialEtcd([]string{endpoint})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { c.Close() })
		return c
	}
}

func waitFor(t *testing.T, what string, timeout time.Duration, ok func() bool) {
	t.Helper()
	for deadline := time.Now().Add(timeout); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		if ok() {
			return
		}
	}
	t.Fatalf("timed out waiting for %s", what)
}

func primaryOf(r *router.Router, shard int) string {
	for _, s := range r.View().Shards {
		if s.ID == shard {
			return s.Primary
		}
	}
	return ""
}

// Two routers share an etcd cluster: one leads and fails shards over, the other follows the map;
// when the leader stops, the other takes over within the session TTL and keeps failing over.
func TestTwoRoutersShareTheMapThroughEtcd(t *testing.T) {
	dial := startEtcd(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	c, err := sim.New(sim.Options{
		Nodes: []string{"n1", "n2", "n3"}, Dim: 4, HealthInterval: 100 * time.Millisecond,
		Store: router.NewEtcdStore(dial(), "test", 1),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctxA, stopA := context.WithCancel(ctx)
	c.Start(ctxA) // router A ("router")
	b, err := c.AddRouter("router-b", router.NewEtcdStore(dial(), "test", 1))
	if err != nil {
		t.Fatal(err)
	}
	b.Bootstrap(ctx)
	go b.Run(ctx)

	waitFor(t, "router A to lead", 10*time.Second, c.Router.Leading)
	if b.Leading() {
		t.Fatal("both routers lead")
	}

	// Some data, written through the follower (writes need no leader, only the map).
	labels := make([]uint64, 30)
	var vectors []float32
	for i := range labels {
		labels[i] = uint64(i)
		vectors = append(vectors, float32(i), 0, 0, 0)
	}
	waitFor(t, "replicas in sync", 10*time.Second, func() bool {
		for _, s := range c.Router.View().Shards {
			if !s.ReplicaInSync {
				return false
			}
		}
		return true
	})
	if _, err := b.Upsert(ctx, labels, vectors); err != nil {
		t.Fatal(err)
	}

	// The leader fails n1's shards over; the follower sees the new map through its watch.
	before := primaryOf(b, 0)
	c.Nodes[before].Kill()
	waitFor(t, "router B to see shard 0 fail over", 10*time.Second, func() bool { return primaryOf(b, 0) != before })
	res, err := b.SearchWith(ctx, []float32{7, 0, 0, 0}, 1, 16, router.Strong)
	if err != nil || res.Partial || len(res.Hits) != 1 || res.Hits[0].Label != 7 {
		t.Fatalf("search through the follower after failover: %+v, %v", res, err)
	}

	// The leader stops; B takes over and handles the next failure.
	stopA()
	waitFor(t, "router B to take over", 15*time.Second, b.Leading)
	c.Nodes[before].Restart()
	waitFor(t, fmt.Sprintf("%s to rejoin", before), 10*time.Second, func() bool {
		for _, n := range b.View().Nodes {
			if n.Name == before {
				return n.Up
			}
		}
		return false
	})
	victim := primaryOf(b, 1)
	c.Nodes[victim].Kill()
	waitFor(t, "router B to fail shard 1 over", 10*time.Second, func() bool { return primaryOf(b, 1) != victim })
}

// A deposed leader cannot change the map: its saves are fenced by the leader key.
func TestEtcdFencesADeposedLeader(t *testing.T) {
	dial := startEtcd(t)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	client := dial()
	store := router.NewEtcdStore(client, "fence", 1)
	shards := []router.ShardMap{{ID: 0, Epoch: 1, Primary: "a", Replica: "b"}}
	version, err := store.Save(ctx, shards, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Save(ctx, shards, version); !errors.Is(err, router.ErrNotLeader) {
		t.Fatalf("a non-leader saved: %v", err)
	}
	leadCtx, resign := context.WithCancel(ctx)
	lead := store.Lead(leadCtx, "a")
	if !<-lead {
		t.Fatal("did not become leader")
	}
	shards[0].Epoch = 2
	if version, err = store.Save(ctx, shards, version); err != nil {
		t.Fatal(err)
	}
	// Another router takes the lead (the first one's lease is revoked, as if it had been cut off).
	leases, _ := client.Leases(ctx)
	for _, l := range leases.Leases {
		client.Revoke(ctx, l.ID)
	}
	other := router.NewEtcdStore(dial(), "fence", 1)
	if !<-other.Lead(ctx, "b") {
		t.Fatal("the second router did not take over")
	}
	shards[0].Epoch = 3
	if _, err := store.Save(ctx, shards, version); !errors.Is(err, router.ErrNotLeader) && !errors.Is(err, router.ErrConflict) {
		t.Fatalf("the deposed leader saved: %v", err)
	}
	if _, err := other.Save(ctx, shards, version); err != nil {
		t.Fatalf("the new leader could not save: %v", err)
	}
	resign()
}
