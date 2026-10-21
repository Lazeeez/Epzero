//go:build integration

// End-to-end test against real epzero-node processes:
//
//	EPZERO_NODE_BIN=/path/to/epzero-node go test -tags integration ./router -v
package router

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"syscall"
	"testing"
	"time"
)

const dim = 32

type proc struct {
	name, addr, dir string
	cmd             *exec.Cmd
}

func freePort(t *testing.T) int {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func (p *proc) start(t *testing.T, bin, tokenFile string) {
	p.cmd = exec.Command(bin, "--listen", p.addr, "--name", p.name, "--data", p.dir, "--dim", fmt.Sprint(dim),
		"--metric", "l2", "--max", "20000", "--token-file", tokenFile, "--sync", "always")
	p.cmd.Stderr = os.Stderr
	if err := p.cmd.Start(); err != nil {
		t.Fatal(err)
	}
}

func (p *proc) kill() {
	p.cmd.Process.Signal(syscall.SIGKILL)
	p.cmd.Wait()
}

func randomVectors(n int, seed int64) [][]float32 {
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, 16)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for d := range centers[i] {
			centers[i][d] = float32(rng.NormFloat64())
		}
	}
	out := make([][]float32, n)
	for i := range out {
		c := centers[rng.Intn(len(centers))]
		out[i] = make([]float32, dim)
		for d := range out[i] {
			out[i][d] = c[d] + 0.3*float32(rng.NormFloat64())
		}
	}
	return out
}

func l2(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func exactTopK(data map[uint64][]float32, q []float32, k int) []uint64 {
	type pair struct {
		label uint64
		d     float32
	}
	all := make([]pair, 0, len(data))
	for l, v := range data {
		all = append(all, pair{l, l2(q, v)})
	}
	sort.Slice(all, func(i, j int) bool { return all[i].d < all[j].d })
	out := make([]uint64, 0, k)
	for i := 0; i < k && i < len(all); i++ {
		out = append(out, all[i].label)
	}
	return out
}

func waitFor(t *testing.T, what string, timeout time.Duration, ok func() bool) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if ok() {
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestClusterFailoverAndRecovery(t *testing.T) {
	bin := os.Getenv("EPZERO_NODE_BIN")
	if bin == "" {
		t.Skip("set EPZERO_NODE_BIN to the epzero-node binary")
	}
	base := t.TempDir()
	tokenFile := filepath.Join(base, "token")
	os.WriteFile(tokenFile, []byte("test-token\n"), 0o600)

	procs := make([]*proc, 3)
	var nodes []NodeConfig
	for i := range procs {
		p := &proc{name: fmt.Sprintf("node%d", i+1), addr: fmt.Sprintf("127.0.0.1:%d", freePort(t)),
			dir: filepath.Join(base, fmt.Sprintf("node%d", i+1))}
		p.start(t, bin, tokenFile)
		defer p.kill()
		procs[i] = p
		nodes = append(nodes, NodeConfig{Name: p.name, Addr: p.addr})
	}

	r, err := New(Config{Nodes: nodes, Dim: dim, Token: "test-token", StatePath: filepath.Join(base, "router.json"),
		HealthInterval: 200 * time.Millisecond, Logger: slog.New(slog.NewTextHandler(os.Stderr, nil))})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go r.Run(ctx)
	waitFor(t, "all replicas in sync", 20*time.Second, func() bool {
		for _, s := range r.View().Shards {
			if !s.ReplicaInSync {
				return false
			}
		}
		return true
	})

	// Load 3,000 vectors.
	vectors := randomVectors(4000, 1)
	data := map[uint64][]float32{}
	upsert := func(lo, hi int) *WriteResult {
		var labels []uint64
		var flat []float32
		for i := lo; i < hi; i++ {
			labels = append(labels, uint64(i))
			flat = append(flat, vectors[i]...)
			data[uint64(i)] = vectors[i]
		}
		res, err := r.Upsert(ctx, labels, flat)
		if err != nil {
			t.Fatalf("upsert %d..%d: %v", lo, hi, err)
		}
		return res
	}
	for lo := 0; lo < 3000; lo += 250 {
		if res := upsert(lo, lo+250); !res.Replicated {
			t.Fatalf("healthy cluster write was not replicated")
		}
	}

	recall := func() float64 {
		queries := randomVectors(50, 7)
		hits := 0
		for _, q := range queries {
			res, err := r.Search(ctx, q, 10, 128)
			if err != nil {
				t.Fatal(err)
			}
			truth := map[uint64]bool{}
			for _, l := range exactTopK(data, q, 10) {
				truth[l] = true
			}
			for _, h := range res.Hits {
				if truth[h.Label] {
					hits++
				}
			}
		}
		return float64(hits) / float64(len(queries)*10)
	}
	if got := recall(); got < 0.95 {
		t.Fatalf("recall %.3f before failure", got)
	}

	// Kill node2 without warning. Every shard still has a copy, so searches stay complete.
	victim := procs[1]
	victim.kill()
	waitFor(t, "node2 marked down", 10*time.Second, func() bool { return !r.isUp("node2") })
	for _, s := range r.View().Shards {
		if s.Primary == "node2" {
			t.Fatalf("shard %d still has the dead node as primary", s.ID)
		}
	}
	if got := recall(); got < 0.95 {
		t.Fatalf("recall %.3f with a node down", got)
	}
	// Writes keep working (single copy for the shards that lost a replica).
	upsert(3000, 3500)

	// Bring it back: it recovers from its log, rejoins as a replica and catches up.
	victim.start(t, bin, tokenFile)
	waitFor(t, "node2 back and every replica in sync", 60*time.Second, func() bool {
		if !r.isUp("node2") {
			return false
		}
		for _, s := range r.View().Shards {
			if !s.ReplicaInSync || s.ReplicaLSN != s.PrimaryLSN {
				return false
			}
		}
		return true
	})
	upsert(3500, 4000)
	if got := recall(); got < 0.95 {
		t.Fatalf("recall %.3f after recovery", got)
	}

	// Nothing acknowledged was lost: every label is its own nearest neighbour.
	for label := uint64(0); label < 4000; label += 97 {
		res, err := r.Search(ctx, data[label], 1, 64)
		if err != nil || len(res.Hits) == 0 || res.Hits[0].Label != label || math.Abs(float64(res.Hits[0].Distance)) > 1e-4 {
			t.Fatalf("label %d missing after failover/recovery: %+v %v", label, res, err)
		}
	}
	// Shard sizes come from the periodic status poll, so allow one refresh.
	waitFor(t, "status to report all 4000 vectors", 5*time.Second, func() bool {
		var total uint64
		for _, s := range r.View().Shards {
			total += s.Vectors
		}
		return total == 4000
	})
	t.Logf("events:\n%v", r.View().Events)
}
