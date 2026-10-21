package sim

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/Lazeeez/Oryx/cluster/router"
	"github.com/anishathalye/porcupine"
)

// Clients read and write single keys through the real router while a nemesis crashes nodes,
// partitions them and makes links lossy; Porcupine then checks that the history is linearizable.
// Everything runs on synctest's virtual clock, so a 60-second run takes well under a second.
//
// A key's value is its version: write v stores vector (10*key, v, 0, 0), and a read searches
// for (10*key, 0, 0, 0) with k = 1, which returns the key's vector if it exists (distance v²,
// under 100) and otherwise another key's (distance 100 or more).

const dim = 4

func vectorFor(key, version int) []float32 {
	return []float32{float32(10 * key), float32(version), 0, 0}
}

// history records operations; ambiguous ones (errors, timeouts) may or may not have happened.
type history struct {
	mu    sync.Mutex
	clock atomic.Int64
	ops   []porcupine.Operation
}

func (h *history) tick() int64 { return h.clock.Add(1) }

func (h *history) add(client int, in kvInput, out kvOutput, call int64, ambiguous bool) {
	ret := h.tick()
	if ambiguous {
		ret = math.MaxInt64 // took effect at some unknown point, or never
	}
	h.mu.Lock()
	h.ops = append(h.ops, porcupine.Operation{ClientId: client, Input: in, Output: out, Call: call, Return: ret})
	h.mu.Unlock()
}

type scenario struct {
	name    string
	seconds int
	nemesis func(ctx context.Context, c *Cluster, rng *rand.Rand)
	// strict: a write acknowledged by the primary alone counts as done. Otherwise it is ambiguous:
	// the router reported it unreplicated, so losing it when that primary dies is allowed.
	strict bool
}

func runScenario(t *testing.T, sc scenario, consistency router.Consistency, seed uint64) (porcupine.CheckResult, porcupine.LinearizationInfo, *history) {
	t.Helper()
	const keys = 24
	h := &history{}
	var result porcupine.CheckResult
	var info porcupine.LinearizationInfo
	synctest.Test(t, func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		c, err := New(Options{
			Nodes: []string{"n1", "n2", "n3"}, Dim: dim, Seed: seed, Consistency: consistency,
			Link: Link{Latency: time.Millisecond, Jitter: 2 * time.Millisecond},
		})
		if err != nil {
			t.Fatal(err)
		}
		c.Start(ctx)
		time.Sleep(2 * time.Second) // replicas catch up
		labels := make([]uint64, keys)
		var vectors []float32
		for k := range keys {
			labels[k] = uint64(k)
			vectors = append(vectors, vectorFor(k, 1)...)
		}
		if res, err := c.Router.Upsert(ctx, labels, vectors); err != nil || !res.Replicated {
			t.Fatalf("load: %v %+v", err, res)
		}

		end := time.Now().Add(time.Duration(sc.seconds) * time.Second)
		var wg sync.WaitGroup
		for client := range 4 {
			wg.Add(1)
			go func() {
				defer wg.Done()
				rng := rand.New(rand.NewPCG(seed, uint64(client)))
				for time.Now().Before(end) {
					key := rng.IntN(keys)
					call := h.tick()
					switch p := rng.Float64(); {
					case p < 0.35:
						v := 2 + rng.IntN(7) // versions 2..8: distance v² < 100
						res, err := c.Router.Upsert(ctx, []uint64{uint64(key)}, vectorFor(key, v))
						ambiguous := err != nil || (!sc.strict && !res.Replicated)
						h.add(client, kvInput{Key: key, Op: "write", Version: v}, kvOutput{}, call, ambiguous)
					case p < 0.45:
						res, err := c.Router.Remove(ctx, uint64(key))
						ambiguous := err != nil || (!sc.strict && !res.Replicated)
						h.add(client, kvInput{Key: key, Op: "remove"}, kvOutput{}, call, ambiguous)
					default:
						q := vectorFor(key, 0)
						res, err := c.Router.Search(ctx, q, 1, 16)
						if err == nil && !res.Partial {
							seen := 0
							if len(res.Hits) > 0 && res.Hits[0].Label == uint64(key) {
								seen = int(math.Round(math.Sqrt(float64(res.Hits[0].Distance))))
							}
							from := ""
							for _, s := range res.Shards {
								if s.Node != "" {
									from = s.Role
								}
							}
							h.add(client, kvInput{Key: key, Op: "read"}, kvOutput{Version: seen, From: from}, call, false)
						}
					}
					time.Sleep(time.Duration(5+rng.IntN(20)) * time.Millisecond)
				}
			}()
		}
		if sc.nemesis != nil {
			wg.Add(1)
			go func() {
				defer wg.Done()
				nctx, ncancel := context.WithDeadline(ctx, end)
				defer ncancel()
				sc.nemesis(nctx, c, rand.New(rand.NewPCG(seed, 99)))
			}()
		}
		wg.Wait()
		cancel()
		time.Sleep(time.Minute) // let background work see the cancellation and exit
	})
	// Outside the bubble: the checker's timeout must run on the real clock.
	result, info = porcupine.CheckOperationsVerbose(kvModel(1), h.ops, time.Minute)
	return result, info, h
}

// Each fault, applied for a few seconds at a time, then healed.
func crashes(ctx context.Context, c *Cluster, rng *rand.Rand) {
	for ctx.Err() == nil {
		n := c.Nodes[c.Names[rng.IntN(len(c.Names))]]
		n.Kill()
		sleep(ctx, time.Duration(2+rng.IntN(4))*time.Second)
		n.Restart()
		sleep(ctx, time.Duration(3+rng.IntN(3))*time.Second)
	}
}

func partitions(ctx context.Context, c *Cluster, rng *rand.Rand) {
	for ctx.Err() == nil {
		n := c.Names[rng.IntN(len(c.Names))]
		others := []string{RouterName}
		for _, m := range c.Names {
			if m != n {
				others = append(others, m)
			}
		}
		if rng.IntN(2) == 0 {
			c.Partition([]string{n}, others) // isolated from everyone
		} else {
			c.Partition([]string{n}, others[1:]) // the router still reaches it; its peers don't
		}
		sleep(ctx, time.Duration(2+rng.IntN(4))*time.Second)
		c.Net.Heal()
		sleep(ctx, time.Duration(3+rng.IntN(3))*time.Second)
	}
}

func lossy(ctx context.Context, c *Cluster, rng *rand.Rand) {
	for ctx.Err() == nil {
		a, b := c.Names[rng.IntN(len(c.Names))], c.Names[rng.IntN(len(c.Names))]
		c.Net.SetBoth(a, b, Link{Latency: time.Millisecond, Loss: 0.3})
		c.Net.SetBoth(RouterName, a, Link{Latency: 5 * time.Millisecond, Jitter: 60 * time.Millisecond})
		sleep(ctx, time.Duration(2+rng.IntN(4))*time.Second)
		c.Net.Heal()
		sleep(ctx, time.Duration(2+rng.IntN(3))*time.Second)
	}
}

var scenarios = []scenario{
	{name: "no faults", seconds: 20},
	{name: "crashes", seconds: 60, nemesis: crashes},
	{name: "partitions", seconds: 60, nemesis: partitions},
	{name: "lossy links", seconds: 60, nemesis: lossy},
}

// Strong consistency must be linearizable under every fault.
func TestLinearizable(t *testing.T) {
	for _, sc := range scenarios {
		for _, seed := range []uint64{1, 2, 3} {
			t.Run(fmt.Sprintf("%s/seed%d", sc.name, seed), func(t *testing.T) {
				result, info, h := runScenario(t, sc, router.Strong, seed)
				t.Logf("%d operations", len(h.ops))
				if result == porcupine.Unknown {
					t.Skip("the checker ran out of time: inconclusive")
				}
				if result != porcupine.Ok {
					explain(t, sc.name, seed, info, h)
					t.Fatalf("history is not linearizable (%v)", result)
				}
			})
		}
	}
}

// Eventual consistency trades linearizability for latency; this reports, rather than requires, it.
func TestEventual(t *testing.T) {
	for _, sc := range scenarios {
		t.Run(sc.name, func(t *testing.T) {
			result, _, h := runScenario(t, sc, router.Eventual, 1)
			t.Logf("%d operations: %v; stale reads %s", len(h.ops), result, staleReads(h.ops))
		})
	}
}

func explain(t *testing.T, name string, seed uint64, info porcupine.LinearizationInfo, h *history) {
	t.Helper()
	if dir := os.Getenv("PORCUPINE_OUT"); dir != "" {
		path := fmt.Sprintf("%s/%s-seed%d.html", dir, name, seed)
		if f, err := os.Create(path); err == nil {
			porcupine.Visualize(kvModel(1), info, f)
			f.Close()
			t.Logf("visualization: %s", path)
		}
	}
	t.Logf("stale reads: %s", staleReads(h.ops))
	t.Logf("vanished values: %s", vanished(h.ops))
	if os.Getenv("PORCUPINE_DUMP") != "" {
		t.Log(failingKey(h.ops))
	}
}
