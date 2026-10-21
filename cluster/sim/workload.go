package sim

import (
	"context"
	"math"
	"math/rand/v2"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Lazeeez/Epzero/cluster/router"
	"github.com/anishathalye/porcupine"
)

// Workload drives a simulated cluster the way the site's cluster page shows it: a steady stream
// of searches over a fixed dataset, each checked against an exact answer, and reads and writes of
// a few keys, recorded so the history can be checked for linearizability at any time.
type Workload struct {
	c    *Cluster
	dim  int
	base []float32 // count x dim, labels 0..count-1
	n    int

	queries [][]float32
	truth   [][]uint64 // exact top-k per query

	mu          sync.Mutex
	rate        float64 // searches per second
	wrate       float64 // key operations per second
	consistency router.Consistency
	seconds     []Second
	cur         Second
	lat         []float64 // this second's search latencies (ms)

	hmu     sync.Mutex
	history []porcupine.Operation
	clock   atomic.Int64
	dropped bool // the oldest operations were dropped: keys start unknown
}

// Second summarizes one second of the workload.
type Second struct {
	T            int64   `json:"t"` // unix ms
	Searches     int     `json:"searches"`
	Partial      int     `json:"partial"`
	Failed       int     `json:"failed"`
	Hedged       int     `json:"hedged"`
	Recall       float64 `json:"recall"` // mean recall@10 of complete answers vs. the exact top 10 (-1: none)
	P50          float64 `json:"p50"`
	P99          float64 `json:"p99"`
	Writes       int     `json:"writes"`
	Unreplicated int     `json:"unreplicated"`
	WriteErrors  int     `json:"write_errors"`
	recallSum    float64
	recallN      int
}

// Keys for the read/write stream live far from the dataset, so they never change search answers.
const (
	K          = 10
	kvKeys     = 32
	kvOffset   = 1000
	historyCap = 20000
)

// NewWorkload makes a clustered dataset of `count` vectors and its exact answers for `queries`
// query vectors.
func NewWorkload(c *Cluster, count, dim, queries int, seed uint64) *Workload {
	rng := rand.New(rand.NewPCG(seed, 7))
	w := &Workload{c: c, dim: dim, n: count, rate: 40, wrate: 20, consistency: router.Eventual}
	centers := make([][]float32, 24)
	for i := range centers {
		centers[i] = make([]float32, dim)
		for d := range centers[i] {
			centers[i][d] = float32(rng.NormFloat64() * 3)
		}
	}
	w.base = make([]float32, count*dim)
	for i := range count {
		c := centers[rng.IntN(len(centers))]
		for d := range dim {
			w.base[i*dim+d] = c[d] + float32(rng.NormFloat64())
		}
	}
	for range queries {
		i := rng.IntN(count)
		q := make([]float32, dim)
		for d := range dim {
			q[d] = w.base[i*dim+d] + float32(rng.NormFloat64()*0.5)
		}
		w.queries = append(w.queries, q)
		w.truth = append(w.truth, exactTopK(w.base, dim, q, K))
	}
	return w
}

func exactTopK(base []float32, dim int, q []float32, k int) []uint64 {
	type hit struct {
		label uint64
		d     float32
	}
	hits := make([]hit, len(base)/dim)
	for i := range hits {
		var d float32
		for j, x := range base[i*dim : (i+1)*dim] {
			t := x - q[j]
			d += t * t
		}
		hits[i] = hit{uint64(i), d}
	}
	sort.Slice(hits, func(a, b int) bool {
		return hits[a].d < hits[b].d || hits[a].d == hits[b].d && hits[a].label < hits[b].label
	})
	out := make([]uint64, k)
	for i := range out {
		out[i] = hits[i].label
	}
	return out
}

// Load writes the dataset and the keys (version 1) through the router.
func (w *Workload) Load(ctx context.Context) error {
	const batch = 500
	for start := 0; start < w.n; start += batch {
		end := min(w.n, start+batch)
		labels := make([]uint64, end-start)
		for i := range labels {
			labels[i] = uint64(start + i)
		}
		if _, err := w.c.Router.Upsert(ctx, labels, w.base[start*w.dim:end*w.dim]); err != nil {
			return err
		}
	}
	labels := make([]uint64, kvKeys)
	var vectors []float32
	for k := range kvKeys {
		labels[k] = w.keyLabel(k)
		vectors = append(vectors, w.keyVector(k, 1)...)
	}
	_, err := w.c.Router.Upsert(ctx, labels, vectors)
	return err
}

func (w *Workload) keyLabel(k int) uint64 { return uint64(w.n + k) }

func (w *Workload) keyVector(k, version int) []float32 {
	v := make([]float32, w.dim)
	v[0] = kvOffset + float32(10*k)
	v[1] = float32(version)
	return v
}

// SetConsistency sets which copies searches and key reads may use. The recorded history restarts,
// so a check covers one consistency level.
func (w *Workload) SetConsistency(c router.Consistency) {
	w.mu.Lock()
	w.consistency = c
	w.mu.Unlock()
	w.hmu.Lock()
	w.history, w.dropped = nil, true
	w.hmu.Unlock()
}

func (w *Workload) level() router.Consistency {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.consistency
}

// SetRates sets searches and key operations per second.
func (w *Workload) SetRates(searches, keyOps float64) {
	w.mu.Lock()
	w.rate, w.wrate = searches, keyOps
	w.mu.Unlock()
}

// Run drives the workload until ctx ends.
func (w *Workload) Run(ctx context.Context, seed uint64) {
	go w.searchLoop(ctx, rand.New(rand.NewPCG(seed, 1)))
	for client := range 3 {
		go w.keyLoop(ctx, client, rand.New(rand.NewPCG(seed, uint64(10+client))))
	}
	tick := time.NewTicker(time.Second)
	defer tick.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case now := <-tick.C:
			w.roll(now)
		}
	}
}

func (w *Workload) interval(perSecond float64, rng *rand.Rand) time.Duration {
	if perSecond <= 0 {
		return 200 * time.Millisecond
	}
	// Poisson arrivals.
	return time.Duration(rng.ExpFloat64() / perSecond * float64(time.Second))
}

func (w *Workload) searchLoop(ctx context.Context, rng *rand.Rand) {
	for ctx.Err() == nil {
		w.mu.Lock()
		rate := w.rate
		w.mu.Unlock()
		sleep(ctx, w.interval(rate, rng))
		if rate <= 0 {
			continue
		}
		go w.search(ctx, rng.IntN(len(w.queries)))
	}
}

func (w *Workload) search(ctx context.Context, qi int) {
	res, err := w.c.Router.SearchWith(ctx, w.queries[qi], K, 64, w.level())
	w.mu.Lock()
	defer w.mu.Unlock()
	w.cur.Searches++
	if err != nil {
		w.cur.Failed++
		return
	}
	for _, s := range res.Shards {
		if s.Hedged {
			w.cur.Hedged++
			break
		}
	}
	w.lat = append(w.lat, res.TotalMs)
	if res.Partial {
		w.cur.Partial++
		return
	}
	found := 0
	want := w.truth[qi]
	for _, h := range res.Hits {
		for _, t := range want {
			if h.Label == t {
				found++
				break
			}
		}
	}
	w.cur.recallSum += float64(found) / K
	w.cur.recallN++
}

func (w *Workload) keyLoop(ctx context.Context, client int, rng *rand.Rand) {
	for ctx.Err() == nil {
		w.mu.Lock()
		rate := w.wrate / 3
		w.mu.Unlock()
		sleep(ctx, w.interval(rate, rng))
		if rate <= 0 || ctx.Err() != nil {
			continue
		}
		key := rng.IntN(kvKeys)
		call := w.clock.Add(1)
		switch p := rng.Float64(); {
		case p < 0.4:
			v := 2 + rng.IntN(7)
			res, err := w.c.Router.Upsert(ctx, []uint64{w.keyLabel(key)}, w.keyVector(key, v))
			w.recordWrite(client, kvInput{Key: key, Op: "write", Version: v}, call, err, res != nil && res.Replicated)
		case p < 0.5:
			res, err := w.c.Router.Remove(ctx, w.keyLabel(key))
			w.recordWrite(client, kvInput{Key: key, Op: "remove"}, call, err, res != nil && res.Replicated)
		default:
			q := w.keyVector(key, 0)
			res, err := w.c.Router.SearchWith(ctx, q, 1, 16, w.level())
			if err == nil && !res.Partial {
				seen := 0
				if len(res.Hits) > 0 && res.Hits[0].Label == w.keyLabel(key) {
					seen = int(math.Round(math.Sqrt(float64(res.Hits[0].Distance))))
				}
				w.record(porcupine.Operation{ClientId: client, Input: kvInput{Key: key, Op: "read"}, Output: kvOutput{Version: seen}, Call: call, Return: w.clock.Add(1)})
			}
		}
	}
}

// recordWrite logs a write. One that failed, or that only the primary acknowledged, may or may not
// survive a failover, so it is recorded as ambiguous (it took effect at some unknown time, or never).
func (w *Workload) recordWrite(client int, in kvInput, call int64, err error, replicated bool) {
	ret := w.clock.Add(1)
	if err != nil || !replicated {
		ret = math.MaxInt64
	}
	w.mu.Lock()
	w.cur.Writes++
	if err != nil {
		w.cur.WriteErrors++
	} else if !replicated {
		w.cur.Unreplicated++
	}
	w.mu.Unlock()
	w.record(porcupine.Operation{ClientId: client, Input: in, Output: kvOutput{}, Call: call, Return: ret})
}

func (w *Workload) record(op porcupine.Operation) {
	w.hmu.Lock()
	w.history = append(w.history, op)
	if len(w.history) > historyCap {
		// Keep the newest half. Checking from an unknown starting state is weaker but still sound.
		w.history = append([]porcupine.Operation(nil), w.history[len(w.history)/2:]...)
		w.dropped = true
	}
	w.hmu.Unlock()
}

func (w *Workload) roll(now time.Time) {
	w.mu.Lock()
	defer w.mu.Unlock()
	s := w.cur
	s.T = now.UnixMilli()
	if s.recallN > 0 {
		s.Recall = s.recallSum / float64(s.recallN)
	} else {
		s.Recall = -1 // no complete answers this second (JSON has no NaN)
	}
	if len(w.lat) > 0 {
		sort.Float64s(w.lat)
		s.P50 = w.lat[len(w.lat)/2]
		s.P99 = w.lat[min(len(w.lat)-1, int(float64(len(w.lat))*0.99))]
	}
	w.seconds = append(w.seconds, s)
	if len(w.seconds) > 120 {
		w.seconds = w.seconds[len(w.seconds)-120:]
	}
	w.cur, w.lat = Second{}, w.lat[:0]
}

// Seconds returns the per-second summaries, oldest first (up to two minutes).
func (w *Workload) Seconds() []Second {
	w.mu.Lock()
	defer w.mu.Unlock()
	return append([]Second(nil), w.seconds...)
}

// CheckResult is the outcome of checking the recorded history.
type CheckResult struct {
	Operations int    `json:"operations"`
	Ambiguous  int    `json:"ambiguous"`
	Result     string `json:"result"` // "ok", "illegal" or "unknown" (ran out of time)
	Millis     int64  `json:"millis"`
}

// Check verifies the recorded history (up to the last historyCap operations) with Porcupine.
func (w *Workload) Check(timeout time.Duration) CheckResult {
	w.hmu.Lock()
	ops := append([]porcupine.Operation(nil), w.history...)
	initial := 1
	if w.dropped {
		initial = unknown
	}
	w.hmu.Unlock()
	ambiguous := 0
	for _, op := range ops {
		if op.Return == math.MaxInt64 {
			ambiguous++
		}
	}
	start := time.Now()
	result := porcupine.CheckOperationsTimeout(kvModel(initial), ops, timeout)
	r := CheckResult{Operations: len(ops), Ambiguous: ambiguous, Millis: time.Since(start).Milliseconds()}
	switch result {
	case porcupine.Ok:
		r.Result = "ok"
	case porcupine.Illegal:
		r.Result = "illegal"
	default:
		r.Result = "unknown"
	}
	return r
}
