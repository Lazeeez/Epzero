package router

import (
	"sync"
	"sync/atomic"
	"time"
)

// Counters the router keeps for its /metrics endpoint (http.go).

var latencyBuckets = []float64{0.5, 1, 2, 5, 10, 20, 50, 100, 250, 500, 1000}

type metrics struct {
	searches, partial, hedges, failovers, writes atomic.Uint64
	mu                                           sync.Mutex
	buckets                                      []uint64
	sumMs                                        float64
	up                                           map[string]bool
}

func newMetrics() *metrics {
	return &metrics{buckets: make([]uint64, len(latencyBuckets)), up: map[string]bool{}}
}

func (m *metrics) observeSearch(d time.Duration, partial bool) {
	m.searches.Add(1)
	if partial {
		m.partial.Add(1)
	}
	ms := float64(d.Microseconds()) / 1000
	m.mu.Lock()
	m.sumMs += ms
	for i, b := range latencyBuckets {
		if ms <= b {
			m.buckets[i]++
		}
	}
	m.mu.Unlock()
}

func (m *metrics) nodeUp(name string, up bool) {
	m.mu.Lock()
	m.up[name] = up
	m.mu.Unlock()
}
