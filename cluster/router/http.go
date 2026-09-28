package router

import (
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Handler serves the public HTTP API:
//
//	GET  /healthz            liveness
//	GET  /v1/cluster         nodes, shard map, replication state, recent events
//	POST /v1/search          {"vector": [...], "k": 10, "ef": 64}
//	POST /v1/upsert          {"labels": [...], "vectors": [[...], ...]}      (token)
//	POST /v1/remove          {"label": 42}                                   (token)
//	POST /v1/chaos/crash     {"node": "node2"}                               (token)
//	GET  /metrics            Prometheus text format
func (r *Router) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	mux.HandleFunc("GET /v1/cluster", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, http.StatusOK, r.View()) })
	mux.HandleFunc("POST /v1/search", r.handleSearch)
	mux.HandleFunc("POST /v1/upsert", r.authorized(r.handleUpsert))
	mux.HandleFunc("POST /v1/remove", r.authorized(r.handleRemove))
	mux.HandleFunc("POST /v1/chaos/crash", r.authorized(r.handleCrash))
	mux.HandleFunc("GET /metrics", r.metrics.serve)
	return r.cors(mux)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error()})
}

func (r *Router) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if origin := req.Header.Get("Origin"); origin != "" && slices.Contains(r.cfg.AllowedOrigins, origin) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
			w.Header().Set("Access-Control-Max-Age", "600")
		}
		if req.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, req)
	})
}

func (r *Router) authorized(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		token := strings.TrimPrefix(req.Header.Get("Authorization"), "Bearer ")
		if r.cfg.Token == "" || subtle.ConstantTimeCompare([]byte(token), []byte(r.cfg.Token)) != 1 {
			writeError(w, http.StatusUnauthorized, fmt.Errorf("a valid bearer token is required"))
			return
		}
		next(w, req)
	}
}

func (r *Router) handleSearch(w http.ResponseWriter, req *http.Request) {
	var body struct {
		Vector []float32 `json:"vector"`
		K      int       `json:"k"`
		EF     int       `json:"ef"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 1<<20)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	body.K = min(max(body.K, 1), 100)
	body.EF = min(max(body.EF, body.K, 16), 1024)
	result, err := r.Search(req.Context(), body.Vector, body.K, body.EF)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (r *Router) handleUpsert(w http.ResponseWriter, req *http.Request) {
	var body struct {
		Labels  []uint64    `json:"labels"`
		Vectors [][]float32 `json:"vectors"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 256<<20)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if len(body.Labels) != len(body.Vectors) || len(body.Labels) == 0 {
		writeError(w, http.StatusBadRequest, fmt.Errorf("need one vector per label"))
		return
	}
	flat := make([]float32, 0, len(body.Vectors)*r.cfg.Dim)
	for _, v := range body.Vectors {
		flat = append(flat, v...)
	}
	result, err := r.Upsert(req.Context(), body.Labels, flat)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (r *Router) handleRemove(w http.ResponseWriter, req *http.Request) {
	var body struct {
		Label uint64 `json:"label"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 4096)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	result, err := r.Remove(req.Context(), body.Label)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (r *Router) handleCrash(w http.ResponseWriter, req *http.Request) {
	var body struct {
		Node string `json:"node"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 4096)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if err := r.Crash(req.Context(), body.Node); err != nil {
		writeError(w, http.StatusBadGateway, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"crashed": body.Node})
}

// ---------------------------------------------------------------------------------------------
// Metrics in the Prometheus text format, without a client library.

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

func (m *metrics) serve(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# TYPE oryx_searches_total counter\noryx_searches_total %d\n", m.searches.Load())
	fmt.Fprintf(w, "# TYPE oryx_searches_partial_total counter\noryx_searches_partial_total %d\n", m.partial.Load())
	fmt.Fprintf(w, "# TYPE oryx_hedged_requests_total counter\noryx_hedged_requests_total %d\n", m.hedges.Load())
	fmt.Fprintf(w, "# TYPE oryx_failovers_total counter\noryx_failovers_total %d\n", m.failovers.Load())
	fmt.Fprintf(w, "# TYPE oryx_vectors_written_total counter\noryx_vectors_written_total %d\n", m.writes.Load())
	m.mu.Lock()
	defer m.mu.Unlock()
	fmt.Fprintln(w, "# TYPE oryx_search_latency_ms histogram")
	for i, b := range latencyBuckets {
		fmt.Fprintf(w, "oryx_search_latency_ms_bucket{le=\"%g\"} %d\n", b, m.buckets[i])
	}
	fmt.Fprintf(w, "oryx_search_latency_ms_bucket{le=\"+Inf\"} %d\n", m.searches.Load())
	fmt.Fprintf(w, "oryx_search_latency_ms_sum %g\noryx_search_latency_ms_count %d\n", m.sumMs, m.searches.Load())
	fmt.Fprintln(w, "# TYPE oryx_node_up gauge")
	for name, up := range m.up {
		v := 0
		if up {
			v = 1
		}
		fmt.Fprintf(w, "oryx_node_up{node=%q} %d\n", name, v)
	}
}
