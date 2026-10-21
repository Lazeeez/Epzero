//go:build !js

package router

import (
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
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
		// "strong" or "eventual"; empty: the router's default.
		Consistency Consistency `json:"consistency"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 1<<20)).Decode(&body); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	body.K = min(max(body.K, 1), 100)
	body.EF = min(max(body.EF, body.K, 16), 1024)
	if body.Consistency == "" {
		body.Consistency = r.cfg.Consistency
	} else if body.Consistency != Strong && body.Consistency != Eventual {
		writeError(w, http.StatusBadRequest, fmt.Errorf("consistency must be %q or %q", Strong, Eventual))
		return
	}
	result, err := r.SearchWith(req.Context(), body.Vector, body.K, body.EF, body.Consistency)
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
	// Check every vector: a total-length check alone would accept, say, a 3- and a 5-value vector in
	// a 4-dimension index and store both misaligned.
	flat := make([]float32, 0, len(body.Vectors)*r.cfg.Dim)
	for i, v := range body.Vectors {
		if len(v) != r.cfg.Dim {
			writeError(w, http.StatusBadRequest, fmt.Errorf("vector %d has %d dimensions, expected %d", i, len(v), r.cfg.Dim))
			return
		}
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
// Metrics in the Prometheus text format, without a client library (counted in metrics.go).

func (m *metrics) serve(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	fmt.Fprintf(w, "# TYPE epzero_searches_total counter\nepzero_searches_total %d\n", m.searches.Load())
	fmt.Fprintf(w, "# TYPE epzero_searches_partial_total counter\nepzero_searches_partial_total %d\n", m.partial.Load())
	fmt.Fprintf(w, "# TYPE epzero_hedged_requests_total counter\nepzero_hedged_requests_total %d\n", m.hedges.Load())
	fmt.Fprintf(w, "# TYPE epzero_failovers_total counter\nepzero_failovers_total %d\n", m.failovers.Load())
	fmt.Fprintf(w, "# TYPE epzero_vectors_written_total counter\nepzero_vectors_written_total %d\n", m.writes.Load())
	m.mu.Lock()
	defer m.mu.Unlock()
	fmt.Fprintln(w, "# TYPE epzero_search_latency_ms histogram")
	for i, b := range latencyBuckets {
		fmt.Fprintf(w, "epzero_search_latency_ms_bucket{le=\"%g\"} %d\n", b, m.buckets[i])
	}
	fmt.Fprintf(w, "epzero_search_latency_ms_bucket{le=\"+Inf\"} %d\n", m.searches.Load())
	fmt.Fprintf(w, "epzero_search_latency_ms_sum %g\nepzero_search_latency_ms_count %d\n", m.sumMs, m.searches.Load())
	fmt.Fprintln(w, "# TYPE epzero_node_up gauge")
	for name, up := range m.up {
		v := 0
		if up {
			v = 1
		}
		fmt.Fprintf(w, "epzero_node_up{node=%q} %d\n", name, v)
	}
}
