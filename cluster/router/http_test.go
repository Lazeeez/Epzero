package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The HTTP layer's validation and auth, without any data nodes (the router connects lazily).
func newTestRouter(t *testing.T) http.Handler {
	t.Helper()
	r, err := New(Config{
		Nodes: []NodeConfig{{Name: "a", Addr: "127.0.0.1:1"}, {Name: "b", Addr: "127.0.0.1:2"}},
		Dim:   4,
		Token: "secret",
	})
	if err != nil {
		t.Fatal(err)
	}
	return r.Handler()
}

func post(h http.Handler, path, body, token string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestWritesNeedTheToken(t *testing.T) {
	h := newTestRouter(t)
	for _, path := range []string{"/v1/upsert", "/v1/remove", "/v1/chaos/crash"} {
		for _, token := range []string{"", "wrong", "secre", "secret2"} {
			if rec := post(h, path, `{}`, token); rec.Code != http.StatusUnauthorized {
				t.Errorf("%s with token %q: got %d, want 401", path, token, rec.Code)
			}
		}
	}
}

func TestUpsertRejectsMisshapenVectors(t *testing.T) {
	h := newTestRouter(t)
	cases := map[string]string{
		// 3 + 5 values = 8 = 2 x 4: the total is right, each vector is wrong.
		"mismatched lengths": `{"labels":[1,2],"vectors":[[1,2,3],[1,2,3,4,5]]}`,
		"wrong dimension":    `{"labels":[1],"vectors":[[1,2,3]]}`,
		"count mismatch":     `{"labels":[1,2],"vectors":[[1,2,3,4]]}`,
		"empty":              `{"labels":[],"vectors":[]}`,
		"not json":           `{"labels":`,
	}
	for name, body := range cases {
		if rec := post(h, "/v1/upsert", body, "secret"); rec.Code != http.StatusBadRequest {
			t.Errorf("%s: got %d, want 400 (%s)", name, rec.Code, rec.Body.String())
		}
	}
}

func TestSearchRejectsWrongDimensionAndOversizedBodies(t *testing.T) {
	h := newTestRouter(t)
	if rec := post(h, "/v1/search", `{"vector":[1,2,3],"k":10}`, ""); rec.Code == http.StatusOK {
		t.Errorf("wrong dimension accepted")
	}
	huge := `{"vector":[` + strings.Repeat("1,", 1<<20) + `1]}` // > the 1 MiB body limit
	if rec := post(h, "/v1/search", huge, ""); rec.Code != http.StatusBadRequest {
		t.Errorf("oversized body: got %d, want 400", rec.Code)
	}
}
