// oryx-router: the Oryx cluster's coordinator and public HTTP API.
//
//	oryx-router -listen :8080 -nodes node1=10.42.1.11:7000,node2=10.42.1.12:7000,node3=10.42.1.13:7000 \
//	            -dim 512 -token-file /etc/oryx/token -state /var/lib/oryx/router.json \
//	            -origins https://lazeeez.github.io
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/Lazeeez/Oryx/cluster/router"
)

func main() {
	listen := flag.String("listen", ":8080", "HTTP listen address")
	nodes := flag.String("nodes", "", "comma-separated name=host:port list of data nodes")
	shards := flag.Int("shards", 0, "number of shards (default: one per node)")
	dim := flag.Int("dim", 512, "vector dimension")
	tokenFile := flag.String("token-file", "", "file holding the bearer token for writes and chaos")
	state := flag.String("state", "router.json", "where the shard map is persisted")
	origins := flag.String("origins", "", "comma-separated CORS origins allowed to call the API")
	hedge := flag.Duration("hedge", 40*time.Millisecond, "query the other copy of a shard if the first is this slow")
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stderr, nil))
	cfg := router.Config{Shards: *shards, Dim: *dim, StatePath: *state, HedgeAfter: *hedge, Logger: log}
	for _, entry := range strings.Split(*nodes, ",") {
		name, addr, ok := strings.Cut(strings.TrimSpace(entry), "=")
		if !ok {
			log.Error("bad -nodes entry, want name=host:port", "entry", entry)
			os.Exit(2)
		}
		cfg.Nodes = append(cfg.Nodes, router.NodeConfig{Name: name, Addr: addr})
	}
	if *origins != "" {
		cfg.AllowedOrigins = strings.Split(*origins, ",")
	}
	if *tokenFile != "" {
		data, err := os.ReadFile(*tokenFile)
		if err != nil {
			log.Error("reading token", "err", err)
			os.Exit(2)
		}
		cfg.Token = strings.TrimSpace(string(data))
	}

	r, err := router.New(cfg)
	if err != nil {
		log.Error("starting router", "err", err)
		os.Exit(1)
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	r.Bootstrap(ctx)
	go r.Run(ctx)

	// Bounded timeouts so slow or idle clients can't hold connections open indefinitely; reads allow
	// time for the largest upsert body (256 MiB).
	server := &http.Server{
		Addr:              *listen,
		Handler:           r.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       2 * time.Minute,
		WriteTimeout:      2 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		server.Shutdown(shutdown)
	}()
	log.Info("router listening", "addr", *listen, "nodes", len(cfg.Nodes))
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Error("http server", "err", err)
		os.Exit(1)
	}
}
