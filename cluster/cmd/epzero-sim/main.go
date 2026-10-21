//go:build js && wasm

// Command epzero-sim runs an Epzero cluster in the browser: the real router (package router) against
// simulated data nodes on a simulated network (package sim). The site's cluster page loads it in a
// Web Worker and drives it through the functions it installs on globalThis.epzeroSim:
//
//	start(options JSON)  builds the cluster, loads the dataset and starts the workload
//	fault(command JSON)  kills, restarts, partitions, slows or heals
//	rates(searches, keyOps)
//	consistency("strong" | "eventual")
//	snapshot() JSON      nodes, shard map, links, events and per-second stats
//	check() JSON         Porcupine linearizability check of the recorded history
//
//	GOOS=js GOARCH=wasm go build -ldflags="-s -w" -o epzero-sim.wasm ./cmd/epzero-sim
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"syscall/js"
	"time"

	"github.com/Lazeeez/Epzero/cluster/router"
	"github.com/Lazeeez/Epzero/cluster/sim"
)

type options struct {
	Nodes   int    `json:"nodes"`
	Vectors int    `json:"vectors"`
	Dim     int    `json:"dim"`
	Seed    uint64 `json:"seed"`
	Latency int    `json:"latency_ms"`
}

type fault struct {
	Kind string   `json:"kind"` // kill, restart, isolate, split, slow, lossy, heal
	Node string   `json:"node"`
	Peer string   `json:"peer"`
	Ms   int      `json:"ms"`
	Loss float64  `json:"loss"`
	Side []string `json:"side"`
}

type nodeView struct {
	Name   string               `json:"name"`
	Up     bool                 `json:"up"`
	SlowMs int                  `json:"slow_ms"`
	Shards []router.ShardStatus `json:"shards"`
}

type linkView struct {
	From    string  `json:"from"`
	To      string  `json:"to"`
	Cut     bool    `json:"cut"`
	Loss    float64 `json:"loss"`
	Latency float64 `json:"latency_ms"`
	Sent    uint64  `json:"sent"`
}

type snapshot struct {
	Ready   bool               `json:"ready"`
	Nodes   []nodeView         `json:"nodes"`
	Links   []linkView         `json:"links"`
	Cluster router.ClusterView `json:"cluster"`
	Seconds []sim.Second       `json:"seconds"`
}

var (
	cluster  *sim.Cluster
	workload *sim.Workload
	ready    bool
	slow     = map[string]int{}
)

func main() {
	api := js.Global().Get("Object").New()
	api.Set("start", js.FuncOf(func(_ js.Value, args []js.Value) any {
		var o options
		if err := json.Unmarshal([]byte(args[0].String()), &o); err != nil {
			return err.Error()
		}
		return promise(func() (any, error) { return nil, start(o) })
	}))
	api.Set("fault", js.FuncOf(func(_ js.Value, args []js.Value) any {
		var f fault
		if err := json.Unmarshal([]byte(args[0].String()), &f); err != nil {
			return err.Error()
		}
		if err := apply(f); err != nil {
			return err.Error()
		}
		return ""
	}))
	api.Set("rates", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if workload != nil {
			workload.SetRates(args[0].Float(), args[1].Float())
		}
		return nil
	}))
	api.Set("consistency", js.FuncOf(func(_ js.Value, args []js.Value) any {
		if workload != nil {
			workload.SetConsistency(router.Consistency(args[0].String()))
		}
		return nil
	}))
	api.Set("snapshot", js.FuncOf(func(js.Value, []js.Value) any { return view() }))
	api.Set("check", js.FuncOf(func(js.Value, []js.Value) any {
		return promise(func() (any, error) {
			if workload == nil {
				return nil, fmt.Errorf("not started")
			}
			out, _ := json.Marshal(workload.Check(20 * time.Second))
			return string(out), nil
		})
	}))
	js.Global().Set("epzeroSim", api)
	select {}
}

// promise runs f off the JavaScript thread's call stack and resolves with its result.
func promise(f func() (any, error)) js.Value {
	handler := js.FuncOf(func(_ js.Value, args []js.Value) any {
		resolve, reject := args[0], args[1]
		go func() {
			v, err := f()
			if err != nil {
				reject.Invoke(err.Error())
				return
			}
			resolve.Invoke(v)
		}()
		return nil
	})
	return js.Global().Get("Promise").New(handler)
}

func start(o options) error {
	if cluster != nil {
		return fmt.Errorf("already started")
	}
	names := make([]string, max(2, o.Nodes))
	for i := range names {
		names[i] = fmt.Sprintf("node-%d", i+1)
	}
	c, err := sim.New(sim.Options{
		Nodes: names, Dim: o.Dim, Seed: o.Seed,
		Link:           sim.Link{Latency: time.Duration(o.Latency) * time.Millisecond, Jitter: time.Duration(o.Latency) * time.Millisecond},
		HealthInterval: 300 * time.Millisecond,
		HedgeAfter:     25 * time.Millisecond,
	})
	if err != nil {
		return err
	}
	ctx := context.Background()
	cluster = c
	c.Start(ctx)
	time.Sleep(time.Second) // replicas catch up before the load
	workload = sim.NewWorkload(c, o.Vectors, o.Dim, 400, o.Seed)
	if err := workload.Load(ctx); err != nil {
		return err
	}
	go workload.Run(ctx, o.Seed)
	ready = true
	return nil
}

func apply(f fault) error {
	if cluster == nil {
		return fmt.Errorf("not started")
	}
	n := cluster.Nodes[f.Node]
	if n == nil && f.Kind != "heal" && f.Kind != "split" {
		return fmt.Errorf("unknown node %q", f.Node)
	}
	switch f.Kind {
	case "kill":
		n.Kill()
	case "restart":
		n.Restart()
	case "isolate": // cut off from the router and every other node
		others := []string{sim.RouterName}
		for _, m := range cluster.Names {
			if m != f.Node {
				others = append(others, m)
			}
		}
		cluster.Partition([]string{f.Node}, others)
	case "split": // the router reaches everyone, but the nodes in Side can't reach the others
		var rest []string
		in := map[string]bool{}
		for _, s := range f.Side {
			in[s] = true
		}
		for _, m := range cluster.Names {
			if !in[m] {
				rest = append(rest, m)
			}
		}
		cluster.Partition(f.Side, rest)
	case "slow":
		n.SetSlow(time.Duration(f.Ms) * time.Millisecond)
		slow[f.Node] = f.Ms
	case "lossy":
		base := cluster.Net.Link(sim.RouterName, f.Node)
		base.Loss = f.Loss
		cluster.Net.SetBoth(sim.RouterName, f.Node, base)
	case "heal":
		cluster.Net.Heal()
		for name, node := range cluster.Nodes {
			node.SetSlow(0)
			delete(slow, name)
			node.Restart()
		}
	default:
		return fmt.Errorf("unknown fault %q", f.Kind)
	}
	return nil
}

func view() string {
	s := snapshot{Ready: ready}
	if cluster != nil {
		for _, name := range cluster.Names {
			n := cluster.Nodes[name]
			s.Nodes = append(s.Nodes, nodeView{Name: name, Up: n.Up(), SlowMs: slow[name], Shards: n.Shards()})
		}
		ends := append([]string{sim.RouterName}, cluster.Names...)
		for _, a := range ends {
			for _, b := range ends {
				if a == b {
					continue
				}
				l := cluster.Net.Link(a, b)
				s.Links = append(s.Links, linkView{From: a, To: b, Cut: l.Cut, Loss: l.Loss,
					Latency: float64(l.Latency.Microseconds()) / 1000, Sent: cluster.Net.Sent(a, b)})
			}
		}
		s.Cluster = cluster.Router.View()
	}
	if workload != nil {
		s.Seconds = workload.Seconds()
	}
	out, _ := json.Marshal(s)
	return string(out)
}
