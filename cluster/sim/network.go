// Package sim runs an Epzero cluster in memory: data nodes that replicate like epzero-node
// (engine/server/node_main.cpp), joined by a network whose links can be slowed, made lossy or cut.
// The real router drives them through router.Node, so the same code that serves production
// traffic is what the fault tests check (under testing/synctest, on a virtual clock) and what the
// site runs in the browser, compiled to WebAssembly.
package sim

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sync"
	"time"

	"github.com/Lazeeez/Epzero/cluster/router"
)

// Link describes the path from one endpoint to another. The zero value is a perfect link.
type Link struct {
	Latency time.Duration // one way
	Jitter  time.Duration // up to this much more, uniformly
	Loss    float64       // probability that a message is lost
	Cut     bool          // nothing gets through (a partition)
}

// ErrUnavailable is what calling a crashed node returns: the connection is refused at once, so
// nothing was sent (and a write certainly did not happen).
var ErrUnavailable = fmt.Errorf("%w: unavailable: connection refused", router.ErrRejected)

// Network carries calls between endpoints ("router" and the node names).
type Network struct {
	mu       sync.Mutex
	rng      *rand.Rand
	base     Link
	links    map[[2]string]Link
	messages uint64
	lost     uint64
	sent     map[[2]string]uint64
}

// NewNetwork returns a network whose links all behave like base, with seeded randomness.
func NewNetwork(base Link, seed uint64) *Network {
	return &Network{rng: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)), base: base, links: map[[2]string]Link{}, sent: map[[2]string]uint64{}}
}

// SetLink sets the path from `from` to `to` (one direction).
func (n *Network) SetLink(from, to string, l Link) {
	n.mu.Lock()
	n.links[[2]string{from, to}] = l
	n.mu.Unlock()
}

// SetBoth sets both directions between a and b.
func (n *Network) SetBoth(a, b string, l Link) {
	n.SetLink(a, b, l)
	n.SetLink(b, a, l)
}

// Heal restores every link to the base behavior.
func (n *Network) Heal() {
	n.mu.Lock()
	n.links = map[[2]string]Link{}
	n.mu.Unlock()
}

// Link reports the current path from `from` to `to`.
func (n *Network) Link(from, to string) Link {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.link(from, to)
}

func (n *Network) link(from, to string) Link {
	if l, ok := n.links[[2]string{from, to}]; ok {
		return l
	}
	return n.base
}

// Stats reports messages sent and lost so far.
func (n *Network) Stats() (sent, lost uint64) {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.messages, n.lost
}

// Sent reports how many messages have been sent from `from` to `to` (lost ones included).
func (n *Network) Sent(from, to string) uint64 {
	n.mu.Lock()
	defer n.mu.Unlock()
	return n.sent[[2]string{from, to}]
}

// send moves one message along from -> to: it waits out the link's delay and reports whether the
// message arrived. A lost message simply never arrives; the caller notices by timing out.
func (n *Network) send(ctx context.Context, from, to string) bool {
	n.mu.Lock()
	l := n.link(from, to)
	n.messages++
	n.sent[[2]string{from, to}]++
	lost := l.Cut || (l.Loss > 0 && n.rng.Float64() < l.Loss)
	delay := l.Latency
	if l.Jitter > 0 {
		delay += time.Duration(n.rng.Int64N(int64(l.Jitter)))
	}
	if lost {
		n.lost++
	}
	n.mu.Unlock()
	if lost {
		return false
	}
	if delay > 0 {
		timer := time.NewTimer(delay)
		defer timer.Stop()
		select {
		case <-timer.C:
		case <-ctx.Done():
			return false
		}
	}
	return true
}

// call runs handler on `to` as a remote call from `from`: the request crosses the network, the
// handler runs, and the reply crosses back. If either message is lost the caller waits until its
// context ends, exactly as with a real timeout, even though the handler may already have run.
func (n *Network) call(ctx context.Context, from, to string, target *Node, handler func() error) error {
	if !target.alive() {
		return ErrUnavailable
	}
	if !n.send(ctx, from, to) {
		<-ctx.Done()
		return ctx.Err()
	}
	if !target.alive() {
		<-ctx.Done() // it crashed while the request was on the wire
		return ctx.Err()
	}
	err := handler()
	if !n.send(ctx, to, from) {
		<-ctx.Done()
		return ctx.Err()
	}
	return err
}
