package router

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func TestMemoryStoreCompareAndSwapAndWatch(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m := NewMemoryStore()
	got := make(chan int64, 4)
	go m.Watch(ctx, func(_ []ShardMap, v int64) { got <- v })
	time.Sleep(10 * time.Millisecond) // let the watch register
	v1, err := m.Save(ctx, []ShardMap{{ID: 0, Epoch: 1}}, 0)
	if err != nil || v1 != 1 {
		t.Fatal(v1, err)
	}
	if _, err := m.Save(ctx, []ShardMap{{ID: 0, Epoch: 2}}, 0); !errors.Is(err, ErrConflict) {
		t.Fatal("a stale version was accepted")
	}
	if v := <-got; v != 1 {
		t.Fatal("watch saw", v)
	}
	shards, v, _ := m.Load(ctx)
	if v != 1 || shards[0].Epoch != 1 {
		t.Fatal(shards, v)
	}
}

func TestMemoryStoreHandsTheLeadOver(t *testing.T) {
	ctx := context.Background()
	m := NewMemoryStore()
	ctxA, stopA := context.WithCancel(ctx)
	a := m.Lead(ctxA, "a")
	ctxB, stopB := context.WithCancel(ctx)
	defer stopB()
	b := m.Lead(ctxB, "b")
	if !<-a || m.Leader() != "a" {
		t.Fatal("a should lead")
	}
	select {
	case <-b:
		t.Fatal("b leads too")
	case <-time.After(20 * time.Millisecond):
	}
	stopA()
	if !<-b || m.Leader() != "b" {
		t.Fatal("b should take over")
	}
}

func TestFileStorePersistsAcrossRestarts(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router.json")
	f := NewFileStore(path)
	if shards, v, err := f.Load(ctx); shards != nil || v != 0 || err != nil {
		t.Fatal(shards, v, err)
	}
	if _, err := f.Save(ctx, []ShardMap{{ID: 0, Epoch: 3, Primary: "x", Replica: "y"}}, 0); err != nil {
		t.Fatal(err)
	}
	g := NewFileStore(path)
	shards, v, err := g.Load(ctx)
	if err != nil || v == 0 || shards[0].Epoch != 3 || shards[0].Primary != "x" {
		t.Fatal(shards, v, err)
	}
	if _, err := g.Save(ctx, shards, v); err != nil {
		t.Fatal(err)
	}
}
