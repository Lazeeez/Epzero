package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// StateStore holds the shard map and decides which router may change it. With one router a file
// (FileStore) is enough; several routers share an etcd cluster (EtcdStore): one of them, the
// leader, health-checks the nodes and fails shards over; every router serves searches and writes
// from the map, which the store pushes to all of them.
type StateStore interface {
	// Load returns the shard map and its version (nil and 0 if there is none yet).
	Load(ctx context.Context) ([]ShardMap, int64, error)
	// Save replaces the map if its version is still `version` (0: no map yet), returning the new
	// version, or ErrConflict if another router changed it first. Only a leader saves.
	Save(ctx context.Context, shards []ShardMap, version int64) (int64, error)
	// Watch calls update with every newer map until ctx ends.
	Watch(ctx context.Context, update func(shards []ShardMap, version int64))
	// Lead campaigns for leadership and reports changes on the returned channel: true once this
	// router leads, false if it loses the lead (e.g. its etcd session expired). Closed when ctx ends.
	Lead(ctx context.Context, name string) <-chan bool
}

// ErrConflict: the shard map changed since it was read.
var ErrConflict = errors.New("shard map changed concurrently")

// ---------------------------------------------------------------------------------------------
// FileStore: one router, the map in a local JSON file (written atomically). Always the leader.

type FileStore struct {
	path    string
	mu      sync.Mutex
	version int64
}

// NewFileStore keeps the map at path ("" keeps it in memory only).
func NewFileStore(path string) *FileStore { return &FileStore{path: path} }

func (f *FileStore) Load(context.Context) ([]ShardMap, int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.path == "" {
		return nil, 0, nil
	}
	data, err := os.ReadFile(f.path)
	if os.IsNotExist(err) {
		return nil, 0, nil
	}
	if err != nil {
		return nil, 0, err
	}
	var saved struct {
		Shards []ShardMap `json:"shards"`
	}
	if err := json.Unmarshal(data, &saved); err != nil {
		return nil, 0, fmt.Errorf("%s: %w", f.path, err)
	}
	f.version = 1
	return saved.Shards, f.version, nil
}

func (f *FileStore) Save(_ context.Context, shards []ShardMap, version int64) (int64, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if version != f.version {
		return 0, ErrConflict
	}
	if f.path != "" {
		if err := writeAtomically(f.path, shards); err != nil {
			return 0, err
		}
	}
	f.version++
	return f.version, nil
}

func (f *FileStore) Watch(ctx context.Context, _ func([]ShardMap, int64)) { <-ctx.Done() } // no other writers

func (f *FileStore) Lead(ctx context.Context, _ string) <-chan bool {
	ch := make(chan bool, 1)
	ch <- true
	go func() {
		<-ctx.Done()
		close(ch)
	}()
	return ch
}

// writeAtomically: write a temporary file, fsync it, rename it over the old one, fsync the directory.
func writeAtomically(path string, shards []ShardMap) error {
	data, _ := json.MarshalIndent(struct {
		Shards []ShardMap `json:"shards"`
	}{shards}, "", "  ")
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return err
	}
	f.Close()
	if err := os.Rename(tmp, path); err != nil {
		return err
	}
	if dir, err := os.Open(filepath.Dir(path)); err == nil {
		dir.Sync()
		dir.Close()
	}
	return nil
}

// ---------------------------------------------------------------------------------------------
// MemoryStore: several routers in one process (the simulator and its tests) sharing a map, with
// the same semantics as EtcdStore: versioned compare-and-swap, watches, and one leader at a time,
// handed to the next campaigner when the leader's context ends (or Resign is called).

type MemoryStore struct {
	mu       sync.Mutex
	shards   []ShardMap
	version  int64
	watchers map[int]func([]ShardMap, int64)
	nextID   int
	leader   string
	queue    []*campaign
}

type campaign struct {
	name string
	ch   chan bool
	done <-chan struct{}
}

func NewMemoryStore() *MemoryStore { return &MemoryStore{watchers: map[int]func([]ShardMap, int64){}} }

func (m *MemoryStore) Load(context.Context) ([]ShardMap, int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]ShardMap(nil), m.shards...), m.version, nil
}

func (m *MemoryStore) Save(_ context.Context, shards []ShardMap, version int64) (int64, error) {
	m.mu.Lock()
	if version != m.version {
		m.mu.Unlock()
		return 0, ErrConflict
	}
	m.shards = append([]ShardMap(nil), shards...)
	m.version++
	v := m.version
	watchers := make([]func([]ShardMap, int64), 0, len(m.watchers))
	for _, w := range m.watchers {
		watchers = append(watchers, w)
	}
	m.mu.Unlock()
	for _, w := range watchers {
		w(append([]ShardMap(nil), shards...), v)
	}
	return v, nil
}

func (m *MemoryStore) Watch(ctx context.Context, update func([]ShardMap, int64)) {
	m.mu.Lock()
	id := m.nextID
	m.nextID++
	m.watchers[id] = update
	m.mu.Unlock()
	<-ctx.Done()
	m.mu.Lock()
	delete(m.watchers, id)
	m.mu.Unlock()
}

func (m *MemoryStore) Lead(ctx context.Context, name string) <-chan bool {
	c := &campaign{name: name, ch: make(chan bool, 2), done: ctx.Done()}
	m.mu.Lock()
	m.queue = append(m.queue, c)
	m.promoteLocked()
	m.mu.Unlock()
	go func() {
		<-ctx.Done()
		m.mu.Lock()
		for i, q := range m.queue {
			if q == c {
				m.queue = append(m.queue[:i], m.queue[i+1:]...)
				break
			}
		}
		if m.leader == name {
			m.leader = ""
			m.promoteLocked()
		}
		m.mu.Unlock()
		close(c.ch)
	}()
	return c.ch
}

// Leader names the current leader ("" if none).
func (m *MemoryStore) Leader() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.leader
}

func (m *MemoryStore) promoteLocked() {
	if m.leader != "" || len(m.queue) == 0 {
		return
	}
	next := m.queue[0]
	m.leader = next.name
	next.ch <- true
}
