//go:build !js

package router

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"go.uber.org/zap"
)

// EtcdStore keeps the shard map in etcd so several routers can run at once: they all serve
// traffic from the map (watched for changes), and an election picks the one that changes it.
//
// Leadership is fenced: Save succeeds only while this router still holds the leader key, checked
// in the same transaction as the version. A leader cut off from etcd loses its session lease after
// TTL; from then on its saves fail even before it notices, so two routers never both hand out
// epochs.
type EtcdStore struct {
	client *clientv3.Client
	key    string // the shard map
	elect  string // the election prefix
	ttl    int    // session lease, seconds

	mu      sync.Mutex
	leadKey string // our key under the election prefix while we lead ("" otherwise)
	leadRev int64
}

// ErrNotLeader: a save was attempted by a router that does not (or no longer) lead.
var ErrNotLeader = errors.New("not the leader")

// NewEtcdStore keeps the map of cluster `name` under /epzero/<name>/ in etcd. ttl is how long a
// leader may be silent before another router takes over (seconds, at least 1).
func NewEtcdStore(client *clientv3.Client, name string, ttl int) *EtcdStore {
	if ttl < 1 {
		ttl = 5
	}
	prefix := "/epzero/" + name
	return &EtcdStore{client: client, key: prefix + "/shards", elect: prefix + "/leader", ttl: ttl}
}

// DialEtcd connects to etcd at the given endpoints. The client's own retry logging is off: the
// router reports what matters (leadership changes, failed saves) through its own logger.
func DialEtcd(endpoints []string) (*clientv3.Client, error) {
	return clientv3.New(clientv3.Config{Endpoints: endpoints, DialTimeout: 5 * time.Second, Logger: zap.NewNop()})
}

func (e *EtcdStore) Load(ctx context.Context) ([]ShardMap, int64, error) {
	resp, err := e.client.Get(ctx, e.key)
	if err != nil {
		return nil, 0, err
	}
	if len(resp.Kvs) == 0 {
		return nil, 0, nil
	}
	var shards []ShardMap
	if err := json.Unmarshal(resp.Kvs[0].Value, &shards); err != nil {
		return nil, 0, fmt.Errorf("etcd %s: %w", e.key, err)
	}
	return shards, resp.Kvs[0].ModRevision, nil
}

func (e *EtcdStore) Save(ctx context.Context, shards []ShardMap, version int64) (int64, error) {
	data, err := json.Marshal(shards)
	if err != nil {
		return 0, err
	}
	conds := []clientv3.Cmp{clientv3.Compare(clientv3.ModRevision(e.key), "=", version)}
	e.mu.Lock()
	leadKey, leadRev := e.leadKey, e.leadRev
	e.mu.Unlock()
	// Creating the first map needs no leader (every router may try; one wins). Changing it does.
	if version != 0 {
		if leadKey == "" {
			return 0, ErrNotLeader
		}
		conds = append(conds, clientv3.Compare(clientv3.CreateRevision(leadKey), "=", leadRev))
	}
	resp, err := e.client.Txn(ctx).If(conds...).Then(clientv3.OpPut(e.key, string(data))).Commit()
	if err != nil {
		return 0, err
	}
	if !resp.Succeeded {
		if version != 0 {
			if lead, err := e.client.Get(ctx, leadKey); err == nil && (len(lead.Kvs) == 0 || lead.Kvs[0].CreateRevision != leadRev) {
				return 0, ErrNotLeader
			}
		}
		return 0, ErrConflict
	}
	return resp.Header.Revision, nil
}

func (e *EtcdStore) Watch(ctx context.Context, update func([]ShardMap, int64)) {
	for ctx.Err() == nil {
		for resp := range e.client.Watch(ctx, e.key) {
			for _, ev := range resp.Events {
				if ev.Type != clientv3.EventTypePut {
					continue
				}
				var shards []ShardMap
				if json.Unmarshal(ev.Kv.Value, &shards) == nil {
					update(shards, ev.Kv.ModRevision)
				}
			}
		}
		// The watch channel closes on errors (e.g. etcd unreachable): catch up and watch again.
		select {
		case <-ctx.Done():
			return
		case <-time.After(time.Second):
		}
		if shards, version, err := e.Load(ctx); err == nil && shards != nil {
			update(shards, version)
		}
	}
}

func (e *EtcdStore) Lead(ctx context.Context, name string) <-chan bool {
	ch := make(chan bool, 4)
	go func() {
		defer close(ch)
		for ctx.Err() == nil {
			session, err := concurrency.NewSession(e.client, concurrency.WithTTL(e.ttl), concurrency.WithContext(ctx))
			if err != nil {
				sleepCtx(ctx, time.Second)
				continue
			}
			election := concurrency.NewElection(session, e.elect)
			if err := election.Campaign(ctx, name); err != nil {
				session.Close()
				sleepCtx(ctx, time.Second)
				continue
			}
			e.mu.Lock()
			e.leadKey, e.leadRev = election.Key(), election.Rev()
			e.mu.Unlock()
			ch <- true
			select {
			case <-ctx.Done():
			case <-session.Done(): // the lease expired: someone else may lead now
			}
			e.mu.Lock()
			e.leadKey, e.leadRev = "", 0
			e.mu.Unlock()
			if ctx.Err() == nil {
				ch <- false
			}
			resign, cancel := context.WithTimeout(context.Background(), time.Second)
			election.Resign(resign)
			cancel()
			session.Close()
		}
	}()
	return ch
}

func sleepCtx(ctx context.Context, d time.Duration) {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C:
	}
}
