package router

import (
	"context"
	"errors"
)

// ErrRejected marks an error after which the node certainly did not apply a write: it refused it
// (stale epoch, not the primary) or never received it. Only such writes are safe to retry. After
// any other error (a timeout, a lost reply) the write may have been applied, and sending it again
// could apply it a second time, after writes that came later.
var ErrRejected = errors.New("rejected")

// Node is how the router talks to one data node. Over the network it is a gRPC client
// (dial_grpc.go); package sim implements it in memory, so the same router runs against simulated
// nodes in deterministic fault tests and, compiled to WebAssembly, in the browser.
type Node interface {
	Status(ctx context.Context) (*NodeStatus, error)
	Assign(ctx context.Context, a Assignment) error
	Search(ctx context.Context, shard int, query []float32, k, ef int) (*ShardReply, error)
	// Upsert and Remove go to a shard's primary, fenced by the shard's epoch. replicated is false
	// when only the primary acknowledged (its replica is down or catching up).
	Upsert(ctx context.Context, shard int, epoch uint64, labels []uint64, vectors []float32) (replicated bool, err error)
	Remove(ctx context.Context, shard int, epoch uint64, label uint64) (replicated bool, err error)
	Crash(ctx context.Context, token string) error
}

// Role is a node's part in serving one shard.
type Role int

const (
	RoleNone Role = iota // not assigned (e.g. just restarted)
	RolePrimary
	RoleReplica
)

// MarshalText writes the role by name in JSON.
func (r Role) MarshalText() ([]byte, error) { return []byte(r.String()), nil }

func (r Role) String() string {
	switch r {
	case RolePrimary:
		return "primary"
	case RoleReplica:
		return "replica"
	}
	return "none"
}

// Assignment tells a node its role for a shard. Peer is the other copy's address: a primary's
// replica ("" for none), or a replica's primary.
type Assignment struct {
	Shard int
	Epoch uint64
	Role  Role
	Peer  string
}

// NodeStatus is a node's answer to a health check.
type NodeStatus struct {
	StartedUnixMs uint64
	SIMD          string
	Shards        []ShardStatus
}

// ShardStatus is one shard copy on a node. InSync: for a primary, its replica is receiving
// writes; for a replica, it has caught up with its primary.
type ShardStatus struct {
	Shard   int    `json:"shard"`
	Role    Role   `json:"role"`
	Epoch   uint64 `json:"epoch"`
	LSN     uint64 `json:"lsn"`
	Vectors uint64 `json:"vectors"`
	InSync  bool   `json:"in_sync"`
}

// ShardReply is one shard copy's answer to a search.
type ShardReply struct {
	Hits                 []NodeHit
	Micros               uint64
	DistanceComputations uint64
}

// NodeHit is one result from a shard.
type NodeHit struct {
	Label    uint64
	Distance float32
}
