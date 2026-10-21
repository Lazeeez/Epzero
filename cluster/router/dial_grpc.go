//go:build !js

package router

import (
	"context"
	"fmt"

	pb "github.com/Lazeeez/Oryx/cluster/gen/oryxv1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// DialGRPC connects (lazily) to a data node over gRPC. It is the default Config.Dial.
func DialGRPC(nc NodeConfig) (Node, error) {
	conn, err := grpc.NewClient(nc.Addr, grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(64<<20), grpc.MaxCallSendMsgSize(256<<20)))
	if err != nil {
		return nil, err
	}
	return grpcNode{pb.NewNodeClient(conn)}, nil
}

var defaultDial = DialGRPC

type grpcNode struct{ client pb.NodeClient }

func (g grpcNode) Status(ctx context.Context) (*NodeStatus, error) {
	reply, err := g.client.Status(ctx, &pb.StatusRequest{})
	if err != nil {
		return nil, err
	}
	status := &NodeStatus{StartedUnixMs: reply.StartedUnixMs, SIMD: reply.Simd}
	for _, s := range reply.Shards {
		status.Shards = append(status.Shards, ShardStatus{
			Shard: int(s.Shard), Role: Role(s.Role), Epoch: s.Epoch, LSN: s.Lsn, Vectors: s.Vectors, InSync: s.InSync,
		})
	}
	return status, nil
}

func (g grpcNode) Assign(ctx context.Context, a Assignment) error {
	_, err := g.client.Assign(ctx, &pb.AssignRequest{Shard: uint32(a.Shard), Epoch: a.Epoch, Role: pb.Role(a.Role), Peer: a.Peer})
	return err
}

func (g grpcNode) Search(ctx context.Context, shard int, query []float32, k, ef int) (*ShardReply, error) {
	reply, err := g.client.Search(ctx, &pb.SearchRequest{Shard: uint32(shard), Query: query, K: uint32(k), Ef: uint32(ef)})
	if err != nil {
		return nil, err
	}
	out := &ShardReply{Micros: reply.Micros, DistanceComputations: reply.DistanceComputations, Hits: make([]NodeHit, len(reply.Hits))}
	for i, h := range reply.Hits {
		out.Hits[i] = NodeHit{Label: h.Label, Distance: h.Distance}
	}
	return out, nil
}

// rpcWriteError marks the node's refusals, after which the write certainly did not happen.
func rpcWriteError(err error) error {
	switch status.Code(err) {
	case codes.FailedPrecondition, codes.NotFound, codes.InvalidArgument, codes.PermissionDenied:
		return fmt.Errorf("%w: %w", ErrRejected, err)
	}
	return err
}

func (g grpcNode) Upsert(ctx context.Context, shard int, epoch uint64, labels []uint64, vectors []float32) (bool, error) {
	reply, err := g.client.Upsert(ctx, &pb.UpsertRequest{Shard: uint32(shard), Epoch: epoch, Labels: labels, Vectors: vectors})
	if err != nil {
		return false, rpcWriteError(err)
	}
	return reply.Replicated, nil
}

func (g grpcNode) Remove(ctx context.Context, shard int, epoch uint64, label uint64) (bool, error) {
	reply, err := g.client.Remove(ctx, &pb.RemoveRequest{Shard: uint32(shard), Epoch: epoch, Label: label})
	if err != nil {
		return false, rpcWriteError(err)
	}
	return reply.Replicated, nil
}

func (g grpcNode) Crash(ctx context.Context, token string) error {
	_, err := g.client.Crash(ctx, &pb.CrashRequest{Token: token})
	return err
}
