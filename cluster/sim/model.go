package sim

import (
	"context"
	"fmt"
	"time"

	"github.com/anishathalye/porcupine"
)

// The history the linearizability checks run on: single keys holding a version number. Write v
// stores a vector that encodes v; a read finds the key's vector by search (or finds it missing).

type kvInput struct {
	Key     int
	Op      string // "write", "remove", "read"
	Version int
}

type kvOutput struct {
	Version int    // read: the version seen (0: absent)
	From    string // read: the copy that answered ("primary" or "replica"), for diagnosis
}

// unknown is a key's state before the history says anything about it.
const unknown = -1

// kvModel checks histories that start with every key at version `initial`, or at unknown
// (accepting whatever the first read sees), for a history whose beginning was dropped.
func kvModel(initial int) porcupine.Model {
	return porcupine.Model{
		Partition: func(history []porcupine.Operation) [][]porcupine.Operation {
			byKey := map[int][]porcupine.Operation{}
			var keys []int
			for _, op := range history {
				k := op.Input.(kvInput).Key
				if _, ok := byKey[k]; !ok {
					keys = append(keys, k)
				}
				byKey[k] = append(byKey[k], op)
			}
			out := make([][]porcupine.Operation, 0, len(keys))
			for _, k := range keys {
				out = append(out, byKey[k])
			}
			return out
		},
		Init: func() any { return initial },
		Step: func(state, input, output any) (bool, any) {
			in := input.(kvInput)
			switch in.Op {
			case "write":
				return true, in.Version
			case "remove":
				return true, 0
			default:
				seen := output.(kvOutput).Version
				if state.(int) == unknown {
					return true, seen
				}
				return seen == state.(int), state
			}
		},
		DescribeOperation: func(input, output any) string {
			in := input.(kvInput)
			if in.Op == "read" {
				out := output.(kvOutput)
				return fmt.Sprintf("read(%d) -> %d from %s", in.Key, out.Version, out.From)
			}
			return fmt.Sprintf("%s(%d, v%d)", in.Op, in.Key, in.Version)
		},
	}
}

func sleep(ctx context.Context, d time.Duration) {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
	case <-t.C:
	}
}
