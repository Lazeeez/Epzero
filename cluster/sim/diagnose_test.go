package sim

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/anishathalye/porcupine"
)

// staleReads finds the plainest kind of violation: a read that began after a write to its key had
// definitely completed (acknowledged as replicated), yet saw an older value, with no other write
// to that key overlapping the read. It reports how many there were and which copy served them.
func staleReads(ops []porcupine.Operation) string {
	byKey := map[int][]porcupine.Operation{}
	for _, op := range ops {
		k := op.Input.(kvInput).Key
		byKey[k] = append(byKey[k], op)
	}
	byRole := map[string]int{}
	example := ""
	for _, kops := range byKey {
		sort.Slice(kops, func(a, b int) bool { return kops[a].Call < kops[b].Call })
		for _, r := range kops {
			in := r.Input.(kvInput)
			if in.Op != "read" {
				continue
			}
			// The last write that completed before this read began, and whether any write overlaps it.
			var last *porcupine.Operation
			overlap := false
			for i := range kops {
				w := &kops[i]
				if w.Input.(kvInput).Op == "read" {
					continue
				}
				if w.Return < r.Call {
					if last == nil || w.Return > last.Return {
						last = w
					}
				} else if w.Call < r.Return {
					overlap = true
				}
			}
			if last == nil || overlap || last.Return == math.MaxInt64 {
				continue
			}
			want := 0
			if last.Input.(kvInput).Op == "write" {
				want = last.Input.(kvInput).Version
			}
			got := r.Output.(kvOutput)
			if got.Version != want {
				byRole[got.From]++
				if example == "" {
					example = fmt.Sprintf("key %d: %s completed, then a read from the %s saw v%d", in.Key,
						kvModel(1).DescribeOperation(last.Input, last.Output), got.From, got.Version)
				}
			}
		}
	}
	return fmt.Sprintf("%v; e.g. %s", byRole, example)
}

// vanished finds reads that saw a value which a later read no longer saw, with no write to the key
// in between that could explain it: a write was visible and then lost.
func vanished(ops []porcupine.Operation) string {
	byKey := map[int][]porcupine.Operation{}
	for _, op := range ops {
		k := op.Input.(kvInput).Key
		byKey[k] = append(byKey[k], op)
	}
	count := 0
	example := ""
	for key, kops := range byKey {
		var reads []porcupine.Operation
		for _, op := range kops {
			if op.Input.(kvInput).Op == "read" {
				reads = append(reads, op)
			}
		}
		sort.Slice(reads, func(a, b int) bool { return reads[a].Call < reads[b].Call })
		for i, a := range reads {
			for _, b := range reads[i+1:] {
				if b.Call <= a.Return || b.Output.(kvOutput).Version == a.Output.(kvOutput).Version {
					continue
				}
				// Any write that could have run between the end of a and the end of b explains it.
				explained := false
				for _, w := range kops {
					if w.Input.(kvInput).Op != "read" && w.Return > a.Call && w.Call < b.Return {
						explained = true
						break
					}
				}
				if !explained {
					count++
					if example == "" {
						example = fmt.Sprintf("key %d: read from the %s saw v%d, a later read from the %s saw v%d", key,
							a.Output.(kvOutput).From, a.Output.(kvOutput).Version, b.Output.(kvOutput).From, b.Output.(kvOutput).Version)
					}
				}
				break
			}
		}
	}
	return fmt.Sprintf("%d; e.g. %s", count, example)
}

// failingKey checks each key on its own and returns the first non-linearizable key's history,
// ordered by call, one operation per line ("ambiguous" for writes that may or may not have run).
func failingKey(ops []porcupine.Operation) string {
	byKey := map[int][]porcupine.Operation{}
	for _, op := range ops {
		k := op.Input.(kvInput).Key
		byKey[k] = append(byKey[k], op)
	}
	for key := range len(byKey) {
		kops := byKey[key]
		if porcupine.CheckOperationsTimeout(kvModel(1), kops, 10*time.Second) != porcupine.Illegal {
			continue
		}
		sort.Slice(kops, func(a, b int) bool { return kops[a].Call < kops[b].Call })
		var b strings.Builder
		fmt.Fprintf(&b, "key %d (%d operations):\n", key, len(kops))
		// Where the checker got stuck: the end of its longest valid ordering, and the operation it
		// could not place next (the earliest-returning one left over).
		_, info := porcupine.CheckOperationsVerbose(kvModel(1), kops, 10*time.Second)
		var longest []porcupine.Operation
		for _, partial := range info.PartialLinearizationsOperations()[0] {
			if len(partial) > len(longest) {
				longest = partial
			}
		}
		placed := map[[2]int64]bool{}
		for _, op := range longest {
			placed[[2]int64{op.Call, op.Return}] = true
		}
		var stuck *porcupine.Operation
		for i := range kops {
			if !placed[[2]int64{kops[i].Call, kops[i].Return}] && (stuck == nil || kops[i].Return < stuck.Return) {
				stuck = &kops[i]
			}
		}
		fmt.Fprintf(&b, "  valid ordering ends with:\n")
		for _, op := range longest[max(0, len(longest)-8):] {
			fmt.Fprintf(&b, "    [%d, %d] %s\n", op.Call, op.Return, kvModel(1).DescribeOperation(op.Input, op.Output))
		}
		if stuck != nil {
			fmt.Fprintf(&b, "  cannot place: [%d, %d] %s\n  all operations:\n", stuck.Call, stuck.Return, kvModel(1).DescribeOperation(stuck.Input, stuck.Output))
		}
		for _, op := range kops {
			ret := fmt.Sprint(op.Return)
			if op.Return == math.MaxInt64 {
				ret = "ambiguous"
			}
			fmt.Fprintf(&b, "  [%6d, %9s] c%d %s\n", op.Call, ret, op.ClientId, kvModel(1).DescribeOperation(op.Input, op.Output))
		}
		return b.String()
	}
	return "no single key fails"
}
